import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - test env compatibility
    import yaml_compat as yaml

from opensportslib.core.trainer.vqa_trainer import (
    OptionalDependencyError,
    VQAXVarsVideoChatGPTLoraTrainer,
    build_vqa_sft_text,
)
from opensportslib.core.utils.hf_runtime import apply_lora_for_causal_lm, has_peft_adapter_artifacts
from opensportslib.models.base.xvars_videochatgpt import XVarsVideoChatGPTCausalLM


def _sample():
    return {
        "id": "action_0",
        "question": "What card would you give?",
        "references": ["No card, because contact was low intensity."],
        "labels": {
            "action": {"label": "Challenge"},
            "offence": {"label": "Offence: No card"},
        },
        "metadata": {"league": "TestLeague"},
    }


def _cfg(tmp_path, *, dry_run=True):
    return SimpleNamespace(
        SYSTEM=SimpleNamespace(paths=SimpleNamespace(save_dir=str(tmp_path / "ckpt"))),
        MODEL=SimpleNamespace(
            runtime=SimpleNamespace(dtype="fp32"),
            components=SimpleNamespace(
                llm_decoder=SimpleNamespace(
                    kind="decoder",
                    source=SimpleNamespace(provider="huggingface", name="distilgpt2"),
                    params=SimpleNamespace(repo_id="distilgpt2"),
                    overrides=SimpleNamespace(),
                )
            ),
        ),
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                splits=SimpleNamespace(
                    train=SimpleNamespace(dataloader=SimpleNamespace(batch_size=1)),
                    valid=SimpleNamespace(dataloader=SimpleNamespace(batch_size=1)),
                )
            )
        ),
        TRAIN=SimpleNamespace(
            epochs=1,
            optimizer=SimpleNamespace(type="AdamW", lr=1e-4),
            execution={
                "training_backend": "xvars_videochatgpt_lora",
                "dry_run": dry_run,
                "acc_grad_iter": 1,
                "log_interval": 1,
                "prompt": {"include_priors": True, "prior_fields": ["action", "offence"], "video_token_len": 2},
                "sft": {"include_video_tokens": True},
                "hf": {"local_files_only": True, "prefer_cuda": False},
                "lora": {"target_modules": ["q_proj", "v_proj"]},
                "quantization": {"enabled": False},
                "checkpoint": {"save_adapter": True, "merge_and_save": False},
            }
        ),
    )


def test_build_vqa_sft_text_uses_priors_and_video_tokens():
    row = build_vqa_sft_text(
        _sample(),
        config=_cfg(Path(".")),
        prompt_cfg={"include_priors": True, "prior_fields": ["action", "offence"]},
        sft_cfg={"include_video_tokens": True, "video_token_len": 2},
    )
    assert "USER: What card would you give?" in row["prompt"]
    assert "action=Challenge" in row["prompt"]
    assert "<vid_start><vid_patch><vid_patch><vid_end>" in row["prompt"]
    assert row["answer"] == "No card, because contact was low intensity."
    assert row["completion"] == f'{row["answer"]}</s>'
    assert row["text"].endswith(f'{row["answer"]}</s>')

def test_peft_adapter_artifact_detection(tmp_path):
    ckpt = tmp_path / "adapter"
    ckpt.mkdir()
    assert not has_peft_adapter_artifacts(str(ckpt))

    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert has_peft_adapter_artifacts(str(ckpt))


def test_xvars_videochatgpt_lora_wraps_with_peft_without_missing_generation_hook():
    pytest.importorskip("peft")

    class TinyTokenizer:
        def __init__(self):
            self.vocab = {"<vid_start>": 2, "<vid_patch>": 3, "<vid_end>": 4}

        def convert_tokens_to_ids(self, tok):
            return self.vocab[tok]

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, vocab_size=32, model_type="llama")
            self.emb = torch.nn.Embedding(64, 4)
            self.q_proj = torch.nn.Linear(4, 4)
            self.v_proj = torch.nn.Linear(4, 4)
            self.lm_head = torch.nn.Linear(4, 64)
            self.model = torch.nn.Module()
            self.model.mm_projector = torch.nn.Linear(1024, 4)
            self.gradient_checkpointing_kwargs = None

        def get_input_embeddings(self):
            return self.emb

        def resize_token_embeddings(self, size):
            del size
            return self.emb

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

        def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
            del input_ids, attention_mask, kwargs
            hidden = self.v_proj(self.q_proj(inputs_embeds))
            logits = self.lm_head(hidden)
            loss = logits.sum() * 0
            if labels is not None:
                loss = loss + 0.123
            return SimpleNamespace(loss=loss, logits=logits)

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
        ):
            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                **kwargs,
            }

    model = XVarsVideoChatGPTCausalLM(TinyCausalLM(), mm_hidden_size=1024)
    wrapped = apply_lora_for_causal_lm(model, {"target_modules": ["q_proj", "v_proj"]})
    wrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    input_ids = torch.tensor([[9, 2, 3, 3, 3, 4, 10]])
    features = torch.ones((1, 3, 1024))
    out = wrapped(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids,
        video_spatio_temporal_features=features,
        tokenizer=TinyTokenizer(),
    )

    assert hasattr(wrapped, "prepare_inputs_for_generation")
    assert model.base_lm.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert float(out.loss) > 0


def test_xvars_lora_excludes_inactive_embedded_projector():
    peft = pytest.importorskip("peft")

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, vocab_size=32, model_type="llama")
            self.emb = torch.nn.Embedding(64, 4)
            self.model = torch.nn.Module()
            self.model.mm_projector = torch.nn.Linear(1024, 4)

        def get_input_embeddings(self):
            return self.emb

        def resize_token_embeddings(self, size):
            del size
            return self.emb

    model = XVarsVideoChatGPTCausalLM(TinyCausalLM(), mm_hidden_size=1024)
    wrapped = apply_lora_for_causal_lm(
        model,
        {
            "target_modules": ["mm_projector"],
            "exclude_modules": r"^base_lm\.model\.mm_projector$",
        },
    )
    trainable = [name for name, param in wrapped.named_parameters() if param.requires_grad]

    assert any("base_model.model.mm_projector.lora_" in name for name in trainable)
    supports_exclude_modules = "exclude_modules" in inspect.signature(peft.LoraConfig.__init__).parameters
    if supports_exclude_modules:
        assert not any("base_lm.model.mm_projector.lora_" in name for name in trainable)
    else:
        assert any("base_lm.model.mm_projector.lora_" in name for name in trainable)


def test_apply_lora_for_causal_lm_skips_exclude_modules_for_older_peft(monkeypatch):
    import opensportslib.core.utils.hf_runtime as mod

    captured = {}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4)
            self.v_proj = torch.nn.Linear(4, 4)

        def named_modules(self, memo=None, prefix="", remove_duplicate=True):
            yield "", self
            yield "q_proj", self.q_proj
            yield "v_proj", self.v_proj

    class FakeLoraConfig:
        def __init__(self, r, lora_alpha, lora_dropout, bias, task_type, target_modules):
            captured["kwargs"] = {
                "r": r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "bias": bias,
                "task_type": task_type,
                "target_modules": target_modules,
            }

    monkeypatch.setattr(mod, "require_optional_package", lambda package, install_hint=None: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "peft",
        SimpleNamespace(
            LoraConfig=FakeLoraConfig,
            get_peft_model=lambda model, peft_config: ("wrapped", model, peft_config),
            prepare_model_for_kbit_training=lambda model, use_gradient_checkpointing=True: model,
        ),
    )

    wrapped = mod.apply_lora_for_causal_lm(
        TinyModel(),
        {
            "target_modules": ["q_proj", "v_proj"],
            "exclude_modules": r"^base_lm\.model\.mm_projector$",
        },
    )

    assert wrapped[0] == "wrapped"
    assert "exclude_modules" not in captured["kwargs"]


def test_xvars_videochatgpt_trainer_enables_wandb_reporting(monkeypatch, tmp_path):
    import opensportslib.core.trainer.vqa_trainer as mod

    captured = {}

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"
        pad_token_id = 0
        eos_token_id = 0

        def save_pretrained(self, output_dir):
            del output_dir

        def __call__(self, text, truncation=True, max_length=768, padding="max_length"):
            del truncation, max_length, padding
            toks = list(range(1, len(text.split()) + 1))
            return {"input_ids": toks + [0] * max(0, 8 - len(toks)), "attention_mask": [1] * len(toks) + [0] * max(0, 8 - len(toks))}

    class FakeModel:
        config = SimpleNamespace(use_cache=True)

        def save_pretrained(self, output_dir):
            del output_dir

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        def train(self):
            return None

    sample = _sample() | {"video_spatio_temporal_features": [[0.1] * 4] * 300}
    cfg = _cfg(tmp_path, dry_run=False)
    cfg.TRAIN.execution["training_backend"] = "xvars_videochatgpt_lora"
    cfg.MODEL = SimpleNamespace(
        runtime=SimpleNamespace(dtype="fp16"),
        components=SimpleNamespace(
            video_encoder=SimpleNamespace(params=SimpleNamespace(feature_source="indexed")),
            mm_projector=SimpleNamespace(params=SimpleNamespace(input_dim=4)),
            llm_decoder=SimpleNamespace(
                source=SimpleNamespace(provider="huggingface", name="base_model_videoChatGPT"),
                params=SimpleNamespace(repo_id="base_model_videoChatGPT"),
                overrides=SimpleNamespace(),
            ),
        ),
    )
    cfg.TRAIN.epochs = 3
    cfg.TRAIN.optimizer.lr = 2e-4
    cfg.TRAIN.optimizer.weight_decay = 0.001
    cfg.TRAIN.execution["acc_grad_iter"] = 1
    cfg.TRAIN.execution["log_interval"] = 1
    cfg.TRAIN.execution["prompt"]["video_token_len"] = 300
    cfg.TRAIN.execution["xvars"] = {
        "projection_path": None,
    }
    cfg.TRAIN.execution["sft"] = {
        "max_seq_length": 480,
        "max_steps": 50,
        "save_strategy": "epoch",
        "disable_tqdm": True,
        "gradient_checkpointing": True,
    }
    cfg.TRAIN.execution["lora"] = {
        "r": 16,
        "alpha": 32,
        "target_modules": [
            "mm_projector",
            "upsample_features",
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ],
    }

    monkeypatch.setattr(mod, "require_optional_package", lambda package, install_hint=None: None)
    captured_sources = {}

    def _load_tok(model_id, **kwargs):
        del kwargs
        captured_sources["tokenizer"] = model_id
        return FakeTokenizer()

    def _load_model(model_id, **kwargs):
        del kwargs
        captured_sources["model"] = model_id
        return object()

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=_load_tok),
            TrainingArguments=FakeTrainingArguments,
        ),
    )
    monkeypatch.setattr(mod, "build_bitsandbytes_config", lambda cfg: None)
    def _apply_lora(model, lora_cfg, distributed=False):
        captured["lora_cfg"] = dict(lora_cfg)
        captured["lora_distributed"] = distributed
        return model

    monkeypatch.setattr(mod, "apply_lora_for_causal_lm", _apply_lora)
    monkeypatch.setattr(mod, "load_videochatgpt_compatible_causal_lm", _load_model)
    monkeypatch.setattr(
        "opensportslib.core.utils.hf_runtime._ensure_video_special_tokens",
        lambda tokenizer, model=None: 0,
    )
    monkeypatch.setattr(
        mod.XVarsVideoChatGPTCausalLM,
        "from_pretrained_projector",
        staticmethod(lambda base_lm, projection_path, mm_hidden_size=1024: FakeModel()),
    )
    monkeypatch.setattr(mod, "XVarsVideoChatGPTTrainer", FakeTrainer)

    trainer = VQAXVarsVideoChatGPTLoraTrainer(cfg)
    trainer.train([sample], [sample], rank=0, world_size=4, use_wandb=True)

    assert captured["report_to"] == ["wandb"]
    assert captured["gradient_accumulation_steps"] == 1
    assert captured["num_train_epochs"] == 3
    assert captured["learning_rate"] == 2e-4
    assert captured["logging_steps"] == 1
    assert captured["optim"] == "paged_adamw_8bit"
    assert captured["weight_decay"] == 0.001
    assert captured["lr_scheduler_type"] == "constant"
    assert captured.get("eval_strategy", captured.get("evaluation_strategy")) == "epoch"
    assert captured["fp16"] is True
    assert captured["bf16"] is False
    assert captured["gradient_checkpointing"] is True
    assert captured["max_steps"] == 50
    assert captured["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert captured["ddp_find_unused_parameters"] is True
    assert captured["lora_cfg"]["target_modules"] == [
        "mm_projector",
        "upsample_features",
        "up_proj",
        "down_proj",
        "gate_proj",
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
    ]
    assert captured["lora_distributed"] is True
    assert FakeModel.config.use_cache is False
    assert captured_sources["model"] == "base_model_videoChatGPT"
    assert captured_sources["tokenizer"] == "base_model_videoChatGPT"


def test_vqa_xvars_videochatgpt_lora_single_process_omits_ddp_only_args(monkeypatch, tmp_path):
    import opensportslib.core.trainer.vqa_trainer as mod

    captured = {}

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"
        pad_token_id = 0
        eos_token_id = 0

        def save_pretrained(self, output_dir):
            del output_dir

        def __call__(self, text, truncation=True, max_length=768, padding="max_length"):
            del truncation, max_length, padding
            toks = list(range(1, len(text.split()) + 1))
            return {"input_ids": toks + [0] * max(0, 8 - len(toks)), "attention_mask": [1] * len(toks) + [0] * max(0, 8 - len(toks))}

    class FakeModel:
        config = SimpleNamespace(use_cache=True)

        def save_pretrained(self, output_dir):
            del output_dir

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        def train(self):
            return None

    sample = _sample() | {"video_spatio_temporal_features": [[0.1] * 4] * 300}
    cfg = _cfg(tmp_path, dry_run=False)
    cfg.TRAIN.execution["training_backend"] = "xvars_videochatgpt_lora"
    cfg.MODEL = SimpleNamespace(
        runtime=SimpleNamespace(dtype="fp16"),
        components=SimpleNamespace(
            video_encoder=SimpleNamespace(params=SimpleNamespace(feature_source="indexed")),
            mm_projector=SimpleNamespace(params=SimpleNamespace(input_dim=4)),
            llm_decoder=SimpleNamespace(
                source=SimpleNamespace(provider="huggingface", name="base_model_videoChatGPT"),
                params=SimpleNamespace(repo_id="base_model_videoChatGPT"),
                overrides=SimpleNamespace(),
            ),
        ),
    )
    cfg.TRAIN.epochs = 3
    cfg.TRAIN.optimizer.lr = 2e-4
    cfg.TRAIN.optimizer.weight_decay = 0.001
    cfg.TRAIN.execution["acc_grad_iter"] = 1
    cfg.TRAIN.execution["log_interval"] = 1
    cfg.TRAIN.execution["prompt"]["video_token_len"] = 300
    cfg.TRAIN.execution["xvars"] = {
        "projection_path": None,
    }
    cfg.TRAIN.execution["sft"] = {
        "max_seq_length": 480,
        "max_steps": 50,
        "save_strategy": "epoch",
        "disable_tqdm": True,
        "gradient_checkpointing": True,
    }
    cfg.TRAIN.execution["lora"] = {
        "r": 16,
        "alpha": 32,
        "target_modules": [
            "mm_projector",
            "upsample_features",
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ],
    }

    monkeypatch.setattr(mod, "require_optional_package", lambda package, install_hint=None: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda model_id, **kwargs: FakeTokenizer()),
            TrainingArguments=FakeTrainingArguments,
        ),
    )
    monkeypatch.setattr(mod, "build_bitsandbytes_config", lambda cfg: None)
    monkeypatch.setattr(mod, "apply_lora_for_causal_lm", lambda model, lora_cfg, distributed=False: model)
    monkeypatch.setattr(mod, "load_videochatgpt_compatible_causal_lm", lambda model_id, **kwargs: object())
    monkeypatch.setattr(
        "opensportslib.core.utils.hf_runtime._ensure_video_special_tokens",
        lambda tokenizer, model=None: 0,
    )
    monkeypatch.setattr(
        mod.XVarsVideoChatGPTCausalLM,
        "from_pretrained_projector",
        staticmethod(lambda base_lm, projection_path, mm_hidden_size=1024: FakeModel()),
    )
    monkeypatch.setattr(mod, "XVarsVideoChatGPTTrainer", FakeTrainer)

    trainer = VQAXVarsVideoChatGPTLoraTrainer(cfg)
    trainer.train([sample], [sample], rank=0, world_size=1, use_wandb=False)

    assert captured["gradient_checkpointing"] is True
    assert "ddp_find_unused_parameters" not in captured
    assert "gradient_checkpointing_kwargs" not in captured


def test_vqa_lora_train_checkpoint_round_trip(vqa_config_path, tmp_path):
    from opensportslib.apis import VQAModel

    cfg_path = Path(vqa_config_path)
    payload = yaml.safe_load(cfg_path.read_text())
    payload["SYSTEM"]["paths"]["save_dir"] = str(tmp_path / "vqa_roundtrip_ckpt")
    payload["SYSTEM"]["paths"]["work_dir"] = str(tmp_path / "vqa_roundtrip_ckpt")
    payload["TRAIN"]["execution"].update(
        {
            "training_backend": "xvars_videochatgpt_lora",
            "dry_run": True,
            "prompt": {"include_priors": True, "prior_fields": ["action", "offence"]},
            "sft": {"include_video_tokens": True, "video_token_len": 2},
            "hf": {"model_id": "base_model_videoChatGPT", "local_files_only": True, "prefer_cuda": False},
            "lora": {"target_modules": ["mm_projector", "q_proj", "v_proj"]},
            "quantization": {"enabled": False},
            "checkpoint": {"save_adapter": True, "merge_and_save": False},
        }
    )
    roundtrip_cfg = tmp_path / "vqa_roundtrip.yaml"
    roundtrip_cfg.write_text(yaml.safe_dump(payload), encoding="utf-8")

    api = VQAModel(config=str(roundtrip_cfg))
    ckpt = api.train(use_wandb=False)
    ckpt_path = Path(ckpt)
    assert (ckpt_path / "config.yaml").exists()
    assert (ckpt_path / "training_metadata.json").exists()

    loaded_api = VQAModel(config=str(roundtrip_cfg))
    loaded_api.load_weights(ckpt)

    assert loaded_api.last_loaded_weights == ckpt
    assert loaded_api.best_checkpoint == ckpt
