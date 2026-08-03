import inspect
import json
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
    VQANativeQwenVLSFTDataset,
    VQAQwenVLNativeLoraTrainer,
    Trainer_VQA,
    VQAQwenXVarsLoraTrainer,
    VQAXVarsVideoChatGPTSFTDataset,
    VQAXVarsVideoChatGPTLoraTrainer,
    _prepare_hf_resume_checkpoint,
    build_vqa_sft_text,
)
from opensportslib.core.utils.hf_runtime import apply_lora_for_causal_lm, has_peft_adapter_artifacts
from opensportslib.models.base.qwen_vl_native import NativeQwenVLInvalidRowError, NativeQwenVLTrainer
from opensportslib.models.base.qwen_xvars import QwenXVarsCausalLM
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


def _qwen_cfg(tmp_path, *, dry_run=True):
    cfg = _cfg(tmp_path, dry_run=dry_run)
    cfg.MODEL.components.llm_decoder.source.name = "Qwen/Qwen2.5-7B-Instruct"
    cfg.MODEL.components.llm_decoder.params.repo_id = "Qwen/Qwen2.5-7B-Instruct"
    cfg.MODEL.metadata = SimpleNamespace(backend="qwen_xvars_infer")
    cfg.TRAIN.execution["training_backend"] = "qwen_xvars_lora"
    cfg.TRAIN.execution["lora"] = {"target_modules": ["q_proj", "v_proj"]}
    return cfg


def _qwen_vl_cfg(tmp_path, *, dry_run=True, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    cfg = _cfg(tmp_path, dry_run=dry_run)
    cfg.MODEL.components.video_encoder = SimpleNamespace(
        kind="encoder",
        params=SimpleNamespace(
            feature_source="raw_video",
            native_vl=SimpleNamespace(visual_input_mode="frames", num_frames=4),
        ),
    )
    cfg.MODEL.components.llm_decoder.source.name = model_id
    cfg.MODEL.components.llm_decoder.params.repo_id = model_id
    cfg.MODEL.metadata = SimpleNamespace(backend="qwen_vl_native_infer")
    cfg.TRAIN.execution["training_backend"] = "qwen_vl_native_lora"
    cfg.TRAIN.execution["native_vl"] = {"visual_input_mode": "frames", "num_frames": 4}
    cfg.TRAIN.execution["lora"] = {"target_modules": ["q_proj", "v_proj"]}
    return cfg


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


def test_qwen_sft_dataset_emits_masked_labels_and_features(tmp_path):
    cfg = _qwen_cfg(tmp_path)

    class TinyTokenizer:
        eos_token = "</s>"
        eos_token_id = 1

        def __call__(self, text, truncation=True, max_length=32, padding="max_length"):
            del truncation, padding
            tokens = text.split()
            ids = list(range(2, 2 + min(len(tokens), max_length)))
            attn = [1] * len(ids)
            if len(ids) < max_length:
                pad = [0] * (max_length - len(ids))
                ids.extend(pad)
                attn.extend(pad)
            return {"input_ids": ids, "attention_mask": attn}

    dataset = VQAXVarsVideoChatGPTSFTDataset(
        [
            {
                **_sample(),
                "video_spatio_temporal_features": torch.ones((2, 1024)),
            }
        ],
        config=cfg,
        tokenizer=TinyTokenizer(),
        prompt_cfg={"include_priors": True, "video_token_len": 2},
        sft_cfg={"max_seq_length": 32, "include_video_tokens": True},
        xvars_cfg={},
    )

    row = dataset[0]
    assert tuple(row["video_spatio_temporal_features"].shape) == (2, 1024)
    assert any(label == -100 for label in row["labels"])
    assert any(label != -100 for label in row["labels"])

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


def test_qwen_xvars_lora_wraps_with_peft_when_transformers_reads_base_model_prefix():
    pytest.importorskip("peft")

    class TinyTokenizer:
        def __init__(self):
            self.vocab = {"<vid_start>": 2, "<vid_patch>": 3, "<vid_end>": 4}

        def convert_tokens_to_ids(self, tok):
            return self.vocab[tok]

    class TinyQwenLM(torch.nn.Module):
        base_model_prefix = "model"

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, vocab_size=32, model_type="qwen2")
            self.emb = torch.nn.Embedding(64, 4)
            self.q_proj = torch.nn.Linear(4, 4)
            self.v_proj = torch.nn.Linear(4, 4)
            self.lm_head = torch.nn.Linear(4, 64)
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

    model = QwenXVarsCausalLM(TinyQwenLM(), mm_hidden_size=1024)
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
    assert model.base_model_prefix == "model"
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


def test_qwen_lora_trainer_dry_run_writes_metadata(tmp_path):
    cfg = _qwen_cfg(tmp_path, dry_run=True)
    trainer = VQAQwenXVarsLoraTrainer(cfg)
    ckpt = trainer.train(
        [{"video_spatio_temporal_features": torch.ones((2, 1024))}],
        [{"video_spatio_temporal_features": torch.ones((2, 1024))}],
        use_wandb=False,
    )

    metadata_path = Path(ckpt) / "training_metadata.json"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    assert metadata["backend"] == "qwen_xvars_lora"
    assert metadata["num_train_samples"] == 1
    assert metadata["num_valid_samples"] == 1


def test_trainer_vqa_dispatches_qwen_backend_dry_run(tmp_path):
    cfg = _qwen_cfg(tmp_path, dry_run=True)
    trainer = Trainer_VQA(cfg)

    ckpt = trainer.train(
        None,
        [{"video_spatio_temporal_features": torch.ones((2, 1024))}],
        [{"video_spatio_temporal_features": torch.ones((2, 1024))}],
        use_wandb=False,
    )

    assert ckpt.endswith("qwen_xvars_lora")


def test_native_qwen_vl_sft_dataset_emits_masked_labels(tmp_path):
    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
            del tokenize
            return f"prompt::{len(messages)}::{int(add_generation_prompt)}"

        def __call__(self, text=None, images=None, videos=None, padding=None, truncation=None, max_length=None, return_tensors=None):
            del images, videos, padding, truncation, max_length, return_tensors
            size = 10 if "assistant" not in text[0] else 14
            return {
                "input_ids": torch.arange(size).unsqueeze(0),
                "attention_mask": torch.ones((1, size), dtype=torch.long),
            }

    fake_model = SimpleNamespace(
        _ready=True,
        processor=FakeProcessor(),
        _resolve_visual_inputs=lambda sample: ([torch.zeros((2, 2, 3))], "frames"),
        _build_messages=lambda sample, prompt_cfg=None, visual_type="frames": [{"role": "user", "content": [{"type": visual_type}, {"type": "text", "text": sample["question"]}]}],
    )
    fake_model.build_training_inputs = lambda sample, prompt_cfg=None, answer_text="", max_seq_length=1024: {
        "input_ids": torch.arange(12),
        "attention_mask": torch.ones(12, dtype=torch.long),
        "labels": torch.tensor([-100] * 5 + list(range(7))),
        "pixel_values": torch.ones((3, 2, 2)),
    }

    dataset = VQANativeQwenVLSFTDataset(
        [
            {
                **_sample(),
                "video_path": str(tmp_path / "clip.mp4"),
            }
        ],
        model=fake_model,
        prompt_cfg={},
        sft_cfg={"max_seq_length": 32},
    )
    row = dataset[0]
    assert any(label == -100 for label in row["labels"])
    assert any(label != -100 for label in row["labels"])


def test_native_qwen_vl_sft_dataset_caches_materialized_rows(tmp_path):
    calls = {"count": 0}

    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
            del messages, add_generation_prompt, tokenize
            return "prompt"

    def _build_training_inputs(sample, prompt_cfg=None, answer_text="", max_seq_length=1024):
        del sample, prompt_cfg, answer_text, max_seq_length
        calls["count"] += 1
        return {
            "input_ids": torch.arange(6),
            "attention_mask": torch.ones(6, dtype=torch.long),
            "labels": torch.tensor([-100, -100, 2, 3, 4, 5]),
            "pixel_values": torch.ones((3, 2, 2)),
        }

    fake_model = SimpleNamespace(
        _ready=True,
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        visual_input_mode="frames",
        num_frames=4,
        processor=FakeProcessor(),
        build_training_inputs=_build_training_inputs,
    )

    rows = [
        {
            **_sample(),
            "video_path": str(tmp_path / "clip.mp4"),
        }
    ]
    cache_dir = tmp_path / "native_cache"
    dataset = VQANativeQwenVLSFTDataset(
        rows,
        model=fake_model,
        prompt_cfg={},
        sft_cfg={"max_seq_length": 32},
        cache_dir=str(cache_dir),
        split_name="train",
    )

    first = dataset[0]
    assert calls["count"] == 1
    assert any(cache_dir.glob("*.pt"))
    second = dataset[0]
    assert calls["count"] == 1
    assert torch.equal(first["input_ids"], second["input_ids"])

    dataset_reloaded = VQANativeQwenVLSFTDataset(
        rows,
        model=fake_model,
        prompt_cfg={},
        sft_cfg={"max_seq_length": 32},
        cache_dir=str(cache_dir),
        split_name="train",
    )
    cached = dataset_reloaded[0]
    assert calls["count"] == 1
    assert torch.equal(cached["labels"], first["labels"])


def test_native_qwen_vl_trainer_forwards_resume_checkpoint(monkeypatch, tmp_path):
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def train(self, resume_from_checkpoint=None):
            captured["resume_from_checkpoint"] = resume_from_checkpoint
            return "trained"

        def save_state(self):
            captured["save_state"] = True

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(Trainer=FakeTrainer),
    )

    trainer = NativeQwenVLTrainer(
        model=SimpleNamespace(),
        args=SimpleNamespace(output_dir=str(tmp_path)),
        train_dataset=[],
        eval_dataset=None,
    )
    out = trainer.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2072"))

    assert out == "trained"
    assert captured["resume_from_checkpoint"] == str(tmp_path / "checkpoint-2072")


def test_native_qwen_vl_trainer_warns_once_for_nan_grad_norm_with_finite_gradients(monkeypatch, caplog, tmp_path):
    records = []

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        def training_step(self, model, inputs, *args, **kwargs):
            del inputs, args, kwargs
            for _, param in model.named_parameters():
                param.grad = torch.ones_like(param)
            return torch.tensor(0.5)

        def log(self, logs, *args, **kwargs):
            del args, kwargs
            records.append(dict(logs))

        def train(self, resume_from_checkpoint=None):
            del resume_from_checkpoint
            return "trained"

        def save_state(self):
            return None

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def named_parameters(self, *args, **kwargs):
            return super().named_parameters(*args, **kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(Trainer=FakeTrainer),
    )

    trainer = NativeQwenVLTrainer(
        model=TinyModel(),
        args=SimpleNamespace(output_dir=str(tmp_path)),
        train_dataset=[],
        eval_dataset=None,
    )

    caplog.set_level("WARNING")
    trainer._trainer.training_step(trainer.model, {})
    trainer._trainer.log({"loss": 0.5, "grad_norm": float("nan")})
    trainer._trainer.log({"loss": 0.4, "grad_norm": float("nan")})

    assert any("grad_norm=nan" in record.getMessage() for record in caplog.records)
    assert len([record for record in caplog.records if "grad_norm=nan" in record.getMessage()]) == 1
    assert records[0]["grad_norm"] != records[0]["grad_norm"]


def test_prepare_hf_resume_checkpoint_rewrites_training_args(tmp_path):
    source = tmp_path / "checkpoint-2072"
    output_dir = tmp_path / "run"
    source.mkdir()
    output_dir.mkdir()
    (source / "trainer_state.json").write_text("{}", encoding="utf-8")
    (source / "optimizer.pt").write_bytes(b"old-optimizer")
    (source / "scheduler.pt").write_bytes(b"old-scheduler")
    (source / "scaler.pt").write_bytes(b"old-scaler")
    torch.save(SimpleNamespace(ddp_find_unused_parameters=True), source / "training_args.bin")
    stale_target = output_dir / "_resume_sanitized" / "checkpoint-2072"
    stale_target.mkdir(parents=True)
    (stale_target / "optimizer.pt").write_bytes(b"stale-optimizer")
    (stale_target / "scheduler.pt").write_bytes(b"stale-scheduler")
    (stale_target / "scaler.pt").write_bytes(b"stale-scaler")

    current_args = SimpleNamespace(ddp_find_unused_parameters=False, eval_strategy="no")
    sanitized = _prepare_hf_resume_checkpoint(
        str(source),
        current_args,
        str(output_dir),
        resume_optimizer_state=False,
    )

    assert sanitized is not None
    assert Path(sanitized).name == "checkpoint-2072"
    assert Path(sanitized).parent.name == "_resume_sanitized"
    assert (Path(sanitized) / "trainer_state.json").exists()
    assert not (Path(sanitized) / "optimizer.pt").exists()
    assert not (Path(sanitized) / "scheduler.pt").exists()
    assert not (Path(sanitized) / "scaler.pt").exists()
    loaded = torch.load(Path(sanitized) / "training_args.bin", map_location="cpu", weights_only=False)
    assert loaded.ddp_find_unused_parameters is False
    assert loaded.eval_strategy == "no"


def test_qwen_vl_native_lora_trainer_dry_run_writes_metadata(tmp_path):
    cfg = _qwen_vl_cfg(tmp_path, dry_run=True)
    trainer = VQAQwenVLNativeLoraTrainer(cfg)
    ckpt = trainer.train(
        [{"video_path": str(tmp_path / "clip.mp4")}],
        [{"frame_paths": [str(tmp_path / "frame.jpg")]}],
        use_wandb=False,
    )

    metadata_path = Path(ckpt) / "training_metadata.json"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    assert metadata["backend"] == "qwen_vl_native_lora"
    assert metadata["num_train_samples"] == 1
    assert metadata["num_valid_samples"] == 1


def test_qwen_vl_native_lora_rejects_awq_training(tmp_path):
    cfg = _qwen_vl_cfg(tmp_path, dry_run=False, model_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    trainer = VQAQwenVLNativeLoraTrainer(cfg)
    with pytest.raises(ValueError, match="inference-only"):
        trainer.train([{"video_path": str(tmp_path / "clip.mp4"), "references": ["yes"]}], use_wandb=False)


def test_native_qwen_vl_dataset_skips_and_reports_invalid_rows(tmp_path):
    class FakeNativeModel:
        def build_training_inputs(self, sample, **kwargs):
            del kwargs
            if str(sample.get("id")) == "bad":
                raise NativeQwenVLInvalidRowError(
                    "Native Qwen VL training row has all labels masked after prompt and padding masking.",
                    context={"prompt_length": 32, "full_length": 32, "max_seq_length": 32},
                )
            return {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
            }

    rows = [
        {"id": "bad", "question": "q1", "references": ["a"], "video_path": str(tmp_path / "bad.mp4")},
        {"id": "good", "question": "q2", "references": ["b"], "video_path": str(tmp_path / "good.mp4")},
    ]
    report_path = tmp_path / "invalid_rows.json"
    dataset = VQANativeQwenVLSFTDataset(
        rows,
        model=FakeNativeModel(),
        prompt_cfg={},
        sft_cfg={"max_seq_length": 32, "disable_tqdm": True},
        split_name="train",
        invalid_row_report_path=str(report_path),
        fail_on_invalid=False,
        rank=0,
    )

    row = dataset[0]

    assert row["id"] == "good"
    assert dataset.invalid_row_count == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["count"] == 1
    assert report["rows"][0]["sample_id"] == "bad"


def test_native_qwen_vl_dataset_fail_fast_on_invalid_rows_for_ddp(tmp_path):
    class FakeNativeModel:
        def build_training_inputs(self, sample, **kwargs):
            del sample, kwargs
            raise NativeQwenVLInvalidRowError(
                "Native Qwen VL training row lost assistant supervision after truncation (full_len=16, prompt_len=16).",
                context={"prompt_length": 16, "full_length": 16, "max_seq_length": 16},
            )

    dataset = VQANativeQwenVLSFTDataset(
        [{"id": "bad", "question": "q", "references": ["a"], "video_path": str(tmp_path / "bad.mp4")}],
        model=FakeNativeModel(),
        prompt_cfg={},
        sft_cfg={"max_seq_length": 16, "disable_tqdm": True},
        split_name="train",
        fail_on_invalid=True,
        rank=2,
    )

    with pytest.raises(RuntimeError, match="rank=2"):
        dataset[0]


def test_native_qwen_vl_dataset_disables_memory_cache_when_configured(tmp_path):
    calls = {"count": 0}

    class FakeNativeModel:
        def build_training_inputs(self, sample, **kwargs):
            del sample, kwargs
            calls["count"] += 1
            return {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
            }

    dataset = VQANativeQwenVLSFTDataset(
        [{"id": "good", "question": "q", "references": ["a"], "video_path": str(tmp_path / "good.mp4")}],
        model=FakeNativeModel(),
        prompt_cfg={},
        sft_cfg={"max_seq_length": 32, "disable_tqdm": True, "memory_cache_rows": 0},
        split_name="train",
    )

    _ = dataset[0]
    assert calls["count"] == 1
    assert len(dataset._memory_cache) == 0


def test_native_qwen_vl_dataset_bounds_memory_cache(tmp_path):
    calls = {"count": 0}

    class FakeNativeModel:
        def build_training_inputs(self, sample, **kwargs):
            del kwargs
            calls["count"] += 1
            token = int(str(sample.get("id", "0")).replace("row_", "") or 0)
            return {
                "input_ids": torch.tensor([token, token + 1], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, token + 1], dtype=torch.long),
            }

    rows = [
        {"id": "row_0", "question": "q0", "references": ["a"], "video_path": str(tmp_path / "0.mp4")},
        {"id": "row_1", "question": "q1", "references": ["b"], "video_path": str(tmp_path / "1.mp4")},
        {"id": "row_2", "question": "q2", "references": ["c"], "video_path": str(tmp_path / "2.mp4")},
    ]
    dataset = VQANativeQwenVLSFTDataset(
        rows,
        model=FakeNativeModel(),
        prompt_cfg={},
        sft_cfg={"max_seq_length": 32, "disable_tqdm": True, "memory_cache_rows": 2},
        split_name="train",
    )

    _ = dataset[0]
    _ = dataset[1]
    _ = dataset[2]

    assert calls["count"] == 3
    assert dataset.memory_cache_rows == 2
    assert len(dataset._memory_cache) == 2
    assert 0 not in dataset._memory_cache
    assert 1 in dataset._memory_cache
    assert 2 in dataset._memory_cache


def test_qwen_vl_native_lora_ddp_can_disable_unused_parameter_detection(monkeypatch, tmp_path):
    import opensportslib.core.trainer.vqa_trainer as mod

    captured = {}
    cfg = _qwen_vl_cfg(tmp_path, dry_run=False)
    cfg.TRAIN.execution["sft"] = {
        "max_seq_length": 32,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "memory_cache_rows": 0,
        "ddp_find_unused_parameters": False,
        "ddp_broadcast_buffers": False,
        "cache_tokenized_rows": False,
        "evaluation_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,
    }

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeModel:
        config = SimpleNamespace(use_cache=True)

        def save_pretrained(self, output_dir):
            captured["saved"] = output_dir

    class FakeProcessor:
        def save_pretrained(self, output_dir):
            captured["processor_saved"] = output_dir

    class FakeNativeModel:
        def __init__(self, config, model_id):
            del config
            self.model_id = model_id
            self._ready = True
            self._error = None
            self.model = FakeModel()
            self.processor = FakeProcessor()
            self.visual_input_mode = "frames"
            self.num_frames = 4

        def prepare_training_sample(self, sample):
            resolved = dict(sample)
            resolved["video_frames"] = [torch.zeros((2, 2, 3))]
            resolved["frame_paths"] = []
            return resolved

    class FakeNativeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs
            self.model = kwargs["model"]

        def train(self, resume_from_checkpoint=None):
            captured["resume_from_checkpoint"] = resume_from_checkpoint

    monkeypatch.setattr(mod, "require_optional_package", lambda package, install_hint=None: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(TrainingArguments=FakeTrainingArguments),
    )
    monkeypatch.setattr(mod, "QwenVLNativeModel", FakeNativeModel)
    monkeypatch.setattr(mod, "apply_lora_for_causal_lm", lambda model, lora_cfg, distributed=False: model)
    monkeypatch.setattr(mod, "NativeQwenVLTrainer", FakeNativeTrainer)

    trainer = VQAQwenVLNativeLoraTrainer(cfg)
    trainer.train(
        [{"video_path": str(tmp_path / "clip.mp4"), "references": ["yes"]}],
        rank=0,
        world_size=4,
        use_wandb=False,
    )

    assert captured["ddp_find_unused_parameters"] is False
    assert captured["ddp_broadcast_buffers"] is False
    assert captured["max_grad_norm"] == 1.0
    assert captured["learning_rate"] == 1e-4
    assert captured["optim"] == "adamw_torch"
    assert captured["weight_decay"] == 0.001
    assert captured["save_strategy"] == "steps"
    assert captured["save_steps"] == 100
    assert captured["save_total_limit"] == 3
    assert captured["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert captured["resume_from_checkpoint"] is None
    assert captured["trainer_kwargs"]["train_dataset"].cache_dir == ""
    assert captured["trainer_kwargs"]["train_dataset"].memory_cache_rows == 0


def test_trainer_vqa_dispatches_qwen_vl_native_backend_dry_run(tmp_path):
    cfg = _qwen_vl_cfg(tmp_path, dry_run=True)
    trainer = Trainer_VQA(cfg)

    ckpt = trainer.train(
        None,
        [{"video_path": str(tmp_path / "clip.mp4")}],
        [{"video_path": str(tmp_path / "clip_valid.mp4")}],
        use_wandb=False,
    )

    assert ckpt.endswith("qwen_vl_native_lora")


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
