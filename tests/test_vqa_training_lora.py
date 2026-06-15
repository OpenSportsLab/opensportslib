from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from opensportslib.core.trainer.vqa_trainer import (
    OptionalDependencyError,
    VQALoraTrainer,
    VQAXVarsVideoChatGPTLoraTrainer,
    build_vqa_sft_text,
)
from opensportslib.core.utils.hf_runtime import has_peft_adapter_artifacts


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
        TRAIN=SimpleNamespace(
            execution={
                "training_backend": "xvars_lora",
                "dry_run": dry_run,
                "prompt": {"include_priors": True, "prior_fields": ["action", "offence"]},
                "sft": {"include_video_tokens": True, "video_token_len": 2},
                "hf": {"model_id": "distilgpt2", "local_files_only": True, "prefer_cuda": False},
                "lora": {"target_modules": ["q_proj", "v_proj"]},
                "quantization": {"enabled": False},
                "checkpoint": {"save_adapter": True, "merge_and_save": False},
            }
        ),
    )


def test_build_vqa_sft_text_uses_priors_and_video_tokens():
    row = build_vqa_sft_text(
        _sample(),
        prompt_cfg={"include_priors": True, "prior_fields": ["action", "offence"]},
        sft_cfg={"include_video_tokens": True, "video_token_len": 2},
    )
    assert "USER: What card would you give?" in row["prompt"]
    assert "action=Challenge" in row["prompt"]
    assert "<vid_start><vid_patch><vid_patch><vid_end>" in row["prompt"]
    assert row["answer"] == "No card, because contact was low intensity."
    assert row["completion"] == row["answer"]
    assert row["text"].endswith(row["answer"])


def test_lora_trainer_dry_run_writes_checkpoint_artifacts(tmp_path):
    out = VQALoraTrainer(_cfg(tmp_path, dry_run=True)).train([_sample()], [_sample()])
    out_path = Path(out)
    assert (out_path / "config.yaml").exists()
    assert (out_path / "training_metadata.json").exists()
    assert (out_path / "adapter_model").exists()


def test_lora_trainer_filters_tokenization_mismatch_rows():
    class TinyTok:
        def __call__(self, text, truncation=True, max_length=32):
            toks = text.strip().split()
            if truncation:
                toks = toks[:max_length]
            return SimpleNamespace(input_ids=toks)

    rows = [
        {"prompt": "USER: q ASSISTANT:", "completion": "valid answer"},
        {"prompt": "USER: q ASSISTANT:", "completion": ""},
    ]
    kept, dropped = VQALoraTrainer._filter_tokenization_mismatch(rows, tokenizer=TinyTok(), max_seq_length=32)
    assert len(kept) == 1
    assert dropped == 1


def test_lora_trainer_missing_optional_dependency_is_actionable(tmp_path, monkeypatch):
    import opensportslib.core.trainer.vqa_trainer as mod

    monkeypatch.setattr(
        mod,
        "require_optional_package",
        lambda package, install_hint=None: (_ for _ in ()).throw(
            OptionalDependencyError("Install it with: pip install trl")
        ),
    )
    with pytest.raises(OptionalDependencyError, match="pip install trl"):
        VQALoraTrainer(_cfg(tmp_path, dry_run=False)).train([_sample()], [_sample()])


def test_peft_adapter_artifact_detection(tmp_path):
    ckpt = tmp_path / "adapter"
    ckpt.mkdir()
    assert not has_peft_adapter_artifacts(str(ckpt))

    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert has_peft_adapter_artifacts(str(ckpt))


def test_vqa_lora_trainer_enables_wandb_reporting(monkeypatch, tmp_path):
    import opensportslib.core.trainer.vqa_trainer as mod

    captured = {}

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured["report_to"] = kwargs.get("report_to")
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeDataset:
        @staticmethod
        def from_list(rows):
            return rows

    class FakeTrainer:
        def train(self):
            return None

        @property
        def model(self):
            class _Model:
                def save_pretrained(self, output_dir):
                    del output_dir

            return _Model()

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def save_pretrained(self, output_dir):
            del output_dir

        def __call__(self, text, truncation=True, max_length=512, padding="max_length"):
            del truncation, max_length, padding
            toks = list(range(1, len(text.split()) + 1))
            return {"input_ids": toks + [0] * max(0, 8 - len(toks)), "attention_mask": [1] * len(toks) + [0] * max(0, 8 - len(toks))}

    class FakeModel:
        pass

    monkeypatch.setattr(mod, "require_optional_package", lambda package, install_hint=None: None)
    monkeypatch.setitem(__import__("sys").modules, "datasets", SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(__import__("sys").modules, "transformers", SimpleNamespace(TrainingArguments=FakeTrainingArguments))
    monkeypatch.setattr(mod, "load_hf_causal_lm_for_training", lambda *args, **kwargs: (FakeTokenizer(), FakeModel(), "cpu"))
    monkeypatch.setattr(mod, "apply_lora_for_causal_lm", lambda model, lora_cfg, distributed=False: model)
    monkeypatch.setattr(mod, "build_trl_sft_trainer", lambda **kwargs: FakeTrainer())

    trainer = VQALoraTrainer(_cfg(tmp_path, dry_run=False))
    trainer.train([_sample()], [_sample()], use_wandb=True)

    assert captured["report_to"] == ["wandb"]


def test_xvars_videochatgpt_trainer_enables_wandb_reporting(monkeypatch, tmp_path):
    import opensportslib.core.trainer.vqa_trainer as mod

    captured = {}

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured["report_to"] = kwargs.get("report_to")

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

    class FakeBaseLM:
        pass

    class FakeModel:
        def save_pretrained(self, output_dir):
            del output_dir

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        def train(self):
            return None

    sample = _sample() | {"video_spatio_temporal_features": [[0.1] * 4, [0.2] * 4]}
    cfg = _cfg(tmp_path, dry_run=False)
    cfg.TRAIN.execution["training_backend"] = "xvars_videochatgpt_lora"
    cfg.TRAIN.execution["xvars"] = {
        "base_model": "fake",
        "video_token_len": 2,
        "mm_hidden_size": 4,
        "projection_path": None,
    }
    cfg.TRAIN.execution["sft"] = {
        "max_seq_length": 32,
        "video_token_len": 2,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 1,
        "learning_rate": 1e-4,
        "logging_steps": 1,
        "save_strategy": "no",
        "disable_tqdm": True,
    }

    monkeypatch.setattr(mod, "require_optional_package", lambda package, install_hint=None: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeBaseLM()),
            TrainingArguments=FakeTrainingArguments,
        ),
    )
    monkeypatch.setattr(mod, "build_bitsandbytes_config", lambda cfg: None)
    monkeypatch.setattr(mod, "apply_lora_for_causal_lm", lambda model, lora_cfg, distributed=False: model)
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
    trainer.train([sample], [sample], use_wandb=True)

    assert captured["report_to"] == ["wandb"]


def test_vqa_lora_train_checkpoint_round_trip(vqa_config_path, tmp_path):
    from opensportslib.apis import VQAModel

    cfg_path = Path(vqa_config_path)
    payload = yaml.safe_load(cfg_path.read_text())
    payload["SYSTEM"]["paths"]["save_dir"] = str(tmp_path / "vqa_roundtrip_ckpt")
    payload["SYSTEM"]["paths"]["work_dir"] = str(tmp_path / "vqa_roundtrip_ckpt")
    payload["TRAIN"]["execution"].update(
        {
            "training_backend": "xvars_lora",
            "dry_run": True,
            "prompt": {"include_priors": True, "prior_fields": ["action", "offence"]},
            "sft": {"include_video_tokens": True, "video_token_len": 2},
            "hf": {"model_id": "distilgpt2", "local_files_only": True, "prefer_cuda": False},
            "lora": {"target_modules": ["q_proj", "v_proj"]},
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

    loaded_api = VQAModel(config=str(roundtrip_cfg), weights=ckpt)
    predictions = loaded_api.infer(use_wandb=False)

    assert loaded_api.last_loaded_weights == ckpt
    assert predictions["task"] == "vqa"
    assert len(predictions["data"]) > 0
    assert predictions["data"][0]["answer_text"]
