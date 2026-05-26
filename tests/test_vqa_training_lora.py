from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from opensportslib.core.trainer.vqa_trainer import (
    OptionalDependencyError,
    VQALoraTrainer,
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
