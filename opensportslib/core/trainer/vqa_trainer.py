"""Trainer/infer/eval helpers for VQA task."""

from __future__ import annotations

import json
import os
from typing import Any

from opensportslib.core.config.accessors import (
    get_system_path,
    get_train_execution,
    get_vqa_generation_cfg,
)
from opensportslib.core.utils.config import save_config
from opensportslib.core.utils.hf_runtime import (
    OptionalDependencyError,
    apply_lora_for_causal_lm,
    load_hf_causal_lm_for_training,
    require_optional_package,
)
from opensportslib.metrics.vqa_metric import compute_vqa_metrics
from opensportslib.models.utils.vqa_prompting import build_prior_text


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items()}
    return {}


def build_vqa_sft_text(
    sample: dict[str, Any],
    *,
    prompt_cfg: dict[str, Any] | None = None,
    sft_cfg: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Convert a VQADataset sample into prompt/answer/text fields for SFT."""
    prompt_cfg = prompt_cfg or {}
    sft_cfg = sft_cfg or {}
    refs = sample.get("references") or []
    answer = str(refs[0]).strip() if refs else ""
    question = str(sample.get("question", "")).strip()
    system_prompt = str(
        prompt_cfg.get(
            "system_prompt",
            "You are an artificial intelligence assistant for visual football referee questions.",
        )
    ).strip()

    prior_text = ""
    if bool(prompt_cfg.get("include_priors", True)):
        prior_text = build_prior_text(
            sample.get("labels", {}) or {},
            sample.get("metadata", {}) or {},
            include_fields=prompt_cfg.get("prior_fields"),
        )

    video_token = ""
    if bool(sft_cfg.get("include_video_tokens", True)):
        token_len = int(sft_cfg.get("video_token_len", 300))
        video_token = "<vid_start>" + ("<vid_patch>" * token_len) + "<vid_end>"

    prompt_parts = [system_prompt, f"USER: {question}"]
    if prior_text:
        prompt_parts.append(f"The prediction for this video is {prior_text}.")
    if video_token:
        prompt_parts.append(video_token)
    prompt_parts.append("ASSISTANT:")
    prompt = "\n".join(prompt_parts)
    return {"prompt": prompt, "answer": answer, "text": f"{prompt} {answer}".strip()}


class VQALoraSFTDataset:
    """Small adapter expected by TRL SFTTrainer."""

    def __init__(self, dataset, prompt_cfg: dict[str, Any], sft_cfg: dict[str, Any]):
        self.rows = [
            build_vqa_sft_text(sample, prompt_cfg=prompt_cfg, sft_cfg=sft_cfg)
            for sample in dataset
            if sample.get("references")
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class VQALoraTrainer:
    """LoRA/QLoRA trainer wrapper for VQA SFT."""

    def __init__(self, config):
        self.config = config

    def train(self, train_data, valid_data=None) -> str:
        execution = get_train_execution(self.config)
        prompt_cfg = _as_dict(execution.get("prompt"))
        sft_cfg = _as_dict(execution.get("sft"))
        hf_cfg = _as_dict(execution.get("hf"))
        lora_cfg = _as_dict(execution.get("lora"))
        quant_cfg = _as_dict(execution.get("quantization"))
        checkpoint_cfg = _as_dict(execution.get("checkpoint"))

        save_root = get_system_path(self.config, "save_dir", "./checkpoints") or "./checkpoints"
        output_dir = os.path.join(save_root, "xvars_lora")
        os.makedirs(output_dir, exist_ok=True)

        train_sft = VQALoraSFTDataset(train_data, prompt_cfg, sft_cfg)
        valid_sft = VQALoraSFTDataset(valid_data, prompt_cfg, sft_cfg) if valid_data is not None else None

        metadata = {
            "backend": "xvars_lora",
            "num_train_samples": len(train_sft),
            "num_valid_samples": len(valid_sft) if valid_sft is not None else 0,
            "status": "metadata_only",
        }

        if bool(execution.get("dry_run", False)):
            return self._write_artifacts(output_dir, metadata)

        require_optional_package("trl", "pip install trl")
        from transformers import TrainingArguments
        from trl import SFTTrainer

        model_id = str(hf_cfg.get("model_id", "distilgpt2"))
        tokenizer, model, _device = load_hf_causal_lm_for_training(
            model_id,
            local_files_only=bool(hf_cfg.get("local_files_only", False)),
            prefer_cuda=bool(hf_cfg.get("prefer_cuda", True)),
            quantization_cfg=quant_cfg,
        )
        model = apply_lora_for_causal_lm(model, lora_cfg)

        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=int(sft_cfg.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(sft_cfg.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(sft_cfg.get("gradient_accumulation_steps", 1)),
            num_train_epochs=float(sft_cfg.get("num_train_epochs", 1)),
            learning_rate=float(sft_cfg.get("learning_rate", 2e-4)),
            logging_steps=int(sft_cfg.get("logging_steps", 1)),
            save_strategy=str(sft_cfg.get("save_strategy", "epoch")),
            report_to=[],
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_sft,
            eval_dataset=valid_sft,
            tokenizer=tokenizer,
            args=args,
            dataset_text_field="text",
            max_seq_length=int(sft_cfg.get("max_seq_length", 512)),
        )
        trainer.train()

        if bool(checkpoint_cfg.get("save_adapter", True)):
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        metadata["status"] = "trained"
        return self._write_artifacts(output_dir, metadata)

    def _write_artifacts(self, output_dir: str, metadata: dict[str, Any]) -> str:
        save_config(self.config, os.path.join(output_dir, "config.yaml"))
        with open(os.path.join(output_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        marker = os.path.join(output_dir, "adapter_model")
        os.makedirs(marker, exist_ok=True)
        with open(os.path.join(marker, "README.txt"), "w", encoding="utf-8") as f:
            f.write("VQA LoRA adapter artifacts are stored in the parent directory when training runs.\n")
        return output_dir


class Trainer_VQA:
    """VQA trainer dispatcher plus inference/evaluation helpers."""

    def __init__(self, config):
        self.config = config
        self.best_checkpoint_path: str | None = None
        self.loaded_checkpoint_metadata: dict[str, Any] | None = None

    def load(self, weights: str):
        # VQA LoRA checkpoints are directory-based adapter artifacts.
        metadata_path = os.path.join(weights, "training_metadata.json") if os.path.isdir(weights) else ""
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                self.loaded_checkpoint_metadata = json.load(f)
        self.best_checkpoint_path = weights
        return weights

    def train(self, model, train_data, valid_data=None) -> str:
        del model
        execution = get_train_execution(self.config)
        backend = str(execution.get("training_backend", "placeholder")).lower()
        if backend == "xvars_lora":
            ckpt = VQALoraTrainer(self.config).train(train_data, valid_data)
            self.best_checkpoint_path = ckpt
            return ckpt

        del train_data, valid_data
        save_dir = get_system_path(self.config, "save_dir", "./checkpoints") or "./checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        ckpt = os.path.join(save_dir, "vqa-best.ckpt")
        with open(ckpt, "w", encoding="utf-8") as f:
            f.write("vqa placeholder checkpoint")
        self.best_checkpoint_path = ckpt
        return ckpt

    def infer(self, model, dataset) -> dict[str, Any]:
        exec_cfg = get_train_execution(self.config)
        prompt_cfg = exec_cfg.get("prompt", {}) if isinstance(exec_cfg, dict) else {}
        generation_cfg = get_vqa_generation_cfg(self.config)

        preds = []
        for sample in dataset:
            answer = model.generate_answer(
                sample,
                prompt_cfg=prompt_cfg,
                generation_cfg=generation_cfg,
            )
            preds.append(
                {
                    "id": sample.get("id"),
                    "question": sample.get("question"),
                    "answer_text": answer,
                    "video_path": sample.get("video_path"),
                }
            )
        return {"task": "vqa", "data": preds}

    def evaluate(self, predictions: dict[str, Any], dataset) -> dict[str, Any]:
        return compute_vqa_metrics(predictions, dataset)


__all__ = [
    "OptionalDependencyError",
    "Trainer_VQA",
    "VQALoraSFTDataset",
    "VQALoraTrainer",
    "build_vqa_sft_text",
]
