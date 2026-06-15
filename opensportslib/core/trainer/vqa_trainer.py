"""Trainer/infer/eval helpers for VQA task."""

from __future__ import annotations

import json
import logging
import os
import inspect
from typing import Any

import torch

from opensportslib.core.config.accessors import (
    get_split_dataloader_cfg,
    get_system_path,
    get_train_execution,
    get_vqa_eval_profile_cfg,
    get_vqa_generation_cfg,
)
from opensportslib.core.utils.config import save_config
from opensportslib.core.utils.hf_runtime import (
    OptionalDependencyError,
    apply_lora_for_causal_lm,
    build_bitsandbytes_config,
    build_trl_sft_trainer,
    load_hf_causal_lm_for_training,
    optional_package_available,
    require_optional_package,
)
from opensportslib.metrics.vqa_metric import compute_vqa_metrics
from opensportslib.models.base.xvars_videochatgpt import (
    DEFAULT_XVARS_TARGET_MODULES,
    XVarsVideoChatGPTCausalLM,
)
from opensportslib.models.utils.vqa_prompting import build_prior_text, build_xvars_prompt


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items()}
    return {}


def _resolve_sft_per_device_batch_sizes(config, sft_cfg: dict[str, Any]) -> tuple[int, int]:
    """Resolve HF per-device batch sizes from SFT overrides or split dataloaders."""

    train_dl = get_split_dataloader_cfg(config, "train")
    valid_dl = get_split_dataloader_cfg(config, "valid")
    train_default = getattr(train_dl, "batch_size", 1)
    valid_default = getattr(valid_dl, "batch_size", train_default)
    train_bs = sft_cfg.get("per_device_train_batch_size", train_default)
    eval_bs = sft_cfg.get("per_device_eval_batch_size", valid_default)
    return int(train_bs or 1), int(eval_bs or 1)


def _extract_cuda_device_index(config, hf_cfg: dict[str, Any]) -> int | None:
    # If CUDA_VISIBLE_DEVICES is set, CUDA indices are remapped; let runtime use
    # torch.cuda.current_device() to avoid 4-bit device mismatch with Accelerate.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None

    explicit = hf_cfg.get("cuda_device_index")
    if explicit is not None:
        try:
            return int(explicit)
        except Exception:
            return None
    system = getattr(config, "SYSTEM", None)
    gpu = getattr(system, "gpu", None) if system is not None else None
    if gpu is not None:
        gid = getattr(gpu, "id", None)
        if gid is not None:
            try:
                return int(gid)
            except Exception:
                return None
    return None


def _maybe_log_vqa_predictions(predictions: dict[str, Any], *, use_wandb: bool) -> None:
    if not use_wandb:
        return
    try:
        import wandb
        from opensportslib.core.utils.wandb import log_table_wandb
    except ImportError:
        return
    if getattr(wandb, "run", None) is None:
        return

    rows = predictions.get("data", []) if isinstance(predictions, dict) else []
    wandb.log({"vqa/infer_prediction_count": len(rows)})
    preview = [
        [
            row.get("id"),
            row.get("question"),
            row.get("answer_text"),
        ]
        for row in rows[:10]
    ]
    if preview:
        log_table_wandb(
            name="vqa/infer_preview",
            rows=preview,
            headers=["id", "question", "answer_text"],
        )


def _maybe_log_vqa_metrics(metrics: dict[str, Any], *, use_wandb: bool) -> None:
    if not use_wandb:
        return
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, "run", None) is None or not isinstance(metrics, dict):
        return

    payload = {
        f"vqa/eval/{key}": value
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool))
    }
    if payload:
        wandb.log(payload)


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
        built_prior = build_prior_text(
            sample.get("labels", {}) or {},
            sample.get("metadata", {}) or {},
            include_fields=prompt_cfg.get("prior_fields"),
        )
        prior_text = str(sample.get("prior_prediction_text", "")).strip() or built_prior

    token_len = int(sft_cfg.get("video_token_len", 300)) if bool(sft_cfg.get("include_video_tokens", True)) else 0
    prompt = build_xvars_prompt(
        system_prompt=system_prompt,
        question=question,
        prior_text=prior_text,
        video_token_len=token_len,
    )
    return {
        "prompt": prompt,
        "answer": answer,
        "completion": answer,
        "text": f"{prompt} {answer}".strip(),
    }


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

    def train(self, train_data, valid_data=None, *, rank: int = 0, world_size: int = 1, use_wandb: bool = False) -> str:
        execution = get_train_execution(self.config)
        prompt_cfg = _as_dict(execution.get("prompt"))
        sft_cfg = _as_dict(execution.get("sft"))
        hf_cfg = _as_dict(execution.get("hf"))
        lora_cfg = _as_dict(execution.get("lora"))
        quant_cfg = _as_dict(execution.get("quantization"))
        checkpoint_cfg = _as_dict(execution.get("checkpoint"))

        is_ddp = int(world_size) > 1
        if is_ddp:
            logging.info("VQA LoRA trainer | mode=ddp | world_size=%s | rank=%s", world_size, rank)
        else:
            logging.info("VQA LoRA trainer | mode=single | rank=%s", rank)

        save_root = get_system_path(self.config, "save_dir", "./checkpoints") or "./checkpoints"
        output_dir = os.path.join(save_root, "xvars_lora")
        os.makedirs(output_dir, exist_ok=True)

        train_sft = VQALoraSFTDataset(train_data, prompt_cfg, sft_cfg)
        valid_sft = VQALoraSFTDataset(valid_data, prompt_cfg, sft_cfg) if valid_data is not None else None

        metadata = {
            "backend": "xvars_lora",
            "model_id": str(hf_cfg.get("model_id", "distilgpt2")),
            "num_train_samples": len(train_sft),
            "num_valid_samples": len(valid_sft) if valid_sft is not None else 0,
            "status": "metadata_only",
            "optional_dependencies": {
                "trl": optional_package_available("trl"),
                "peft": optional_package_available("peft"),
                "bitsandbytes": optional_package_available("bitsandbytes"),
            },
            "data_quality": {
                "train_total_rows": len(train_sft),
                "valid_total_rows": len(valid_sft) if valid_sft is not None else 0,
                "train_dropped_tokenization_mismatch": 0,
                "valid_dropped_tokenization_mismatch": 0,
            },
        }

        if bool(execution.get("dry_run", False)):
            return self._write_artifacts(output_dir, metadata)

        require_optional_package("trl", "pip install trl")
        require_optional_package("datasets", "pip install datasets")
        from datasets import Dataset
        from transformers import TrainingArguments

        model_id = str(hf_cfg.get("model_id", "distilgpt2"))
        tokenizer, model, _device = load_hf_causal_lm_for_training(
            model_id,
            local_files_only=bool(hf_cfg.get("local_files_only", False)),
            prefer_cuda=bool(hf_cfg.get("prefer_cuda", True)),
            quantization_cfg=quant_cfg,
            cuda_device_index=rank if is_ddp else _extract_cuda_device_index(self.config, hf_cfg),
        )
        model = apply_lora_for_causal_lm(model, lora_cfg, distributed=is_ddp)
        max_seq_length = int(sft_cfg.get("max_seq_length", 512))
        train_rows, dropped_train = self._tokenize_and_mask_rows_xvars_style(
            train_sft.rows,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        valid_rows = []
        dropped_valid = 0
        if valid_sft is not None:
            valid_rows, dropped_valid = self._tokenize_and_mask_rows_xvars_style(
                valid_sft.rows,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
        metadata["data_quality"]["train_dropped_tokenization_mismatch"] = int(dropped_train)
        metadata["data_quality"]["valid_dropped_tokenization_mismatch"] = int(dropped_valid)
        metadata["data_quality"]["train_kept_rows"] = int(len(train_rows))
        metadata["data_quality"]["valid_kept_rows"] = int(len(valid_rows))
        if len(train_rows) == 0:
            raise ValueError(
                "All training rows were dropped by tokenization/masking checks. "
                "Increase sft.max_seq_length or reduce prompt/video token length."
            )

        hf_train_sft = Dataset.from_list(train_rows)
        hf_valid_sft = Dataset.from_list(valid_rows) if valid_sft is not None else None
        train_bs, eval_bs = _resolve_sft_per_device_batch_sizes(self.config, sft_cfg)

        training_kwargs = {
            "output_dir": output_dir,
            "per_device_train_batch_size": train_bs,
            "per_device_eval_batch_size": eval_bs,
            "gradient_accumulation_steps": int(sft_cfg.get("gradient_accumulation_steps", 1)),
            "num_train_epochs": float(sft_cfg.get("num_train_epochs", 1)),
            "learning_rate": float(sft_cfg.get("learning_rate", 2e-4)),
            "logging_steps": int(sft_cfg.get("logging_steps", 1)),
            "save_strategy": str(sft_cfg.get("save_strategy", "epoch")),
            "report_to": ["wandb"] if use_wandb else [],
            "fp16": bool(sft_cfg.get("fp16", False)),
            "bf16": bool(sft_cfg.get("bf16", False)),
            "use_cpu": not bool(hf_cfg.get("prefer_cuda", True)),
            "disable_tqdm": bool(sft_cfg.get("disable_tqdm", True)),
            "gradient_checkpointing": bool(sft_cfg.get("gradient_checkpointing", False)),
        }
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        if is_ddp and "ddp_find_unused_parameters" in ta_params:
            training_kwargs["ddp_find_unused_parameters"] = False
        if is_ddp and "gradient_checkpointing_kwargs" in ta_params:
            training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        args = TrainingArguments(**training_kwargs)
        trainer = build_trl_sft_trainer(
            model=model,
            train_dataset=hf_train_sft,
            eval_dataset=hf_valid_sft,
            tokenizer=tokenizer,
            args=args,
            dataset_text_field=None,
            max_seq_length=max_seq_length,
            completion_only_loss=bool(sft_cfg.get("completion_only_loss", True)),
        )
        trainer.train()

        if rank == 0 and bool(checkpoint_cfg.get("save_adapter", True)):
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        metadata["status"] = "trained"
        if rank == 0:
            return self._write_artifacts(output_dir, metadata)
        return output_dir

    def _write_artifacts(self, output_dir: str, metadata: dict[str, Any]) -> str:
        save_config(self.config, os.path.join(output_dir, "config.yaml"))
        with open(os.path.join(output_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        marker = os.path.join(output_dir, "adapter_model")
        os.makedirs(marker, exist_ok=True)
        with open(os.path.join(marker, "README.txt"), "w", encoding="utf-8") as f:
            f.write("VQA LoRA adapter artifacts are stored in the parent directory when training runs.\n")
        return output_dir

    @staticmethod
    def _filter_tokenization_mismatch(rows, *, tokenizer, max_seq_length: int):
        kept = []
        dropped = 0
        for row in rows:
            prompt = str(row.get("prompt", ""))
            completion = str(row.get("completion", ""))
            if not prompt or not completion:
                dropped += 1
                continue

            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_seq_length).input_ids
            full_ids = tokenizer(
                f"{prompt} {completion}".strip(),
                truncation=True,
                max_length=max_seq_length,
            ).input_ids
            # Prompt must be a prefix and completion must contribute at least one token.
            if len(full_ids) <= len(prompt_ids):
                dropped += 1
                continue
            if full_ids[: len(prompt_ids)] != prompt_ids:
                dropped += 1
                continue
            kept.append(row)
        return kept, dropped

    @staticmethod
    def _tokenize_and_mask_rows_xvars_style(rows, *, tokenizer, max_seq_length: int):
        """Tokenize prompt+completion and mask prompt tokens (answer-only loss)."""
        kept = []
        dropped = 0
        ignore_index = -100
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        for row in rows:
            prompt = str(row.get("prompt", "")).strip()
            completion = str(row.get("completion", "")).strip()
            if not prompt or not completion:
                dropped += 1
                continue
            full_text = f"{prompt} {completion}".strip()
            enc_full = tokenizer(
                full_text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )
            enc_prompt = tokenizer(
                prompt,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )
            input_ids = list(enc_full["input_ids"])
            attn = list(enc_full["attention_mask"])
            prompt_ids = list(enc_prompt["input_ids"])
            full_len = int(sum(attn))
            prompt_len = int(sum(enc_prompt["attention_mask"]))
            if full_len <= prompt_len or prompt_len <= 0:
                dropped += 1
                continue
            if input_ids[:prompt_len] != prompt_ids[:prompt_len]:
                dropped += 1
                continue
            labels = list(input_ids)
            for i in range(len(labels)):
                if i < prompt_len:
                    labels[i] = ignore_index
                elif attn[i] == 0:
                    labels[i] = ignore_index
                elif input_ids[i] == pad_id:
                    labels[i] = ignore_index
            if all(x == ignore_index for x in labels):
                dropped += 1
                continue
            kept.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "labels": labels,
                }
            )
        return kept, dropped


class XVarsVideoChatGPTDataCollator:
    """Batch tokenized X-VARS samples including visual feature tensors."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        out = {
            "input_ids": torch.tensor([x["input_ids"] for x in instances], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in instances], dtype=torch.long),
            "labels": torch.tensor([x["labels"] for x in instances], dtype=torch.long),
            "video_spatio_temporal_features": torch.stack(
                [torch.as_tensor(x["video_spatio_temporal_features"], dtype=torch.float32) for x in instances],
                dim=0,
            ),
        }
        return out


class XVarsVideoChatGPTTrainer:
    """Minimal Trainer wrapper that passes the tokenizer into the X-VARS model."""

    def __init__(self, *, model, tokenizer, args, train_dataset, eval_dataset=None, data_collator=None):
        from transformers import Trainer

        class _Trainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                del kwargs
                outputs = model(**inputs, tokenizer=tokenizer)
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

        self._trainer = _Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator or XVarsVideoChatGPTDataCollator(tokenizer),
        )
        self.model = model
        self.tokenizer = tokenizer

    def train(self):
        return self._trainer.train()

    def save_state(self):
        return self._trainer.save_state()


class VQAXVarsVideoChatGPTSFTDataset:
    """Tokenized multimodal SFT dataset for the true X-VARS backend."""

    def __init__(self, dataset, *, tokenizer, prompt_cfg: dict[str, Any], sft_cfg: dict[str, Any], xvars_cfg: dict[str, Any]):
        self.rows = []
        token_len = int(xvars_cfg.get("video_token_len", sft_cfg.get("video_token_len", 356)))
        max_seq_length = int(sft_cfg.get("max_seq_length", 768))
        for sample in dataset:
            refs = sample.get("references") or []
            if not refs:
                continue
            features = sample.get("video_spatio_temporal_features")
            if features is None:
                continue
            features = torch.as_tensor(features, dtype=torch.float32)
            if features.ndim != 2:
                continue
            if features.shape[0] != token_len:
                # X-VARS requires one feature row per <vid_patch>. Crop/pad keeps
                # legacy indexes usable while preserving the model contract.
                if features.shape[0] > token_len:
                    features = features[:token_len]
                else:
                    pad = torch.zeros((token_len - features.shape[0], features.shape[1]), dtype=features.dtype)
                    features = torch.cat([features, pad], dim=0)
            row = build_vqa_sft_text(
                sample,
                prompt_cfg=prompt_cfg,
                sft_cfg={**sft_cfg, "include_video_tokens": True, "video_token_len": token_len},
            )
            tokenized = self._tokenize_row(row, tokenizer=tokenizer, max_seq_length=max_seq_length)
            if tokenized is None:
                continue
            tokenized["video_spatio_temporal_features"] = features
            self.rows.append(tokenized)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    @staticmethod
    def _tokenize_row(row, *, tokenizer, max_seq_length: int):
        ignore_index = -100
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        prompt = str(row.get("prompt", "")).strip()
        completion = str(row.get("completion", "")).strip()
        if not prompt or not completion:
            return None
        full_text = f"{prompt} {completion}".strip()
        enc_full = tokenizer(full_text, truncation=True, max_length=max_seq_length, padding="max_length")
        enc_prompt = tokenizer(prompt, truncation=True, max_length=max_seq_length, padding="max_length")
        input_ids = list(enc_full["input_ids"])
        attn = list(enc_full["attention_mask"])
        prompt_ids = list(enc_prompt["input_ids"])
        full_len = int(sum(attn))
        prompt_len = int(sum(enc_prompt["attention_mask"]))
        if full_len <= prompt_len or input_ids[:prompt_len] != prompt_ids[:prompt_len]:
            return None
        labels = list(input_ids)
        for i in range(len(labels)):
            if i < prompt_len or attn[i] == 0 or input_ids[i] == pad_id:
                labels[i] = ignore_index
        if all(x == ignore_index for x in labels):
            return None
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


class VQAXVarsVideoChatGPTLoraTrainer:
    """True X-VARS LoRA trainer preserving multimodal feature tensors."""

    def __init__(self, config):
        self.config = config

    def train(self, train_data, valid_data=None, *, rank: int = 0, world_size: int = 1, use_wandb: bool = False) -> str:
        execution = get_train_execution(self.config)
        prompt_cfg = _as_dict(execution.get("prompt"))
        sft_cfg = _as_dict(execution.get("sft"))
        hf_cfg = _as_dict(execution.get("hf"))
        lora_cfg = _as_dict(execution.get("lora"))
        quant_cfg = _as_dict(execution.get("quantization"))
        checkpoint_cfg = _as_dict(execution.get("checkpoint"))
        xvars_cfg = _as_dict(execution.get("xvars"))

        save_root = get_system_path(self.config, "save_dir", "./checkpoints") or "./checkpoints"
        output_dir = os.path.join(save_root, "xvars_videochatgpt_lora")
        os.makedirs(output_dir, exist_ok=True)

        metadata = {
            "backend": "xvars_videochatgpt_lora",
            "model_id": str(xvars_cfg.get("base_model") or hf_cfg.get("model_id", "lmsys/vicuna-7b-v1.1")),
            "status": "metadata_only",
            "multimodal_training": True,
            "video_token_len": int(xvars_cfg.get("video_token_len", sft_cfg.get("video_token_len", 356))),
            "lora_target_modules": list(lora_cfg.get("target_modules") or DEFAULT_XVARS_TARGET_MODULES),
            "num_train_samples": 0,
            "num_valid_samples": 0,
        }
        if bool(execution.get("dry_run", False)):
            # Dry-run still validates that dataset items carry visual features.
            metadata["num_train_samples"] = sum(1 for x in train_data if x.get("video_spatio_temporal_features") is not None)
            if valid_data is not None:
                metadata["num_valid_samples"] = sum(1 for x in valid_data if x.get("video_spatio_temporal_features") is not None)
            return self._write_artifacts(output_dir, metadata)

        require_optional_package("transformers", "pip install transformers")
        require_optional_package("peft", "pip install peft")
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

        model_id = str(xvars_cfg.get("base_model") or hf_cfg.get("model_id", "lmsys/vicuna-7b-v1.1"))
        local_files_only = bool(hf_cfg.get("local_files_only", False))
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bnb_config = build_bitsandbytes_config(quant_cfg)
        model_kwargs = {"local_files_only": local_files_only}
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            if bool(hf_cfg.get("prefer_cuda", True)) and torch.cuda.is_available():
                model_kwargs["device_map"] = {"": torch.cuda.current_device()}
        base_lm = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        from opensportslib.core.utils.hf_runtime import _ensure_video_special_tokens

        _ensure_video_special_tokens(tokenizer, base_lm)
        model = XVarsVideoChatGPTCausalLM.from_pretrained_projector(
            base_lm,
            xvars_cfg.get("projection_path"),
            mm_hidden_size=int(xvars_cfg.get("mm_hidden_size", 1024)),
        )
        lora_cfg = dict(lora_cfg)
        lora_cfg.setdefault("target_modules", DEFAULT_XVARS_TARGET_MODULES)
        model = apply_lora_for_causal_lm(model, lora_cfg, distributed=int(world_size) > 1)

        train_sft = VQAXVarsVideoChatGPTSFTDataset(
            train_data,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            sft_cfg=sft_cfg,
            xvars_cfg=xvars_cfg,
        )
        valid_sft = (
            VQAXVarsVideoChatGPTSFTDataset(
                valid_data,
                tokenizer=tokenizer,
                prompt_cfg=prompt_cfg,
                sft_cfg=sft_cfg,
                xvars_cfg=xvars_cfg,
            )
            if valid_data is not None
            else None
        )
        if len(train_sft) == 0:
            raise ValueError("No multimodal X-VARS training rows were usable.")
        metadata["num_train_samples"] = len(train_sft)
        metadata["num_valid_samples"] = len(valid_sft) if valid_sft is not None else 0
        train_bs, eval_bs = _resolve_sft_per_device_batch_sizes(self.config, sft_cfg)

        training_kwargs = {
            "output_dir": output_dir,
            "per_device_train_batch_size": train_bs,
            "per_device_eval_batch_size": eval_bs,
            "gradient_accumulation_steps": int(sft_cfg.get("gradient_accumulation_steps", 1)),
            "num_train_epochs": float(sft_cfg.get("num_train_epochs", 1)),
            "learning_rate": float(sft_cfg.get("learning_rate", 2e-4)),
            "logging_steps": int(sft_cfg.get("logging_steps", 1)),
            "save_strategy": str(sft_cfg.get("save_strategy", "epoch")),
            "report_to": ["wandb"] if use_wandb else [],
            "remove_unused_columns": False,
            "disable_tqdm": bool(sft_cfg.get("disable_tqdm", True)),
            "use_cpu": not bool(hf_cfg.get("prefer_cuda", True)),
        }
        args = TrainingArguments(**training_kwargs)
        trainer = XVarsVideoChatGPTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_sft,
            eval_dataset=valid_sft,
            data_collator=XVarsVideoChatGPTDataCollator(tokenizer),
        )
        trainer.train()
        if rank == 0 and bool(checkpoint_cfg.get("save_adapter", True)):
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        metadata["status"] = "trained"
        if rank == 0:
            return self._write_artifacts(output_dir, metadata)
        return output_dir

    def _write_artifacts(self, output_dir: str, metadata: dict[str, Any]) -> str:
        save_config(self.config, os.path.join(output_dir, "config.yaml"))
        with open(os.path.join(output_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        marker = os.path.join(output_dir, "adapter_model")
        os.makedirs(marker, exist_ok=True)
        with open(os.path.join(marker, "README.txt"), "w", encoding="utf-8") as f:
            f.write("X-VARS VideoChatGPT LoRA adapter artifacts are stored in the parent directory when training runs.\n")
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

    def train(self, model, train_data, valid_data=None, *, rank: int = 0, world_size: int = 1, use_wandb: bool = False) -> str:
        del model
        execution = get_train_execution(self.config)
        backend = str(execution.get("training_backend", "placeholder")).lower()
        if backend == "xvars_videochatgpt_lora":
            ckpt = VQAXVarsVideoChatGPTLoraTrainer(self.config).train(
                train_data,
                valid_data,
                rank=rank,
                world_size=world_size,
                use_wandb=use_wandb,
            )
            self.best_checkpoint_path = ckpt
            return ckpt
        if backend == "xvars_lora":
            ckpt = VQALoraTrainer(self.config).train(
                train_data,
                valid_data,
                rank=rank,
                world_size=world_size,
                use_wandb=use_wandb,
            )
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

    def infer(self, model, dataset, *, use_wandb: bool = False) -> dict[str, Any]:
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
        payload = {"task": "vqa", "data": preds}
        _maybe_log_vqa_predictions(payload, use_wandb=use_wandb)
        return payload

    def evaluate(self, predictions: dict[str, Any], dataset, *, use_wandb: bool = False) -> dict[str, Any]:
        metrics = compute_vqa_metrics(predictions, dataset, eval_profile=get_vqa_eval_profile_cfg(self.config))
        _maybe_log_vqa_metrics(metrics, use_wandb=use_wandb)
        return metrics


__all__ = [
    "OptionalDependencyError",
    "Trainer_VQA",
    "XVarsVideoChatGPTDataCollator",
    "XVarsVideoChatGPTTrainer",
    "VQALoraSFTDataset",
    "VQALoraTrainer",
    "VQAXVarsVideoChatGPTLoraTrainer",
    "VQAXVarsVideoChatGPTSFTDataset",
    "build_vqa_sft_text",
]
