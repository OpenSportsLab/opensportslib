"""Trainer/infer/eval helpers for VQA task."""

from __future__ import annotations

import json
import logging
import os
import inspect
import time
from typing import Any

import torch
from tqdm.auto import tqdm

from opensportslib.core.config.accessors import (
    get_xvars_train_model_id,
    get_xvars_train_tokenizer_id,
    get_xvars_train_video_token_len,
    is_xvars_videochatgpt_backend,
    get_model_runtime_dtype,
    get_split_dataloader_cfg,
    get_system_path,
    get_train_epochs,
    get_train_execution,
    get_train_optimizer,
    get_vqa_eval_profile_cfg,
    get_vqa_generation_cfg,
    get_vqa_mm_hidden_size,
    get_vqa_prompt_video_token_len,
)
from opensportslib.core.utils.config import save_config
from opensportslib.core.utils.hf_runtime import (
    OptionalDependencyError,
    apply_lora_for_causal_lm,
    build_bitsandbytes_config,
    require_optional_package,
)
from opensportslib.metrics.vqa_metric import compute_vqa_metrics
from opensportslib.models.base.video_chatgpt_compat import load_videochatgpt_compatible_causal_lm
from opensportslib.models.base.xvars_videochatgpt import (
    DEFAULT_XVARS_TARGET_MODULES,
    XVarsVideoChatGPTCausalLM,
)
from opensportslib.models.utils.vqa_prompting import build_prior_text, build_xvars_prompt
from opensportslib.models.utils.xvars_clip_index import validate_xvars_feature_tensor


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


def _resolve_vqa_video_token_len(config, prompt_cfg: dict[str, Any] | None = None, sft_cfg: dict[str, Any] | None = None) -> int:
    prompt_cfg = prompt_cfg or {}
    sft_cfg = sft_cfg or {}
    if config is not None and is_xvars_videochatgpt_backend(config):
        return get_xvars_train_video_token_len(config)
    if prompt_cfg.get("video_token_len") is not None:
        return int(prompt_cfg["video_token_len"])
    if sft_cfg.get("video_token_len") is not None:
        return int(sft_cfg["video_token_len"])
    return get_vqa_prompt_video_token_len(config, default=300)


def _resolve_training_precision_flags(config, sft_cfg: dict[str, Any]) -> tuple[bool, bool]:
    dtype = get_model_runtime_dtype(config, default="fp32")
    if dtype == "bf16":
        return False, True
    if dtype == "fp16":
        return True, False
    return bool(sft_cfg.get("fp16", False)), bool(sft_cfg.get("bf16", False))


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
    config=None,
    prompt_cfg: dict[str, Any] | None = None,
    sft_cfg: dict[str, Any] | None = None,
    reference: str | None = None,
) -> dict[str, str]:
    """Convert a VQADataset sample into prompt/answer/text fields for SFT."""
    prompt_cfg = prompt_cfg or {}
    sft_cfg = sft_cfg or {}
    refs = sample.get("references") or []
    answer = str(reference).strip() if reference is not None else (str(refs[0]).strip() if refs else "")
    question = str(sample.get("question", "")).strip()
    system_prompt = str(
        prompt_cfg.get(
            "system_prompt",
            "You are an artificial intelligence assistant for visual question answering.",
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

    if config is not None:
        token_len = _resolve_vqa_video_token_len(config, prompt_cfg=prompt_cfg, sft_cfg=sft_cfg)
    else:
        token_len = int(prompt_cfg.get("video_token_len", sft_cfg.get("video_token_len", 300)))
    token_len = token_len if bool(sft_cfg.get("include_video_tokens", True)) else 0
    prompt = build_xvars_prompt(
        system_prompt=system_prompt,
        question=question,
        prior_text=prior_text,
        video_token_len=token_len,
    )
    append_eos = bool(sft_cfg.get("append_eos_token", True))
    completion = f"{answer}</s>" if answer and append_eos else answer
    return {
        "prompt": prompt,
        "answer": answer,
        "completion": completion,
        "text": f"{prompt} {completion}".strip(),
    }


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

    def __init__(self, *, model, tokenizer, args, train_dataset, eval_dataset=None, data_collator=None, callbacks=None):
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
            callbacks=callbacks,
        )
        self.model = model
        self.tokenizer = tokenizer

    def train(self):
        return self._trainer.train()

    def save_state(self):
        return self._trainer.save_state()


class VQAXVarsVideoChatGPTSFTDataset:
    """Tokenized multimodal SFT dataset for the true X-VARS backend."""

    def __init__(self, dataset, *, config=None, tokenizer, prompt_cfg: dict[str, Any], sft_cfg: dict[str, Any], xvars_cfg: dict[str, Any]):
        self.rows = []
        if config is not None and is_xvars_videochatgpt_backend(config):
            token_len = get_xvars_train_video_token_len(config)
        else:
            token_len = int(prompt_cfg.get("video_token_len", xvars_cfg.get("video_token_len", sft_cfg.get("video_token_len", 356))))
        max_seq_length = int(sft_cfg.get("max_seq_length", 768))
        for sample in dataset:
            refs = sample.get("references") or []
            if not refs:
                continue
            features = sample.get("video_spatio_temporal_features")
            if features is None:
                continue
            features = validate_xvars_feature_tensor(
                torch.as_tensor(features, dtype=torch.float32),
                expected_tokens=token_len,
                context=f"X-VARS training features for sample '{sample.get('id', 'unknown')}'",
            )
            references = refs if str(sft_cfg.get("reference_mode", "all")).lower() == "all" else refs[:1]
            for reference in references:
                if not str(reference).strip():
                    continue
                row = build_vqa_sft_text(
                    sample,
                    config=config,
                    prompt_cfg=prompt_cfg,
                    sft_cfg={**sft_cfg, "include_video_tokens": True, "video_token_len": token_len},
                    reference=str(reference),
                )
                tokenized = self._tokenize_row(row, tokenizer=tokenizer, max_seq_length=max_seq_length)
                if tokenized is None:
                    continue
                tokenized["video_spatio_temporal_features"] = features
                tokenized["id"] = str(sample.get("id", ""))
                tokenized["question"] = str(sample.get("question", ""))
                tokenized["prompt"] = row["prompt"]
                self.rows.append(tokenized)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    @staticmethod
    def _tokenize_row(row, *, tokenizer, max_seq_length: int):
        ignore_index = -100
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
        eos_token = str(getattr(tokenizer, "eos_token", "") or "")
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if completion.endswith(eos_token) and eos_token and eos_token_id is not None:
            input_ids[full_len - 1] = int(eos_token_id)
        labels = list(input_ids)
        for i in range(len(labels)):
            if i < prompt_len or attn[i] == 0:
                labels[i] = ignore_index
        if all(x == ignore_index for x in labels):
            return None
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def _score_xvars_generated_answers(
    answers: list[str],
    *,
    required_terms: list[str],
    forbidden_terms: list[str],
    enforce_relevance: bool = True,
) -> dict[str, Any]:
    normalized = [str(answer).lower() for answer in answers]
    relevant = (
        [any(term.lower() in answer for term in required_terms) for answer in normalized]
        if enforce_relevance and required_terms
        else [True] * len(normalized)
    )
    forbidden = [any(term.lower() in answer for term in forbidden_terms) for answer in normalized]
    accepted_count = sum(ok and not bad for ok, bad in zip(relevant, forbidden))
    return {
        "accepted": bool(answers) and accepted_count == len(answers),
        "accepted_count": accepted_count,
        "answer_count": len(answers),
        "forbidden_count": sum(forbidden),
    }


def _build_xvars_generated_validation_callback(
    *,
    tokenizer,
    rows: list[dict[str, Any]],
    output_dir: str,
    validation_cfg: dict[str, Any],
    use_step_schedule: bool,
):
    from transformers import TrainerCallback

    sample_id = str(validation_cfg.get("sample_id") or "").strip()
    if not sample_id:
        logging.warning("Generated validation disabled: missing generated_validation.sample_id.")
        return None
    selected = []
    seen_questions = set()
    for row in rows:
        if str(row.get("id")) != sample_id or row.get("question") in seen_questions:
            continue
        selected.append(row)
        seen_questions.add(row.get("question"))
    if not selected:
        logging.warning("Generated validation disabled: sample id '%s' was not found.", sample_id)
        return None

    require_relevance = bool(validation_cfg.get("require_relevance", False))
    required_terms = list(validation_cfg.get("required_terms") or [])
    forbidden_terms = list(validation_cfg.get("forbidden_terms") or [])
    if require_relevance and not required_terms:
        logging.warning(
            "Generated validation relevance checks requested for sample id '%s' but no required_terms were configured; "
            "relevance gating will be skipped.",
            sample_id,
        )
    every_steps = max(1, int(validation_cfg.get("every_steps", 25)))
    max_new_tokens = max(1, int(validation_cfg.get("max_new_tokens", 128)))

    class _GeneratedValidationCallback(TrainerCallback):
        def __init__(self):
            self.history = []
            self.best_score = (-1, 0, float("-inf"))

        def _consider_checkpoint(self, *, model, record: dict[str, Any]):
            eval_loss = record.get("eval_loss")
            loss_score = -float(eval_loss) if eval_loss is not None else float("-inf")
            rank_score = (record["accepted_count"], -record["forbidden_count"], loss_score)
            if rank_score <= self.best_score:
                return
            self.best_score = rank_score
            best_dir = os.path.join(output_dir, "generated_validation_best")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

        def _run(self, *, model, step: int, epoch: float | None):
            was_training = model.training
            previous_use_cache = getattr(model.config, "use_cache", None)
            model.eval()
            model.config.use_cache = True
            answers = []
            try:
                for row in selected:
                    encoded = tokenizer([row["prompt"]], return_tensors="pt")
                    device = next(model.parameters()).device
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    features = torch.as_tensor(row["video_spatio_temporal_features"], dtype=torch.float32).unsqueeze(0)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            tokenizer=tokenizer,
                            attention_mask=attention_mask,
                            video_spatio_temporal_features=features,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    generated = output_ids[:, input_ids.shape[-1]:] if output_ids.shape[-1] > input_ids.shape[-1] else output_ids
                    answers.append(tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip())
            finally:
                if previous_use_cache is not None:
                    model.config.use_cache = previous_use_cache
                if was_training:
                    model.train()

            score = _score_xvars_generated_answers(
                answers,
                required_terms=required_terms,
                forbidden_terms=forbidden_terms,
                enforce_relevance=require_relevance,
            )
            record = {"step": int(step), "epoch": epoch, "sample_id": sample_id, "answers": answers, **score}
            self.history.append(record)
            logging.info("X-VARS generated validation | %s", json.dumps(record, ensure_ascii=True))
            self._consider_checkpoint(model=model, record=record)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            del args, kwargs
            if use_step_schedule and state.is_world_process_zero and state.global_step % every_steps == 0:
                self._run(model=model, step=state.global_step, epoch=state.epoch)
            return control

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            del args, kwargs
            if not use_step_schedule and state.is_world_process_zero:
                self._run(model=model, step=state.global_step, epoch=state.epoch)
            return control

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            del args, kwargs
            if not state.is_world_process_zero or not self.history or not metrics or "eval_loss" not in metrics:
                return control
            record = self.history[-1]
            if record["step"] == int(state.global_step):
                record["eval_loss"] = float(metrics["eval_loss"])
                self._consider_checkpoint(model=model, record=record)
            return control

    return _GeneratedValidationCallback()


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
        generated_validation_cfg = _as_dict(execution.get("generated_validation"))

        save_root = get_system_path(self.config, "save_dir", "./checkpoints") or "./checkpoints"
        output_dir = os.path.join(save_root, "xvars_videochatgpt_lora")
        os.makedirs(output_dir, exist_ok=True)

        prompt_token_len = _resolve_vqa_video_token_len(self.config, prompt_cfg=prompt_cfg, sft_cfg=sft_cfg)
        model_id = get_xvars_train_model_id(self.config, default=str(hf_cfg.get("model_id", "base_model_videoChatGPT")))
        metadata = {
            "backend": "xvars_videochatgpt_lora",
            "model_id": model_id,
            "status": "metadata_only",
            "multimodal_training": True,
            "video_token_len": prompt_token_len,
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
        from transformers import AutoTokenizer, TrainingArguments

        local_files_only = bool(hf_cfg.get("local_files_only", False))
        tokenizer = AutoTokenizer.from_pretrained(
            get_xvars_train_tokenizer_id(self.config),
            local_files_only=local_files_only,
            use_fast=False,
            model_max_length=int(hf_cfg.get("model_max_length", 1048)),
            padding_side=str(hf_cfg.get("padding_side", "right")),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bnb_config = build_bitsandbytes_config(quant_cfg)
        model_kwargs = {"local_files_only": local_files_only}
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            if bool(hf_cfg.get("prefer_cuda", True)) and torch.cuda.is_available():
                model_kwargs["device_map"] = {"": torch.cuda.current_device()}
        base_lm = load_videochatgpt_compatible_causal_lm(model_id, **model_kwargs)
        from opensportslib.core.utils.hf_runtime import _ensure_video_special_tokens

        _ensure_video_special_tokens(tokenizer, base_lm)
        model = XVarsVideoChatGPTCausalLM.from_pretrained_projector(
            base_lm,
            xvars_cfg.get("projection_path"),
            mm_hidden_size=get_vqa_mm_hidden_size(self.config, default=1024),
        )
        model.config.use_cache = False
        lora_cfg = dict(lora_cfg)
        lora_cfg.setdefault("target_modules", DEFAULT_XVARS_TARGET_MODULES)
        model = apply_lora_for_causal_lm(model, lora_cfg, distributed=int(world_size) > 1)
        model.config.use_cache = False

        logging.info("Building X-VARS SFT datasets | rank=%s", rank)
        train_sft = VQAXVarsVideoChatGPTSFTDataset(
            train_data,
            config=self.config,
            tokenizer=tokenizer,
            prompt_cfg=prompt_cfg,
            sft_cfg=sft_cfg,
            xvars_cfg=xvars_cfg,
        )
        valid_sft = (
            VQAXVarsVideoChatGPTSFTDataset(
                valid_data,
                config=self.config,
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
        feature_shape = tuple(train_sft[0]["video_spatio_temporal_features"].shape) if len(train_sft) else None
        logging.info(
            "Built X-VARS SFT datasets | rank=%s | train=%s | valid=%s | feature_shape=%s",
            rank,
            metadata["num_train_samples"],
            metadata["num_valid_samples"],
            feature_shape,
        )
        train_bs, eval_bs = _resolve_sft_per_device_batch_sizes(self.config, sft_cfg)

        optimizer_cfg = get_train_optimizer(self.config)
        execution_cfg = get_train_execution(self.config)
        fp16, bf16 = _resolve_training_precision_flags(self.config, sft_cfg)
        eval_strategy_key = "evaluation_strategy"
        try:
            if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
                eval_strategy_key = "eval_strategy"
        except Exception:
            eval_strategy_key = "evaluation_strategy"

        training_kwargs = {
            "output_dir": output_dir,
            "per_device_train_batch_size": train_bs,
            "per_device_eval_batch_size": eval_bs,
            "gradient_accumulation_steps": int(execution_cfg.get("acc_grad_iter", sft_cfg.get("gradient_accumulation_steps", 1))),
            "num_train_epochs": float(get_train_epochs(self.config) or sft_cfg.get("num_train_epochs", 1)),
            "learning_rate": float(optimizer_cfg.get("lr", sft_cfg.get("learning_rate", 2e-4))),
            "logging_steps": int(execution_cfg.get("log_interval", sft_cfg.get("logging_steps", 1))),
            "optim": str(optimizer_cfg.get("hf_optim", "paged_adamw_8bit")),
            "weight_decay": float(optimizer_cfg.get("weight_decay", 0.001)),
            "lr_scheduler_type": str(sft_cfg.get("lr_scheduler_type", "constant")),
            "save_strategy": str(sft_cfg.get("save_strategy", "epoch")),
            "report_to": ["wandb"] if use_wandb else [],
            "remove_unused_columns": False,
            "disable_tqdm": bool(sft_cfg.get("disable_tqdm", True)),
            "use_cpu": not bool(hf_cfg.get("prefer_cuda", True)),
            "fp16": fp16,
            "bf16": bf16,
            "gradient_checkpointing": bool(sft_cfg.get("gradient_checkpointing", False)),
        }
        max_steps = sft_cfg.get("max_steps")
        if max_steps is not None and int(max_steps) > 0:
            training_kwargs["max_steps"] = int(max_steps)
        if int(world_size) > 1:
            training_kwargs["ddp_find_unused_parameters"] = True
            training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        training_kwargs[eval_strategy_key] = str(sft_cfg.get("evaluation_strategy", "epoch"))
        args = TrainingArguments(**training_kwargs)
        generated_callback = None
        if bool(generated_validation_cfg.get("enabled", False)):
            generated_callback = _build_xvars_generated_validation_callback(
                tokenizer=tokenizer,
                rows=train_sft.rows,
                output_dir=output_dir,
                validation_cfg=generated_validation_cfg,
                use_step_schedule=bool(training_kwargs.get("max_steps", 0)),
            )
        trainer = XVarsVideoChatGPTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_sft,
            eval_dataset=valid_sft,
            data_collator=XVarsVideoChatGPTDataCollator(tokenizer),
            callbacks=[generated_callback] if generated_callback is not None else None,
        )
        logging.info(
            "Starting X-VARS trainer.train | rank=%s | world_size=%s | precision=%s | "
            "max_seq_length=%s | lora_targets=%s | gradient_checkpointing=%s | "
            "ddp_find_unused_parameters=%s | use_cache=%s",
            rank,
            world_size,
            "fp16" if fp16 else "bf16" if bf16 else "fp32",
            int(sft_cfg.get("max_seq_length", 768)),
            list(lora_cfg["target_modules"]),
            training_kwargs["gradient_checkpointing"],
            training_kwargs.get("ddp_find_unused_parameters"),
            model.config.use_cache,
        )
        trainer.train()
        logging.info("Finished X-VARS trainer.train | rank=%s", rank)
        if rank == 0 and bool(checkpoint_cfg.get("save_adapter", True)):
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        metadata["status"] = "trained"
        selected_output_dir = output_dir
        if rank == 0 and generated_callback is not None:
            generated_path = os.path.join(output_dir, "generated_validation.json")
            with open(generated_path, "w", encoding="utf-8") as f:
                json.dump(generated_callback.history, f, indent=2)
            accepted = any(record.get("accepted") for record in generated_callback.history)
            metadata["generated_validation_accepted"] = accepted
            metadata["generated_validation_history"] = generated_path
            if generated_callback.history:
                selected_output_dir = os.path.join(output_dir, "generated_validation_best")
                metadata["best_generated_checkpoint"] = selected_output_dir
        if rank == 0:
            self._write_artifacts(output_dir, metadata)
            if selected_output_dir != output_dir:
                self._write_artifacts(selected_output_dir, metadata)
            if bool(generated_validation_cfg.get("require_relevance", False)) and not metadata.get(
                "generated_validation_accepted", False
            ):
                raise RuntimeError(
                    "X-VARS generated-answer relevance gate failed. Inspect generated_validation.json before "
                    "starting or accepting a full training run."
                )
            return selected_output_dir
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
            if self.loaded_checkpoint_metadata.get("generated_validation_accepted") is False:
                raise ValueError(
                    "Refusing to load a rejected VQA adapter: generated validation did not pass. "
                    f"Checkpoint: {weights}. Use the configured base model or a validated adapter."
                )
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
        raise ValueError(
            f"Unsupported VQA training backend '{backend}'. "
            "Only 'xvars_videochatgpt_lora' is supported."
        )

    def infer(self, model, dataset, *, use_wandb: bool = False) -> dict[str, Any]:
        exec_cfg = get_train_execution(self.config)
        prompt_cfg = exec_cfg.get("prompt", {}) if isinstance(exec_cfg, dict) else {}
        generation_cfg = get_vqa_generation_cfg(self.config)

        preds = []
        disable_tqdm = bool(exec_cfg.get("disable_tqdm", False) if isinstance(exec_cfg, dict) else False)
        try:
            total = len(dataset)
        except Exception:
            total = None
        logging.info(
            "Starting VQA inference | samples=%s | max_new_tokens=%s",
            total if total is not None else "unknown",
            generation_cfg.get("max_new_tokens"),
        )
        started_at = time.perf_counter()
        iterator = tqdm(
            dataset,
            total=total,
            desc="VQA inference",
            unit="sample",
            disable=disable_tqdm,
            leave=False,
        )
        for sample in iterator:
            sample_started_at = time.perf_counter()
            answer = model.generate_answer(
                sample,
                prompt_cfg=prompt_cfg,
                generation_cfg=generation_cfg,
            )
            if not disable_tqdm:
                iterator.set_postfix_str(f"id={sample.get('id')} elapsed={time.perf_counter() - sample_started_at:.2f}s")
            preds.append(
                {
                    "id": sample.get("id"),
                    "question": sample.get("question"),
                    "answer_text": answer,
                    "video_path": sample.get("video_path"),
                }
            )
        logging.info(
            "Finished VQA inference | samples=%s | elapsed_s=%.2f",
            len(preds),
            time.perf_counter() - started_at,
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
    "VQAXVarsVideoChatGPTLoraTrainer",
    "VQAXVarsVideoChatGPTSFTDataset",
    "build_vqa_sft_text",
]
