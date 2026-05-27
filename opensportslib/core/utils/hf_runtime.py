"""Shared HuggingFace runtime utilities for generation tasks."""

from __future__ import annotations

import logging
import os
import inspect
import time
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)


@contextmanager
def hf_offline_if_requested(enabled: bool):
    """Temporarily force HF libraries into offline mode for local-only loads."""
    if not enabled:
        yield
        return

    previous = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class OptionalDependencyError(ImportError):
    """Raised when an optional training dependency is required but missing."""


def require_optional_package(package: str, install_hint: str | None = None):
    """Import an optional package or raise an actionable error."""
    try:
        return __import__(package)
    except ImportError as exc:
        hint = install_hint or f"pip install {package}"
        raise OptionalDependencyError(
            f"Optional dependency '{package}' is required for this training backend. "
            f"Install it with: {hint}"
        ) from exc


def optional_package_available(package: str) -> bool:
    """Return whether an optional dependency can be imported."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def build_bitsandbytes_config(quantization_cfg: dict[str, Any] | None = None):
    """Build a BitsAndBytesConfig when 4-bit/8-bit quantization is enabled."""
    quantization_cfg = quantization_cfg or {}
    if not bool(quantization_cfg.get("enabled", False)):
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise OptionalDependencyError(
            "transformers BitsAndBytesConfig is required for quantized training."
        ) from exc

    require_optional_package("bitsandbytes", "pip install bitsandbytes")
    compute_dtype = str(quantization_cfg.get("compute_dtype", "bfloat16")).lower()
    dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(compute_dtype, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=bool(quantization_cfg.get("load_in_4bit", True)),
        load_in_8bit=bool(quantization_cfg.get("load_in_8bit", False)),
        bnb_4bit_quant_type=str(quantization_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=bool(quantization_cfg.get("bnb_4bit_use_double_quant", True)),
    )


def load_hf_causal_lm_for_training(
    model_id: str,
    *,
    local_files_only: bool = False,
    prefer_cuda: bool = True,
    quantization_cfg: dict[str, Any] | None = None,
):
    """Load tokenizer/model pair for causal LM fine-tuning."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"
    bnb_config = build_bitsandbytes_config(quantization_cfg)
    with hf_offline_if_requested(local_files_only):
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"local_files_only": local_files_only}
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto" if device == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if bnb_config is None:
        model = model.to(device)
    return tokenizer, model, device


def apply_lora_for_causal_lm(model, lora_cfg: dict[str, Any] | None = None):
    """Apply PEFT LoRA to a causal LM."""
    lora_cfg = lora_cfg or {}
    require_optional_package("peft", "pip install peft")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if bool(lora_cfg.get("prepare_kbit", False)):
        model = prepare_model_for_kbit_training(model)

    target_modules = lora_cfg.get("target_modules") or ["q_proj", "v_proj"]
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    return get_peft_model(model, peft_config)


def has_peft_adapter_artifacts(adapter_path: str | None) -> bool:
    """Detect whether a directory contains PEFT adapter files."""
    if not adapter_path or not os.path.isdir(adapter_path):
        return False
    return any(
        os.path.exists(os.path.join(adapter_path, name))
        for name in ("adapter_config.json", "adapter_model.bin", "adapter_model.safetensors")
    )


def load_peft_adapter_if_available(model, adapter_path: str | None):
    """Load a PEFT adapter into a model when real adapter artifacts exist."""
    if not has_peft_adapter_artifacts(adapter_path):
        return model, "not_found"
    if not optional_package_available("peft"):
        logger.warning(
            "PEFT adapter artifacts found but optional dependency 'peft' is not installed; "
            "continuing with base decoder."
        )
        return model, "missing_peft"

    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path), "loaded"


def build_trl_sft_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    args,
    dataset_text_field: str = "text",
    max_seq_length: int = 512,
):
    """Construct TRL SFTTrainer across common TRL API versions."""
    require_optional_package("trl", "pip install trl")
    from trl import SFTTrainer

    kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params and "dataset_text_field" not in params:
        try:
            from trl import SFTConfig

            if not isinstance(args, SFTConfig):
                args = SFTConfig(
                    output_dir=getattr(args, "output_dir", "./checkpoints"),
                    per_device_train_batch_size=getattr(args, "per_device_train_batch_size", 1),
                    per_device_eval_batch_size=getattr(args, "per_device_eval_batch_size", 1),
                    gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
                    num_train_epochs=getattr(args, "num_train_epochs", 1),
                    learning_rate=getattr(args, "learning_rate", 2e-4),
                    logging_steps=getattr(args, "logging_steps", 1),
                    save_strategy=getattr(args, "save_strategy", "epoch"),
                    report_to=[],
                    dataset_text_field=dataset_text_field,
                    max_length=max_seq_length,
                    gradient_checkpointing=False,
                    bf16=False,
                    fp16=False,
                    use_cpu=bool(getattr(args, "use_cpu", False)),
                )
        except ImportError:
            pass
    kwargs["args"] = args
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = dataset_text_field
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq_length
    return SFTTrainer(**kwargs)


class HFCausalDecoderRuntime:
    """Reusable HF causal decoder wrapper with robust readiness/fallback signals."""

    def __init__(
        self,
        model_id: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        local_files_only: bool = False,
        prefer_cuda: bool = True,
        adapter_path: str | None = None,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.local_files_only = local_files_only
        self.adapter_path = adapter_path
        self.adapter_status = "not_requested"
        self.device = "cuda" if prefer_cuda and torch.cuda.is_available() else "cpu"

        self._ready = False
        self._error: str | None = None
        self._tokenizer = None
        self._model = None

        logger.info(
            "Initializing HF decoder | model_id=%s | local_files_only=%s | device=%s",
            model_id,
            local_files_only,
            self.device,
        )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            with hf_offline_if_requested(local_files_only):
                self._tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
                self._model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only)
            if adapter_path:
                self._model, self.adapter_status = load_peft_adapter_if_available(self._model, adapter_path)
            self._model = self._model.to(self.device)
            self._model.eval()
            self._ready = True
            logger.info("HF decoder ready | model_id=%s", model_id)
        except Exception as exc:
            self._error = str(exc)
            self._ready = False
            logger.warning("HF decoder unavailable | model_id=%s | reason=%s", model_id, self._error)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def generate(self, prompt: str, generation_cfg: dict[str, Any] | None = None) -> str:
        if not self._ready:
            raise RuntimeError(self._error or "HuggingFace decoder not available")

        generation_cfg = generation_cfg or {}
        max_new_tokens = int(generation_cfg.get("max_new_tokens", self.max_new_tokens))
        max_new_tokens_cap = generation_cfg.get("max_new_tokens_cap")
        if max_new_tokens_cap is not None:
            max_new_tokens = min(max_new_tokens, int(max_new_tokens_cap))
        temperature = float(generation_cfg.get("temperature", self.temperature))
        retry_count = int(generation_cfg.get("retry_count", 0))
        retry_backoff_s = float(generation_cfg.get("retry_backoff_s", 0.0))
        timeout_s = float(generation_cfg.get("timeout_s", 0.0))

        last_exc: Exception | None = None
        attempts = max(1, retry_count + 1)
        for attempt_idx in range(attempts):
            try:
                started = time.time()
                inputs = self._tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                do_sample = temperature > 0
                with torch.inference_mode():
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                elapsed = time.time() - started
                if timeout_s > 0 and elapsed > timeout_s:
                    raise TimeoutError(f"Generation exceeded timeout_s={timeout_s:.3f} (elapsed={elapsed:.3f}s)")

                text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
            except Exception as exc:
                last_exc = exc
                if attempt_idx >= attempts - 1:
                    break
                if retry_backoff_s > 0:
                    time.sleep(retry_backoff_s)

        assert last_exc is not None
        raise last_exc
