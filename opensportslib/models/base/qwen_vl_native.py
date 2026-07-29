"""Native Qwen VL VQA backend using end-to-end multimodal inputs."""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from opensportslib.core.config.accessors import (
    get_hf_cuda_device_index,
    get_model_load,
    get_train_execution,
    get_vqa_generation_cfg,
    get_vqa_native_max_pixels,
    get_vqa_native_min_pixels,
    get_vqa_native_num_frames,
    get_vqa_native_visual_input_mode,
    get_xvars_infer_tokenizer_id,
)
from opensportslib.core.utils.hf_runtime import (
    build_bitsandbytes_config,
    configure_generation_cache,
    hf_offline_if_requested,
    load_peft_adapter_if_available,
)
from opensportslib.models.utils.vqa_prompting import build_prior_text

logger = logging.getLogger(__name__)


class NativeQwenVLInvalidRowError(ValueError):
    """Raised when a native Qwen VL SFT row carries no usable supervision."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = dict(context or {})


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items()}
    return {}


def _move_batch_to_device(payload: Any, device: torch.device):
    if hasattr(payload, "to"):
        try:
            return payload.to(device)
        except Exception:
            pass
    if isinstance(payload, dict):
        moved = {}
        for key, value in payload.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved
    return payload


def _runtime_torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(str(dtype_name or "fp16").lower(), torch.float16)


class NativeQwenVLDataCollator:
    """Collate tokenized native-Qwen-VL rows."""

    _CONCAT_KEYS = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}

    def __call__(self, instances):
        if not instances:
            return {}
        first = instances[0]
        out = {}
        for key, value in first.items():
            values = [row[key] for row in instances]
            if torch.is_tensor(value):
                if key in self._CONCAT_KEYS:
                    out[key] = torch.cat(values, dim=0)
                else:
                    out[key] = torch.stack(values, dim=0)
            else:
                out[key] = values
        return out


class NativeQwenVLTrainer:
    """Minimal trainer wrapper for native Qwen VL models."""

    def __init__(
        self,
        *,
        model,
        args,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        callbacks=None,
    ):
        from transformers import Trainer

        class _Trainer(Trainer):
            _warned_grad_norm_nan = False

            def _inspect_current_gradients(self) -> dict[str, Any]:
                finite = True
                param_name = None
                grad_count = 0
                for name, param in model.named_parameters():
                    grad = getattr(param, "grad", None)
                    if grad is None:
                        continue
                    grad_count += 1
                    if torch.isfinite(grad).all():
                        continue
                    finite = False
                    param_name = str(name)
                    break
                return {
                    "finite": finite,
                    "parameter": param_name,
                    "grad_count": grad_count,
                }

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                del kwargs
                labels = inputs.get("labels")
                outputs = model(**inputs)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                return (loss, outputs) if return_outputs else loss

            def training_step(self, model, inputs, *args, **kwargs):
                loss = super().training_step(model, inputs, *args, **kwargs)
                self._last_grad_health = self._inspect_current_gradients()
                return loss

            def log(self, logs: dict[str, float], *args, **kwargs) -> None:
                grad_norm = logs.get("grad_norm")
                loss = logs.get("loss")
                if grad_norm is not None:
                    try:
                        grad_norm_is_finite = torch.isfinite(torch.as_tensor(grad_norm)).item()
                    except Exception:
                        grad_norm_is_finite = True
                    if not grad_norm_is_finite and not self._warned_grad_norm_nan:
                        loss_is_finite = True
                        if loss is not None:
                            try:
                                loss_is_finite = torch.isfinite(torch.as_tensor(loss)).item()
                            except Exception:
                                loss_is_finite = True
                        grad_health = getattr(self, "_last_grad_health", None) or {}
                        if loss_is_finite and bool(grad_health.get("finite", False)):
                            logger.warning(
                                "Native Qwen VL trainer reported grad_norm=nan while loss remained finite; "
                                "parameter gradients inspected after backward were finite. Treating this as "
                                "grad-norm reporting or clipping instability."
                            )
                        else:
                            logger.warning(
                                "Native Qwen VL trainer reported grad_norm=nan and detected non-finite or unavailable "
                                "gradients during debug inspection | offending_parameter=%s | grads_seen=%s",
                                grad_health.get("parameter"),
                                grad_health.get("grad_count", 0),
                            )
                        self._warned_grad_norm_nan = True
                return super().log(logs, *args, **kwargs)

        self._trainer = _Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator or NativeQwenVLDataCollator(),
            callbacks=callbacks,
        )
        self.model = model

    def train(self, resume_from_checkpoint: str | None = None):
        return self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def save_state(self):
        return self._trainer.save_state()


class QwenVLNativeModel(nn.Module):
    """End-to-end native Qwen VL inference wrapper."""

    def __init__(self, config, model_id: str):
        super().__init__()
        self.config = config
        self.model_id = model_id
        self._ready = False
        self._error = None
        self.processor = None
        self.model = None
        self.inference_device = torch.device("cpu")
        self.visual_input_mode = get_vqa_native_visual_input_mode(config, default="frames")
        self.num_frames = get_vqa_native_num_frames(config, default=8)

        exec_cfg = get_train_execution(config)
        hf_cfg = _as_dict(exec_cfg.get("hf"))
        quant_cfg = _as_dict(exec_cfg.get("quantization"))
        local_files_only = bool(hf_cfg.get("local_files_only", False))
        prefer_cuda = bool(hf_cfg.get("prefer_cuda", True))
        cuda_device_index = get_hf_cuda_device_index(config, hf_cfg)
        use_cuda = prefer_cuda and torch.cuda.is_available()
        if use_cuda and cuda_device_index is not None:
            torch.cuda.set_device(cuda_device_index)
        device = torch.device(
            f"cuda:{cuda_device_index}" if use_cuda and cuda_device_index is not None else ("cuda" if use_cuda else "cpu")
        )
        self.inference_device = device
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            bnb_config = build_bitsandbytes_config(quant_cfg)
            processor_kwargs = {"local_files_only": local_files_only}
            min_pixels = get_vqa_native_min_pixels(config)
            max_pixels = get_vqa_native_max_pixels(config)
            if min_pixels is not None:
                processor_kwargs["min_pixels"] = min_pixels
            if max_pixels is not None:
                processor_kwargs["max_pixels"] = max_pixels
            model_kwargs = {"local_files_only": local_files_only, "low_cpu_mem_usage": True}
            dispatched_model = False
            if bnb_config is not None:
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = {"": torch.cuda.current_device()} if use_cuda else None
            else:
                model_kwargs["torch_dtype"] = _runtime_torch_dtype(hf_cfg.get("dtype", "fp16"))
                requested_device_map = hf_cfg.get("device_map")
                if requested_device_map:
                    model_kwargs["device_map"] = requested_device_map
                    dispatched_model = True

            tokenizer_id = get_xvars_infer_tokenizer_id(config, default=model_id)
            with hf_offline_if_requested(local_files_only):
                self.processor = AutoProcessor.from_pretrained(tokenizer_id, **processor_kwargs)
                self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
            configure_generation_cache(self.model, enabled=True)
            adapter_path = get_model_load(config).get("checkpoint_path")
            self.model, adapter_status = load_peft_adapter_if_available(self.model, adapter_path)
            if bnb_config is None and not dispatched_model:
                self.model = self.model.to(device)
            self.model = self.model.eval()
            self._ready = True
            logger.info(
                "Initialized Qwen VL native backend | model_id=%s | visual_input_mode=%s | num_frames=%s | adapter_status=%s",
                model_id,
                self.visual_input_mode,
                self.num_frames,
                adapter_status,
            )
        except Exception as exc:
            self._error = str(exc)
            logger.warning("Qwen VL native backend unavailable | model_id=%s | reason=%s", model_id, self._error)

    def _sample_frames(self, video_path: str, *, num_frames: int | None = None) -> list[np.ndarray]:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video for native VL sampling: {video_path}")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                raise ValueError(f"Video has no readable frames: {video_path}")
            target = max(1, int(num_frames or self.num_frames))
            indices = np.linspace(0, max(frame_count - 1, 0), num=target, dtype=int)
            frames: list[np.ndarray] = []
            for idx in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not frames:
                raise ValueError(f"Could not sample frames from video: {video_path}")
            return frames
        finally:
            capture.release()

    def _select_frame_indices(self, frame_count: int, *, num_frames: int | None = None) -> np.ndarray:
        frame_count = int(frame_count)
        if frame_count <= 0:
            raise ValueError("Expected at least one frame for native VL input.")
        target = max(1, int(num_frames or self.num_frames))
        sample_count = min(frame_count, target)
        return np.linspace(0, frame_count - 1, num=sample_count, dtype=int)

    def _normalize_frame_array(self, frame_array: np.ndarray) -> np.ndarray:
        frame_array = np.asarray(frame_array)
        if frame_array.ndim == 4 and frame_array.shape[-1] in {1, 3, 4}:
            return frame_array
        if frame_array.ndim == 4 and frame_array.shape[1] in {1, 3, 4}:
            return np.transpose(frame_array, (0, 2, 3, 1))
        if frame_array.ndim == 3 and frame_array.shape[-1] in {1, 3, 4}:
            return frame_array[None, ...]
        if frame_array.ndim == 3 and frame_array.shape[0] in {1, 3, 4}:
            return np.transpose(frame_array, (1, 2, 0))[None, ...]
        raise ValueError(
            "Unsupported native VL .npy shape. Expected frame arrays shaped like "
            "(T, H, W, C), (T, C, H, W), (H, W, C), or (C, H, W)."
        )

    def _coerce_rgb_frames(self, frame_array: np.ndarray) -> list[np.ndarray]:
        normalized = self._normalize_frame_array(frame_array)
        indices = self._select_frame_indices(normalized.shape[0])
        selected = normalized[indices]
        frames: list[np.ndarray] = []
        for frame in selected:
            frame = np.asarray(frame)
            if frame.ndim != 3:
                raise ValueError("Native VL .npy frame entries must be rank-3 image arrays.")
            channels = frame.shape[-1]
            if channels == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif channels == 4:
                frame = frame[..., :3]
            elif channels != 3:
                raise ValueError(
                    "Unsupported native VL .npy channel count. Expected 1, 3, or 4 channels per frame."
                )
            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating):
                    scale = 255.0 if float(np.nanmax(frame)) <= 1.0 else 1.0
                    frame = np.clip(frame * scale, 0.0, 255.0)
                else:
                    frame = np.clip(frame, 0, 255)
                frame = frame.astype(np.uint8)
            frames.append(np.ascontiguousarray(frame))
        if not frames:
            raise ValueError("Native VL .npy input did not yield any frames.")
        return frames

    def _load_npy_frames(self, npy_path: str) -> list[np.ndarray]:
        try:
            frame_array = np.load(npy_path, allow_pickle=True)
        except Exception as exc:
            raise ValueError(f"Could not load native VL .npy frames: {npy_path}") from exc
        try:
            return self._coerce_rgb_frames(frame_array)
        except ValueError as exc:
            raise ValueError(f"{exc} Path: {npy_path}") from exc

    def _resolve_visual_inputs(self, sample: dict[str, Any]) -> tuple[list[np.ndarray], str]:
        frame_arrays = sample.get("video_frames")
        if frame_arrays:
            return self._coerce_rgb_frames(np.asarray(frame_arrays)), "frames"

        frame_paths = sample.get("frame_paths") or []
        if frame_paths:
            frames = []
            for path in frame_paths:
                image = cv2.imread(str(path))
                if image is None:
                    continue
                frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if frames:
                return frames, "frames"

        video_path = str(sample.get("video_path") or "").strip()
        if not video_path:
            raise ValueError("Native Qwen VL inference requires either frame_paths, video_frames, or video_path.")
        if video_path.lower().endswith(".npy"):
            return self._load_npy_frames(video_path), "frames"

        if self.visual_input_mode == "video_with_frames_fallback":
            return [video_path], "video"
        return self._sample_frames(video_path), "frames"

    def _build_messages(
        self,
        sample: dict[str, Any],
        prompt_cfg: dict[str, Any] | None = None,
        *,
        visual_type: str,
        visual_count: int = 1,
    ) -> list[dict[str, Any]]:
        prompt_cfg = prompt_cfg or {}
        system_prompt = str(
            prompt_cfg.get(
                "system_prompt",
                "You are an artificial intelligence assistant for visual question answering. Give short and helpful answers.",
            )
        ).strip()
        prior_text = ""
        if bool(prompt_cfg.get("include_priors", True)):
            prior_text = str(sample.get("prior_prediction_text", "")).strip() or build_prior_text(
                sample.get("labels", {}) or {},
                sample.get("metadata", {}) or {},
                include_fields=prompt_cfg.get("prior_fields"),
            )
        question = str(sample.get("question", "")).strip()
        text_parts = [question]
        if prior_text:
            text_parts.append(f"Referee priors: {prior_text}")
        text_parts.append("Answer clearly in a short sentence or two, and include reasoning where applicable.")
        content = []
        if visual_type == "video":
            content.append({"type": "video"})
        else:
            content.extend({"type": "image"} for _ in range(max(int(visual_count), 1)))
        content.append({"type": "text", "text": "\n".join(part for part in text_parts if part).strip()})
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

    def build_training_inputs(
        self,
        sample: dict[str, Any],
        *,
        prompt_cfg: dict[str, Any] | None = None,
        answer_text: str,
        max_seq_length: int = 1024,
        reference_index: int = 0,
    ) -> dict[str, Any]:
        if not self._ready:
            raise RuntimeError(self._error or "Qwen VL native backend is not ready")
        visual_inputs, visual_type = self._resolve_visual_inputs(sample)
        if visual_type == "video":
            visual_inputs = self._sample_frames(str(sample.get("video_path") or ""), num_frames=self.num_frames)
            visual_type = "frames"
        messages = self._build_messages(
            sample,
            prompt_cfg=prompt_cfg,
            visual_type=visual_type,
            visual_count=len(visual_inputs) if visual_type != "video" else 1,
        )
        assistant_answer = str(answer_text).strip()
        if not assistant_answer:
            raise ValueError("Native Qwen VL training row requires a non-empty answer.")
        sample_id = str(sample.get("id", ""))
        question = str(sample.get("question", ""))
        full_messages = list(messages) + [{"role": "assistant", "content": [{"type": "text", "text": assistant_answer}]}]

        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_text = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        process_kwargs = {
            "text": [full_text],
            "padding": "max_length",
            "truncation": True,
            "max_length": int(max_seq_length),
            "return_tensors": "pt",
        }
        if visual_type == "video":
            process_kwargs["videos"] = visual_inputs
        else:
            process_kwargs["images"] = [visual_inputs]
        encoded = self.processor(**process_kwargs)
        labels = encoded["input_ids"].clone()

        prompt_kwargs = dict(process_kwargs)
        prompt_kwargs["text"] = [prompt_text]
        prompt_inputs = self.processor(**prompt_kwargs)
        prompt_len = int(prompt_inputs["attention_mask"][0].sum().item())
        full_len = int(encoded["attention_mask"][0].sum().item())
        context = {
            "sample_id": sample_id,
            "question": question,
            "reference_index": int(reference_index),
            "prompt_length": prompt_len,
            "full_length": full_len,
            "max_seq_length": int(max_seq_length),
            "answer_preview": assistant_answer[:120],
            "answer_length": len(assistant_answer),
        }
        if full_len <= prompt_len:
            raise NativeQwenVLInvalidRowError(
                "Native Qwen VL training row lost assistant supervision after truncation "
                f"(full_len={full_len}, prompt_len={prompt_len}).",
                context=context,
            )
        labels[:, :prompt_len] = -100
        labels = labels.masked_fill(encoded["attention_mask"] == 0, -100)
        if bool(torch.all(labels == -100).item()):
            raise NativeQwenVLInvalidRowError(
                "Native Qwen VL training row has all labels masked after prompt and padding masking.",
                context=context,
            )
        out = {key: value[0] if torch.is_tensor(value) and value.shape[0] == 1 else value for key, value in encoded.items()}
        out["labels"] = labels[0]
        return out

    def generate_answer(self, sample: dict[str, Any], prompt_cfg=None, generation_cfg=None) -> str:
        generation_cfg = generation_cfg or get_vqa_generation_cfg(self.config)
        if not self._ready:
            raise RuntimeError(self._error or "Qwen VL native backend is not ready")

        visual_inputs, visual_type = self._resolve_visual_inputs(sample)
        last_error = None
        attempts = [(visual_inputs, visual_type)]
        if visual_type == "video":
            attempts.append((self._sample_frames(str(sample.get("video_path") or ""), num_frames=self.num_frames), "frames"))

        for current_visuals, current_visual_type in attempts:
            try:
                messages = self._build_messages(
                    sample,
                    prompt_cfg=prompt_cfg,
                    visual_type=current_visual_type,
                    visual_count=len(current_visuals) if current_visual_type != "video" else 1,
                )
                prompt_text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                process_kwargs = {
                    "text": [prompt_text],
                    "return_tensors": "pt",
                }
                if current_visual_type == "video":
                    process_kwargs["videos"] = current_visuals
                else:
                    process_kwargs["images"] = [current_visuals]
                inputs = self.processor(**process_kwargs)
                device = getattr(self.model, "device", self.inference_device)
                moved_inputs = _move_batch_to_device(inputs, device)
                max_new_tokens = int(generation_cfg.get("max_new_tokens", 128))
                max_new_tokens_cap = generation_cfg.get("max_new_tokens_cap")
                if max_new_tokens_cap is not None:
                    max_new_tokens = min(max_new_tokens, int(max_new_tokens_cap))
                temperature = float(generation_cfg.get("temperature", 0.0))
                generation_kwargs = {
                    "do_sample": temperature > 0,
                    "max_new_tokens": max_new_tokens,
                }
                if temperature > 0:
                    generation_kwargs["temperature"] = temperature
                with torch.inference_mode():
                    output_ids = self.model.generate(**moved_inputs, **generation_kwargs)
                input_ids = moved_inputs["input_ids"]
                generated = output_ids[:, input_ids.shape[-1]:] if output_ids.shape[-1] > input_ids.shape[-1] else output_ids
                decode = getattr(self.processor, "decode", None)
                if not callable(decode):
                    decode = getattr(getattr(self.processor, "tokenizer", None), "decode", None)
                if not callable(decode):
                    raise AttributeError("Processor does not expose a decode() method for native Qwen VL outputs.")
                decoded = decode(generated[0], skip_special_tokens=True)
                return str(decoded).strip()
            except Exception as exc:
                last_error = exc
                if current_visual_type != "video":
                    break
                logger.info("Native Qwen VL direct-video path failed; retrying with sampled frames | reason=%s", exc)
        raise last_error if last_error is not None else RuntimeError("Qwen VL native generation failed.")


__all__ = [
    "NativeQwenVLDataCollator",
    "NativeQwenVLTrainer",
    "QwenVLNativeModel",
]
