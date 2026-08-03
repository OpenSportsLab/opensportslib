"""Clean-room X-VARS/Video-ChatGPT compatible VQA backend.

This module implements the tensor contract used by X-VARS: prompts contain a
contiguous video token block and the model receives CLIP
``video_spatio_temporal_features`` that are projected into the language-model
embedding space at those token positions.
"""

from __future__ import annotations

import logging
import json
import os
from typing import Any

import torch
import torch.nn as nn

from opensportslib.core.config.accessors import (
    get_component_load_by_kind,
    get_component_params_by_kind,
    get_data_sampling,
    get_hf_cuda_device_index,
    get_hf_prefer_cuda,
    get_model_load,
    get_model_runtime_dtype,
    get_train_execution,
    get_vqa_feature_source,
    get_vqa_mm_hidden_size,
    get_vqa_xvars_feature_mode,
    get_xvars_infer_tokenizer_id,
    get_xvars_infer_video_token_len,
)
from opensportslib.core.utils.hf_runtime import (
    VIDEO_SPECIAL_TOKENS,
    _ensure_video_special_tokens,
    build_bitsandbytes_config,
    configure_generation_cache,
    hf_offline_if_requested,
    load_peft_adapter_if_available,
)
from opensportslib.models.base.video_chatgpt_compat import load_videochatgpt_compatible_causal_lm
from opensportslib.models.utils.vqa_prediction_priors import build_xvars_classifier_prior
from opensportslib.models.utils.xvars_clip_index import validate_xvars_feature_tensor
from opensportslib.models.utils.vqa_prompting import build_prior_text, build_xvars_prompt

logger = logging.getLogger(__name__)

XVARS_BASE_TOKEN_IDS = {
    "<vid_patch>": 32003,
    "<vid_start>": 32004,
    "<vid_end>": 32005,
}


_XVARS_DIRECT_PARITY_SAMPLE_FLAG = "_xvars_demo_parity_direct_infer"
_XVARS_DIRECT_STOP_STR = "</s>"


class _BaselineFallback:
    """Deterministic fallback answer builder for XVARS runtime failures."""

    def generate_answer(
        self,
        sample: dict[str, Any],
        prompt_cfg: dict[str, Any] | None = None,
        generation_cfg: dict[str, Any] | None = None,
    ) -> str:
        del generation_cfg
        prompt_cfg = prompt_cfg or {}
        style = str(prompt_cfg.get("style", "short")).lower()

        question = str(sample.get("question", "")).strip()
        labels = sample.get("labels", {}) or {}
        include_priors = bool(prompt_cfg.get("include_priors", True))
        priors = (
            build_prior_text(
                labels,
                sample.get("metadata", {}) or {},
                include_fields=prompt_cfg.get("prior_fields"),
            )
            if include_priors
            else ""
        )

        refs = sample.get("references") or []
        if refs:
            base = str(refs[0]).strip()
        elif priors:
            base = f"Available priors: {priors}."
        else:
            base = "Insufficient evidence to provide a definitive answer."

        if style == "detailed" and question:
            if priors:
                return f"Question: {question} Priors: {priors} Answer: {base}"
            return f"Question: {question} Answer: {base}"
        return base


def _is_direct_demo_parity_sample(sample: dict[str, Any]) -> bool:
    return bool(sample.get(_XVARS_DIRECT_PARITY_SAMPLE_FLAG))


def _build_direct_demo_parity_prompt_and_stop(
    sample: dict[str, Any],
    *,
    system_prompt: str,
    prior_text: str,
    video_token_len: int,
) -> tuple[str, str]:
    prompt = build_xvars_prompt(
        system_prompt=system_prompt,
        question=str(sample.get("question", "")),
        prior_text=prior_text,
        video_token_len=video_token_len,
    )
    return prompt, _XVARS_DIRECT_STOP_STR


class _KeywordsStoppingCriteria:
    """Original VideoChatGPT string stopping behavior without a UI dependency."""

    def __init__(self, keywords: list[str], tokenizer, input_ids: torch.Tensor):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            encoded = tokenizer(keyword).input_ids
            if isinstance(encoded, list) and len(encoded) == 1:
                self.keyword_ids.append(encoded[0])
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.start_len = None

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        del scores, kwargs
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            return False
        if any(int(output_ids[0, -1]) == int(keyword_id) for keyword_id in self.keyword_ids):
            return True
        output = self.tokenizer.batch_decode(output_ids[:, self.start_len :], skip_special_tokens=True)[0]
        return any(keyword in output for keyword in self.keywords)


def _configure_native_videochatgpt(base_lm, tokenizer, model_id: str) -> bool:
    """Configure the loaded checkpoint exactly as ``initialize_model`` does upstream."""

    token_ids = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in VIDEO_SPECIAL_TOKENS}
    if os.path.isabs(os.path.expanduser(str(model_id))) and os.path.basename(os.path.normpath(str(model_id))) == "base_model_videoChatGPT":
        if token_ids != XVARS_BASE_TOKEN_IDS:
            raise ValueError(
                "X-VARS base tokenizer IDs do not match the demo checkpoint: "
                f"expected {XVARS_BASE_TOKEN_IDS}, got {token_ids}."
            )

    get_model = getattr(base_lm, "get_model", None)
    native_model = get_model() if callable(get_model) else getattr(base_lm, "model", None)
    vision_config = getattr(native_model, "vision_config", None)
    if vision_config is None:
        return False
    vision_config.vid_patch_token = token_ids["<vid_patch>"]
    vision_config.use_vid_start_end = True
    vision_config.vid_start_token = token_ids["<vid_start>"]
    vision_config.vid_end_token = token_ids["<vid_end>"]
    logger.info("Configured native X-VARS token IDs | %s", token_ids)
    return True


def _restore_native_mm_projector(base_lm, device: torch.device) -> bool:
    """Restore the demo projector in fp16 when the decoder itself is quantized."""

    raw_state = _load_raw_mm_projector_state(base_lm)
    if raw_state is None:
        return False
    get_model = getattr(base_lm, "get_model", None)
    native_model = get_model() if callable(get_model) else getattr(base_lm, "model", None)
    if native_model is None or not hasattr(native_model, "mm_projector"):
        return False
    weight = raw_state["weight"]
    projector = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias="bias" in raw_state)
    projector.load_state_dict(raw_state, strict=True)
    native_model.mm_projector = projector.to(device=device, dtype=weight.dtype)
    logger.info("Restored native X-VARS mm_projector from unquantized checkpoint tensors.")
    return True


def _module_execution_device(module: nn.Module, fallback: torch.device) -> torch.device:
    """Resolve Accelerate's real execution device for CPU/disk-offloaded modules."""

    hook = getattr(module, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)
    try:
        parameter_device = next(module.parameters()).device
    except (StopIteration, AttributeError):
        parameter_device = fallback
    if parameter_device.type == "meta":
        return fallback
    return parameter_device


DEFAULT_XVARS_TARGET_MODULES = [
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


def resolve_xvars_raw_num_frames(config, xvars_cfg: dict[str, Any] | None = None) -> int:
    """Resolve raw-video frame sampling from DATA, falling back to legacy xvars config."""

    xvars_cfg = xvars_cfg or {}
    video_sampling = get_data_sampling(config)
    return int(video_sampling.get("num_frames", xvars_cfg.get("raw_num_frames", 100)))


def resolve_xvars_strict_sampling_cfg(config) -> dict[str, Any]:
    sampling = dict(get_data_sampling(config))
    if sampling:
        return sampling

    data = getattr(config, "DATA", None)
    common = getattr(data, "common", None) if data is not None else None
    common_inputs = getattr(common, "inputs", None) if common is not None else None
    video = getattr(common_inputs, "video", None) if common_inputs is not None else None
    nested_sampling = getattr(video, "sampling", None) if video is not None else None
    if isinstance(nested_sampling, dict):
        return dict(nested_sampling)
    if hasattr(nested_sampling, "__dict__"):
        return dict(vars(nested_sampling))
    return {}


def _runtime_torch_dtype(config) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(get_model_runtime_dtype(config, default="fp16").lower(), torch.float16)


def _get_embedded_mm_projector(base_lm) -> nn.Module | None:
    """Return a projector already stored in a loaded Video-ChatGPT model, if any."""

    candidates = [base_lm]
    get_model = getattr(base_lm, "get_model", None)
    if callable(get_model):
        try:
            candidates.append(get_model())
        except Exception:
            pass
    nested_model = getattr(base_lm, "model", None)
    if nested_model is not None:
        candidates.append(nested_model)

    for candidate in candidates:
        projector = getattr(candidate, "mm_projector", None)
        if isinstance(projector, nn.Module):
            return projector
    return None


def _load_raw_mm_projector_state(base_lm) -> dict[str, torch.Tensor] | None:
    """Load unquantized projector tensors from a local sharded safetensors checkpoint."""

    config = getattr(base_lm, "config", None)
    checkpoint_dir = os.path.expanduser(str(getattr(config, "_name_or_path", "") or ""))
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    tensor_to_file: dict[str, str] = {}
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            tensor_to_file = dict(json.load(f).get("weight_map", {}))
    else:
        single_file = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.isfile(single_file):
            tensor_to_file = {
                "model.mm_projector.weight": "model.safetensors",
                "model.mm_projector.bias": "model.safetensors",
            }

    source_keys = ("model.mm_projector.weight", "model.mm_projector.bias")
    if not all(key in tensor_to_file for key in source_keys):
        return None

    try:
        from safetensors import safe_open
    except ImportError:
        return None

    state = {}
    for source_key in source_keys:
        shard_path = os.path.join(checkpoint_dir, tensor_to_file[source_key])
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            state[source_key.rsplit(".", 1)[-1]] = shard.get_tensor(source_key)
    return state


class XVarsVideoChatGPTCausalLM(nn.Module):
    """Causal LM wrapper with X-VARS-compatible visual feature injection."""

    def __init__(self, base_lm, *, mm_hidden_size: int = 1024):
        super().__init__()
        self.base_lm = base_lm
        config = getattr(base_lm, "config", None)
        hidden_size = int(getattr(config, "hidden_size", None) or getattr(config, "n_embd", 0) or 0)
        if hidden_size <= 0:
            raise ValueError("Could not infer decoder hidden size for X-VARS backend.")
        self.hidden_size = hidden_size
        self.mm_hidden_size = int(mm_hidden_size)
        self.mm_projector = nn.Linear(self.mm_hidden_size, self.hidden_size)

    @property
    def config(self):
        return self.base_lm.config

    @property
    def generation_config(self):
        return self.base_lm.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.base_lm.generation_config = value

    @property
    def device(self):
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.base_lm.get_input_embeddings()

    def resize_token_embeddings(self, size: int):
        return self.base_lm.resize_token_embeddings(size)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        enable = getattr(self.base_lm, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise AttributeError("The wrapped VideoChatGPT decoder does not support gradient checkpointing.")
        return enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        disable = getattr(self.base_lm, "gradient_checkpointing_disable", None)
        if not callable(disable):
            raise AttributeError("The wrapped VideoChatGPT decoder does not support gradient checkpointing.")
        return disable()

    def enable_input_require_grads(self):
        enable = getattr(self.base_lm, "enable_input_require_grads", None)
        if callable(enable):
            return enable()
        return None

    def disable_input_require_grads(self):
        disable = getattr(self.base_lm, "disable_input_require_grads", None)
        if callable(disable):
            return disable()
        return None

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.base_lm.save_pretrained(output_dir)
        torch.save(
            {
                "mm_projector": self.mm_projector.state_dict(),
                "mm_hidden_size": self.mm_hidden_size,
                "hidden_size": self.hidden_size,
            },
            os.path.join(output_dir, "mm_projector.bin"),
        )

    @classmethod
    def from_pretrained_projector(cls, base_lm, projector_path: str | None = None, *, mm_hidden_size: int = 1024):
        model = cls(base_lm, mm_hidden_size=mm_hidden_size)
        embedded_projector = _get_embedded_mm_projector(base_lm)
        initialized = False
        if embedded_projector is not None:
            embedded_state = embedded_projector.state_dict()
            wrapper_state = model.mm_projector.state_dict()
            compatible = (
                set(embedded_state.keys()) == set(wrapper_state.keys())
                and all(tuple(embedded_state[k].shape) == tuple(wrapper_state[k].shape) for k in wrapper_state)
            )
            if compatible:
                model.mm_projector.load_state_dict(
                    {k: v.detach().cpu() for k, v in embedded_state.items()},
                    strict=True,
                )
                initialized = True
                logger.info("Initialized X-VARS mm_projector from loaded VideoChatGPT weights.")
        if embedded_projector is not None and not initialized:
            raw_state = _load_raw_mm_projector_state(base_lm)
            wrapper_state = model.mm_projector.state_dict()
            compatible = raw_state is not None and all(
                key in raw_state and tuple(raw_state[key].shape) == tuple(wrapper_state[key].shape)
                for key in wrapper_state
            )
            if compatible:
                model.mm_projector.load_state_dict(raw_state, strict=True)
                initialized = True
                logger.info("Initialized X-VARS mm_projector from unquantized checkpoint tensors.")
        if embedded_projector is not None and not initialized:
            logger.warning(
                "Loaded VideoChatGPT mm_projector does not match the wrapper and raw checkpoint tensors "
                "were unavailable; leaving the wrapper projector initialized unless projection_path is provided."
            )
        if projector_path:
            state = torch.load(os.path.expanduser(projector_path), map_location="cpu")
            if isinstance(state, dict) and "mm_projector" in state:
                state = state["mm_projector"]
            model.mm_projector.load_state_dict(state, strict=False)
        return model

    def _video_token_ids(self, tokenizer) -> dict[str, int]:
        return {tok: int(tokenizer.convert_tokens_to_ids(tok)) for tok in VIDEO_SPECIAL_TOKENS}

    def _prepare_inputs_embeds(self, input_ids: torch.Tensor, video_spatio_temporal_features: Any, tokenizer):
        if video_spatio_temporal_features is None:
            return self.get_input_embeddings()(input_ids)
        if not isinstance(video_spatio_temporal_features, torch.Tensor):
            video_spatio_temporal_features = torch.as_tensor(video_spatio_temporal_features, dtype=torch.float32)
        if video_spatio_temporal_features.ndim == 2:
            video_spatio_temporal_features = video_spatio_temporal_features.unsqueeze(0)
        if video_spatio_temporal_features.ndim != 3:
            raise ValueError(
                "video_spatio_temporal_features must be [batch, tokens, dim] or [tokens, dim], "
                f"got shape {tuple(video_spatio_temporal_features.shape)}"
            )
        if int(video_spatio_temporal_features.shape[-1]) != self.mm_hidden_size:
            raise ValueError(
                f"Expected X-VARS feature dim {self.mm_hidden_size}, "
                f"got {int(video_spatio_temporal_features.shape[-1])}"
            )

        input_ids = input_ids.to(self.device)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        projector_param = next(self.mm_projector.parameters())
        features = self.mm_projector(
            video_spatio_temporal_features.to(projector_param.device, dtype=projector_param.dtype)
        ).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        token_ids = self._video_token_ids(tokenizer)
        patch_id = token_ids["<vid_patch>"]
        start_id = token_ids["<vid_start>"]
        end_id = token_ids["<vid_end>"]

        new_input_embeds = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_embeds = inputs_embeds[batch_idx]
            patch_positions = (cur_input_ids == patch_id).nonzero(as_tuple=False).flatten()
            if patch_positions.numel() == 0:
                new_input_embeds.append(cur_input_embeds)
                continue
            start_positions = (cur_input_ids == start_id).nonzero(as_tuple=False).flatten()
            end_positions = (cur_input_ids == end_id).nonzero(as_tuple=False).flatten()
            if start_positions.numel() == 0 or end_positions.numel() == 0:
                raise ValueError("Missing required <vid_start>/<vid_end> tokens for X-VARS prompt.")
            if int(features.shape[1]) != int(patch_positions.numel()):
                raise ValueError(
                    f"Patch-feature mismatch: prompt has {int(patch_positions.numel())} <vid_patch> tokens "
                    f"but features have {int(features.shape[1])} rows."
                )
            mask_index_start = patch_positions[0]
            expected_positions = torch.arange(
                mask_index_start,
                mask_index_start + patch_positions.numel(),
                device=patch_positions.device,
                dtype=patch_positions.dtype,
            )
            if torch.any(patch_positions != expected_positions):
                raise ValueError("The <vid_patch> tokens should be consecutive for X-VARS prompts.")
            video_start_pos = start_positions[0]
            video_end_pos = end_positions[0]
            if mask_index_start != video_start_pos + 1 or patch_positions[-1] + 1 != video_end_pos:
                raise ValueError("X-VARS <vid_patch> block must be between <vid_start> and <vid_end>.")
            new_input_embeds.append(
                torch.cat(
                    (
                        cur_input_embeds[:mask_index_start],
                        features[batch_idx].to(device=cur_input_embeds.device),
                        cur_input_embeds[mask_index_start + patch_positions.numel():],
                    ),
                    dim=0,
                )
            )
        return torch.stack(new_input_embeds, dim=0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        video_spatio_temporal_features=None,
        tokenizer=None,
        **kwargs,
    ):
        del inputs_embeds
        kwargs.pop("inputs_embeds", None)
        if tokenizer is None:
            tokenizer = kwargs.pop("_xvars_tokenizer", None)
        if tokenizer is None:
            raise ValueError("XVarsVideoChatGPTCausalLM.forward requires tokenizer for video token ids.")
        inputs_embeds = self._prepare_inputs_embeds(input_ids, video_spatio_temporal_features, tokenizer)
        return self.base_lm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, input_ids, *, tokenizer, video_spatio_temporal_features=None, attention_mask=None, **kwargs):
        inputs_embeds = self._prepare_inputs_embeds(input_ids, video_spatio_temporal_features, tokenizer)
        return self.base_lm.generate(
            input_ids=input_ids.to(self.device),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
            }
        )
        if "tokenizer" in kwargs:
            model_inputs["tokenizer"] = kwargs["tokenizer"]
        if "_xvars_tokenizer" in kwargs:
            model_inputs["_xvars_tokenizer"] = kwargs["_xvars_tokenizer"]
        return model_inputs


class XVarsRawVideoFeatureExtractor:
    """CLIP ViT-L/14 feature extractor matching X-VARS inference behavior."""

    def __init__(self, *, vision_tower: str = "openai/clip-vit-large-patch14", prefer_cuda: bool = True):
        self.vision_tower = vision_tower
        self.prefer_cuda = prefer_cuda
        self._model = None
        self._processor = None
        self._device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self._processor = CLIPImageProcessor.from_pretrained(self.vision_tower)
        self._model = CLIPVisionModel.from_pretrained(self.vision_tower, torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32)
        self._model = self._model.to(self._device).eval()

    def load_video(self, video_path: str, *, num_frames: int = 100):
        from PIL import Image
        import numpy as np
        from decord import VideoReader, cpu

        if str(video_path).lower().endswith(".npy"):
            arr = np.load(video_path)
            if arr.ndim != 4:
                raise ValueError(f"Expected frames_npy clip with shape (T, H, W, C), got {tuple(arr.shape)}")
            if np.issubdtype(arr.dtype, np.floating):
                max_value = float(arr.max()) if arr.size else 0.0
                scale = 255.0 if max_value <= 1.0 else 1.0
                arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            total = int(arr.shape[0])
            take = min(total, int(num_frames))
            if take <= 0:
                raise ValueError(f"No frames found in npy clip: {video_path}")
            seg = float(total - 1) / take
            idx = [int((round(seg * i) + round(seg * (i + 1))) // 2) for i in range(take)]
            arr = arr[idx]
        else:
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            take = min(total, int(num_frames))
            if take <= 0:
                raise ValueError(f"No frames found in video: {video_path}")
            seg = float(total - 1) / take
            idx = [int((round(seg * i) + round(seg * (i + 1))) // 2) for i in range(take)]
            arr = vr.get_batch(idx).asnumpy()

        if take <= 0:
            raise ValueError(f"No frames found in video: {video_path}")
        if arr.shape[-3] != 224 or arr.shape[-2] != 224:
            ten = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
            ten = torch.nn.functional.interpolate(ten, size=(224, 224))
            arr = ten.permute(0, 2, 3, 1).to(torch.uint8).numpy()
        return [Image.fromarray(frame.astype(np.uint8)) for frame in arr]

    @staticmethod
    def spatio_temporal_tokens(frame_features: torch.Tensor, *, temporal_size: int = 100) -> torch.Tensor:
        t, _s, c = frame_features.shape
        temporal_tokens = torch.mean(frame_features, dim=1)
        if t < temporal_size:
            pad = torch.zeros(temporal_size - t, c, device=frame_features.device, dtype=frame_features.dtype)
            temporal_tokens = torch.cat((temporal_tokens, pad), dim=0)
        else:
            temporal_tokens = temporal_tokens[:temporal_size]
        spatial_tokens = torch.mean(frame_features, dim=0)
        return torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    def extract(self, video_path: str, *, num_frames: int = 100) -> torch.Tensor:
        self._ensure_loaded()
        frames = self.load_video(video_path, num_frames=num_frames)
        image_tensor = self._processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self._device)
        if self._device.type == "cuda":
            image_tensor = image_tensor.half()
        with torch.no_grad():
            outs = self._model(image_tensor, output_hidden_states=True)
            frame_features = outs.hidden_states[-2][:, 1:]
        return self.spatio_temporal_tokens(frame_features).cpu().to(torch.float32)


class _XVarsClassifierVisionTower(nn.Module):
    """Original X-VARS CLIP classifier architecture used by the demo."""

    def __init__(self, vision_tower: str):
        super().__init__()
        from transformers import CLIPVisionModel

        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower, low_cpu_mem_usage=True)
        self.inter = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        )
        self.fc_offence = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 4),
        )
        self.fc_action = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 8),
        )

    def forward(self, video: torch.Tensor):
        output = self.vision_tower(video, output_hidden_states=True)
        frame_features = output.hidden_states[-2][:, 1:]
        pooled = self.inter(torch.mean(output.pooler_output, dim=0).unsqueeze(0))
        return self.fc_offence(pooled).squeeze(0), self.fc_action(pooled).squeeze(0), frame_features


def _normalize_xvars_vision_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        key = str(key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("text_model.") or key in {
            "visual_projection.weight",
            "text_projection.weight",
            "logit_scale",
        }:
            continue

        aliases = [key]
        if key.startswith("vision_model."):
            aliases.append("vision_tower." + key)
        elif key.startswith("vision_tower.vision_model."):
            aliases.append("vision_tower." + key[len("vision_tower.vision_model.") :])
        elif key.startswith("vision_tower.") and not key.startswith(("vision_tower.vision_model.", "vision_tower.inter.", "vision_tower.fc_")):
            aliases.append("vision_tower.vision_model." + key[len("vision_tower.") :])

        for alias in aliases:
            normalized.setdefault(alias, value)
    return normalized


class XVarsStrictRawVideoFeatureExtractor:
    """Headless equivalent of ``x_vars_demo.py`` video upload and CLIP inference."""

    def __init__(
        self,
        *,
        weights_path: str,
        vision_tower: str = "openai/clip-vit-large-patch14",
        prefer_cuda: bool = True,
        start_frame: int | None = None,
        end_frame: int | None = None,
        input_fps: float | None = None,
        target_fps: float | None = None,
        temporal_size: int = 44,
    ):
        if not weights_path:
            raise ValueError(
                "Strict X-VARS raw-video inference requires the visual encoder weights_path "
                "(14_model.pth.tar)."
            )
        self.weights_path = os.path.abspath(os.path.expanduser(str(weights_path)))
        self.vision_tower = vision_tower
        self._device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
        self.start_frame = int(start_frame) if start_frame is not None else 63
        self.end_frame = int(end_frame) if end_frame is not None else 87
        self.input_fps = float(input_fps) if input_fps is not None else 25.0
        self.target_fps = float(target_fps) if target_fps is not None else 17.0
        self.temporal_size = int(temporal_size)
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"X-VARS visual encoder checkpoint not found: {self.weights_path}")
        from transformers import CLIPImageProcessor

        self._processor = CLIPImageProcessor.from_pretrained(self.vision_tower)
        model = _XVarsClassifierVisionTower(self.vision_tower)
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported X-VARS visual checkpoint format: {self.weights_path}")
        missing, unexpected = model.load_state_dict(_normalize_xvars_vision_state_dict(state_dict), strict=False)
        required_prefixes = ("vision_tower.", "inter.", "fc_offence.", "fc_action.")
        blocking_missing = [key for key in missing if key.startswith(required_prefixes)]
        if blocking_missing:
            raise RuntimeError(
                "X-VARS visual checkpoint is incompatible; missing required weights such as "
                f"{blocking_missing[:8]}"
            )
        if unexpected:
            logger.info("Ignored unexpected X-VARS visual checkpoint keys: %s", unexpected[:8])
        # The working X-VARS demo runs this classifier in float32 on CUDA.
        self._model = model.to(device=self._device, dtype=torch.float32).eval()

    def _strict_frame_window(self, frames: list, *, start_frame: int | None = None, end_frame: int | None = None, input_fps: float | None = None, target_fps: float | None = None) -> list:
        start_frame = self.start_frame if start_frame is None else int(start_frame)
        end_frame = self.end_frame if end_frame is None else int(end_frame)
        window = frames[start_frame:end_frame]
        if not window:
            return frames
        input_fps = self.input_fps if input_fps is None else float(input_fps)
        target_fps = self.target_fps if target_fps is None else float(target_fps)
        if not target_fps or input_fps <= 0 or target_fps >= input_fps:
            return window
        step = max(int(round(input_fps / target_fps)), 1)
        sampled = list(window[::step])
        if window[-1] not in sampled:
            if sampled:
                sampled[-1] = window[-1]
            else:
                sampled.append(window[-1])
        return sampled or window

    def spatio_temporal_tokens(self, frame_features: torch.Tensor) -> torch.Tensor:
        temporal = torch.mean(frame_features, dim=1)
        if temporal.shape[0] < self.temporal_size:
            padding = torch.zeros(
                self.temporal_size - temporal.shape[0],
                temporal.shape[1],
                device=temporal.device,
                dtype=temporal.dtype,
            )
            temporal = torch.cat((temporal, padding), dim=0)
        else:
            temporal = temporal[: self.temporal_size]
        spatial = torch.mean(frame_features, dim=0)
        return torch.cat((temporal, spatial), dim=0)

    def extract_with_prior(self, video_path: str) -> tuple[torch.Tensor, str]:
        self._ensure_loaded()
        loader = XVarsRawVideoFeatureExtractor(
            vision_tower=self.vision_tower,
            prefer_cuda=self._device.type == "cuda",
        )
        frames = loader.load_video(video_path, num_frames=100)
        frames = self._strict_frame_window(frames)
        image_tensor = self._processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        image_tensor = image_tensor.to(device=self._device, dtype=torch.float32)
        with torch.inference_mode():
            offence_logits, action_logits, frame_features = self._model(image_tensor)
        features = self.spatio_temporal_tokens(frame_features)
        prior = build_xvars_classifier_prior(
            int(torch.argmax(action_logits).item()),
            int(torch.argmax(offence_logits).item()),
        )
        return features.detach().cpu().to(torch.float32), prior

    def extract(self, video_path: str) -> torch.Tensor:
        features, _prior = self.extract_with_prior(video_path)
        return features


class XVarsVideoChatGPTModel(nn.Module):
    """OpenSportsLib VQA model using the X-VARS multimodal tensor contract."""

    def __init__(self, config, model_id: str, projector_params: dict[str, Any] | None = None):
        super().__init__()
        self.config = config
        self.model_id = model_id
        self.baseline = _BaselineFallback()
        self._ready = False
        self._error = None
        self.tokenizer = None
        self.model = None
        self.native_generation = False
        self.inference_device = torch.device("cpu")

        exec_cfg = get_train_execution(config)
        xvars_cfg = _as_dict(exec_cfg.get("xvars"))
        hf_cfg = _as_dict(exec_cfg.get("hf"))
        quant_cfg = _as_dict(exec_cfg.get("quantization"))
        projector_params = projector_params or {}
        self.video_token_len = get_xvars_infer_video_token_len(config)
        self.feature_mode = get_vqa_xvars_feature_mode(config, default="strict_xvars")
        self.conv_mode = str(xvars_cfg.get("conv_mode", "video-chatgpt_v1"))
        self.feature_source = get_vqa_feature_source(config, default="auto")
        self.raw_num_frames = resolve_xvars_raw_num_frames(config, xvars_cfg)
        self.strict_sampling_cfg = resolve_xvars_strict_sampling_cfg(config)
        self.raw_extractor = None

        encoder_params = get_component_params_by_kind(config, "encoder")
        encoder_load = get_component_load_by_kind(config, "encoder")
        self.vision_tower_name = str(
            encoder_params.get("vision_tower") or "openai/clip-vit-large-patch14"
        )
        self.vision_weights_path = str(
            encoder_load.get("weights_path")
            or encoder_params.get("weights_path")
            or xvars_cfg.get("vision_weights_path")
            or ""
        )

        local_files_only = bool(hf_cfg.get("local_files_only", False))
        prefer_cuda = get_hf_prefer_cuda(config, hf_cfg)
        cuda_device_index = get_hf_cuda_device_index(config, hf_cfg)
        adapter_path = get_model_load(config).get("checkpoint_path")
        projection_path = xvars_cfg.get("projection_path")
        mm_hidden_size = get_vqa_mm_hidden_size(config, default=1024)
        use_cuda = prefer_cuda and torch.cuda.is_available()
        if use_cuda and cuda_device_index is not None:
            torch.cuda.set_device(cuda_device_index)
        device = torch.device(
            f"cuda:{cuda_device_index}" if use_cuda and cuda_device_index is not None else ("cuda" if use_cuda else "cpu")
        )
        self.inference_device = device
        try:
            from transformers import AutoTokenizer

            bnb_config = build_bitsandbytes_config(quant_cfg)
            model_kwargs = {"local_files_only": local_files_only, "low_cpu_mem_usage": True}
            dispatched_model = False
            if bnb_config is not None:
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = {"": torch.cuda.current_device()} if use_cuda else None
            else:
                model_kwargs["torch_dtype"] = _runtime_torch_dtype(config)
                requested_device_map = hf_cfg.get("device_map")
                if requested_device_map:
                    model_kwargs["device_map"] = requested_device_map
                    dispatched_model = True
                    max_memory = {}
                    if use_cuda and hf_cfg.get("max_gpu_memory"):
                        max_memory[torch.cuda.current_device()] = str(hf_cfg["max_gpu_memory"])
                    if hf_cfg.get("max_cpu_memory"):
                        max_memory["cpu"] = str(hf_cfg["max_cpu_memory"])
                    if max_memory:
                        model_kwargs["max_memory"] = max_memory
                    offload_folder = hf_cfg.get("offload_folder")
                    if offload_folder:
                        offload_folder = os.path.abspath(os.path.expanduser(str(offload_folder)))
                        os.makedirs(offload_folder, exist_ok=True)
                        model_kwargs["offload_folder"] = offload_folder
                        model_kwargs["offload_state_dict"] = True

            with hf_offline_if_requested(local_files_only):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    get_xvars_infer_tokenizer_id(config),
                    local_files_only=local_files_only,
                    use_fast=False,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                base_lm = load_videochatgpt_compatible_causal_lm(model_id, **model_kwargs)
            configure_generation_cache(base_lm, enabled=True)
            _ensure_video_special_tokens(self.tokenizer, base_lm)
            native_configured = _configure_native_videochatgpt(base_lm, self.tokenizer, model_id)
            if adapter_path:
                self.model = XVarsVideoChatGPTCausalLM.from_pretrained_projector(
                    base_lm,
                    projection_path,
                    mm_hidden_size=mm_hidden_size,
                )
                if bnb_config is not None and hasattr(self.model, "mm_projector"):
                    self.model.mm_projector = self.model.mm_projector.to(device)
                self.model, adapter_status = load_peft_adapter_if_available(self.model, adapter_path)
                logger.info("X-VARS PEFT adapter | status=%s | path=%s", adapter_status, adapter_path)
            else:
                if not native_configured:
                    raise RuntimeError("Loaded X-VARS decoder does not expose native VideoChatGPT vision configuration.")
                self.model = base_lm
                self.native_generation = True
                if bnb_config is not None and not _restore_native_mm_projector(base_lm, device):
                    raise RuntimeError("Could not restore the native X-VARS mm_projector for quantized inference.")
                logger.info("Using native VideoChatGPT generation path | model_id=%s", model_id)
            if bnb_config is None and not dispatched_model:
                self.model = self.model.to(device)
            self.model = self.model.eval()
            self._ready = True
            logger.info("===== XVARS DEBUG =====")
            logger.info("native_generation=%s", self.native_generation)
            logger.info("feature_mode=%s", self.feature_mode)
            logger.info("feature_source=%s", self.feature_source)
            logger.info("video_token_len=%s", self.video_token_len)
            logger.info("vision_weights_path=%s", self.vision_weights_path)
            logger.info("model_class=%s", self.model.__class__.__name__)
            logger.info("=======================")
        except Exception as exc:
            self._error = str(exc)
            logger.warning("X-VARS VideoChatGPT backend unavailable | model_id=%s | reason=%s", model_id, self._error)

    def _build_prompt(self, sample: dict[str, Any], prompt_cfg: dict[str, Any] | None = None) -> str:
        prompt, _stop_str = self._build_prompt_and_stop(sample, prompt_cfg=prompt_cfg)
        return prompt

    def _build_prompt_and_stop(self, sample: dict[str, Any], prompt_cfg: dict[str, Any] | None = None) -> tuple[str, str]:
        prompt_cfg = prompt_cfg or {}
        system_prompt = str(
            prompt_cfg.get(
                "system_prompt",
                "You are an artificial intelligence assistant for visual question answering. Give short and helpful answers.",
            )
        )
        prior_text = ""
        if bool(prompt_cfg.get("include_priors", True)):
            prior_text = str(sample.get("prior_prediction_text", "")).strip() or build_prior_text(
                sample.get("labels", {}) or {},
                sample.get("metadata", {}) or {},
                include_fields=prompt_cfg.get("prior_fields"),
            )
        token_len = int(prompt_cfg.get("video_token_len", self.video_token_len))
        if self.native_generation and _is_direct_demo_parity_sample(sample):
            return _build_direct_demo_parity_prompt_and_stop(
                sample,
                system_prompt=system_prompt,
                prior_text=prior_text,
                video_token_len=token_len,
            )
        return (
            build_xvars_prompt(
                system_prompt=system_prompt,
                question=str(sample.get("question", "")),
                prior_text=prior_text,
                video_token_len=token_len,
            ),
            _XVARS_DIRECT_STOP_STR,
        )

    def _features_for_sample(self, sample: dict[str, Any], prompt_cfg: dict[str, Any] | None):
        features = sample.get("video_spatio_temporal_features")
        raw_sources = {"raw_video", "auto", "indexed_or_raw", "indexed_or_raw_clip"}
        if features is None and self.feature_source in raw_sources:
            video_path = sample.get("video_path")
            if video_path:
                if self.raw_extractor is None:
                    exec_cfg = get_train_execution(self.config)
                    hf_cfg = _as_dict(exec_cfg.get("hf"))
                    logger.info(
                        "feature_mode=%s feature_source=%s",
                        self.feature_mode,
                        self.feature_source,
                    )
                    if self.feature_mode == "strict_xvars":
                        self.raw_extractor = XVarsStrictRawVideoFeatureExtractor(
                            weights_path=self.vision_weights_path,
                            vision_tower=self.vision_tower_name,
                            prefer_cuda=get_hf_prefer_cuda(self.config, hf_cfg),
                            start_frame=self.strict_sampling_cfg.get("start_frame"),
                            end_frame=self.strict_sampling_cfg.get("end_frame"),
                            input_fps=self.strict_sampling_cfg.get("input_fps"),
                            target_fps=self.strict_sampling_cfg.get("target_fps"),
                            temporal_size=self.strict_sampling_cfg.get("temporal_size", 44),
                        )
                    else:
                        self.raw_extractor = XVarsRawVideoFeatureExtractor(
                            vision_tower=self.vision_tower_name,
                            prefer_cuda=get_hf_prefer_cuda(self.config, hf_cfg),
                        )
                if isinstance(self.raw_extractor, XVarsStrictRawVideoFeatureExtractor):
                    logger.info("USING STRICT XVARS EXTRACTOR")
                    features, classifier_prior = self.raw_extractor.extract_with_prior(video_path)
                    if classifier_prior:
                        sample["prior_prediction_text"] = classifier_prior
                    logger.info(
                        "X-VARS demo-parity visual context | video=%s | feature_shape=%s | prior=%s",
                        video_path,
                        tuple(features.shape),
                        classifier_prior,
                    )
                else:
                    logger.info("USING GENERIC CLIP EXTRACTOR")
                    features = self.raw_extractor.extract(video_path, num_frames=self.raw_num_frames)
        if features is None:
            raise ValueError("Missing X-VARS video features and raw-video extraction was not available.")
        token_len = int((prompt_cfg or {}).get("video_token_len", self.video_token_len))
        features = validate_xvars_feature_tensor(
            features,
            expected_tokens=token_len,
            context="X-VARS video_spatio_temporal_features",
        )
        return features

    def generate_answer(self, sample: dict[str, Any], prompt_cfg=None, generation_cfg=None) -> str:
        generation_cfg = generation_cfg or {}
        fallback_policy = str(generation_cfg.get("fallback_policy", "none")).lower()
        if not self._ready:
            if fallback_policy == "baseline_on_failure":
                return self.baseline.generate_answer(sample, prompt_cfg=prompt_cfg, generation_cfg=generation_cfg)
            raise RuntimeError(self._error or "X-VARS VideoChatGPT backend is not ready")
        resolved_sample = dict(sample)
        features = self._features_for_sample(resolved_sample, prompt_cfg)
        logger.info(
            "XVARS_FEATURES shape=%s mean=%f std=%f",
            tuple(features.shape),
            features.mean().item(),
            features.std().item(),
        )
        prompt, stop_str = self._build_prompt_and_stop(resolved_sample, prompt_cfg=prompt_cfg)
        #logger.info("XVARS_PROMPT=%s", prompt)
        logger.info(
            "X-VARS prompt context | id=%s | video_tokens=%s | question=%s | prior=%s",
            resolved_sample.get("id"),
            int(features.shape[0]),
            resolved_sample.get("question"),
            resolved_sample.get("prior_prediction_text", ""),
        )
        encoded = self.tokenizer([prompt], return_tensors="pt")
        input_embeddings = self.model.get_input_embeddings()
        inference_device = getattr(self, "inference_device", torch.device("cpu"))
        device = _module_execution_device(input_embeddings, inference_device)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        max_new_tokens = int(generation_cfg.get("max_new_tokens", 128))
        max_new_tokens_cap = generation_cfg.get("max_new_tokens_cap")
        if max_new_tokens_cap is not None:
            max_new_tokens = min(max_new_tokens, int(max_new_tokens_cap))
        temperature = float(generation_cfg.get("temperature", 0.2))
        try:
            with torch.inference_mode():
                if self.native_generation:
                    use_demo_parity_sampling = _is_direct_demo_parity_sample(resolved_sample)
                    effective_temperature = temperature
                    if use_demo_parity_sampling:
                        # The working x_vars_demo.py path uses greedy decoding and
                        # still passes a positive temperature value from the UI.
                        effective_temperature = effective_temperature if effective_temperature > 0 else 0.2
                    projector = _get_embedded_mm_projector(self.model)
                    logger.info(
                        "PROJECTOR=%s",
                        projector.__class__.__name__ if projector is not None else None
                    )
                    if projector is None:
                        raise RuntimeError("Native VideoChatGPT model is missing mm_projector.")
                    projector_param = next(projector.parameters())
                    projector_device = _module_execution_device(projector, inference_device)
                    native_features = features.unsqueeze(0).to(
                        device=projector_device,
                        dtype=projector_param.dtype,
                    )
                    stopping_criteria = _KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
                    output_ids = self.model.generate(
                        input_ids,
                        video_spatio_temporal_features=native_features,
                        do_sample=False if use_demo_parity_sampling else False,
                        temperature=effective_temperature,
                        max_new_tokens=max_new_tokens,
                        stopping_criteria=[stopping_criteria],
                    )
                else:
                    generation_kwargs = {
                        "do_sample": temperature > 0,
                        "max_new_tokens": max_new_tokens,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "repetition_penalty": float(generation_cfg.get("repetition_penalty", 1.0)),
                        "no_repeat_ngram_size": int(generation_cfg.get("no_repeat_ngram_size", 0)),
                    }
                    if temperature > 0:
                        generation_kwargs["temperature"] = temperature
                    output_ids = self.model.generate(
                        input_ids,
                        tokenizer=self.tokenizer,
                        attention_mask=attention_mask,
                        video_spatio_temporal_features=features.unsqueeze(0),
                        **generation_kwargs,
                    )
            if output_ids.shape[-1] > input_ids.shape[-1]:
                decoded = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[-1]:], skip_special_tokens=True)[0]
            else:
                decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            decoded = decoded.strip()
            if decoded.endswith(stop_str):
                decoded = decoded[: -len(stop_str)].strip()
            return decoded
        except Exception:
            if fallback_policy == "baseline_on_failure":
                return self.baseline.generate_answer(sample, prompt_cfg=prompt_cfg, generation_cfg=generation_cfg)
            raise


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items()}
    return {}


__all__ = [
    "DEFAULT_XVARS_TARGET_MODULES",
    "resolve_xvars_raw_num_frames",
    "XVarsRawVideoFeatureExtractor",
    "XVarsStrictRawVideoFeatureExtractor",
    "XVarsVideoChatGPTCausalLM",
    "XVarsVideoChatGPTModel",
    "build_xvars_classifier_prior",
    "_get_embedded_mm_projector",
]
