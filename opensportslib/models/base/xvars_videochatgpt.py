"""Clean-room X-VARS/Video-ChatGPT compatible VQA backend.

This module implements the tensor contract used by X-VARS: prompts contain a
contiguous video token block and the model receives CLIP
``video_spatio_temporal_features`` that are projected into the language-model
embedding space at those token positions.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

from opensportslib.core.config.accessors import get_data_sampling, get_model_load, get_train_execution
from opensportslib.core.utils.hf_runtime import (
    VIDEO_SPECIAL_TOKENS,
    _ensure_video_special_tokens,
    hf_offline_if_requested,
    load_peft_adapter_if_available,
)
from opensportslib.models.base.vqa import VQABaselineModel
from opensportslib.models.utils.vqa_prompting import build_prior_text, build_xvars_prompt

logger = logging.getLogger(__name__)


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
    def device(self):
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.base_lm.get_input_embeddings()

    def resize_token_embeddings(self, size: int):
        return self.base_lm.resize_token_embeddings(size)

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
        features = self.mm_projector(video_spatio_temporal_features.to(self.device, dtype=inputs_embeds.dtype))
        token_ids = self._video_token_ids(tokenizer)
        patch_id = token_ids["<vid_patch>"]
        start_id = token_ids["<vid_start>"]
        end_id = token_ids["<vid_end>"]

        for batch_idx, cur_input_ids in enumerate(input_ids):
            patch_positions = (cur_input_ids == patch_id).nonzero(as_tuple=False).flatten()
            if patch_positions.numel() == 0:
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
            inputs_embeds[batch_idx, patch_positions, :] = features[batch_idx]
        return inputs_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        video_spatio_temporal_features=None,
        tokenizer=None,
        **kwargs,
    ):
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
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )


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

        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        take = min(total, int(num_frames))
        if take <= 0:
            raise ValueError(f"No frames found in video: {video_path}")
        seg = float(total - 1) / take
        idx = [int((round(seg * i) + round(seg * (i + 1))) // 2) for i in range(take)]
        arr = vr.get_batch(idx).asnumpy()
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


class XVarsVideoChatGPTModel(nn.Module):
    """OpenSportsLib VQA model using the X-VARS multimodal tensor contract."""

    def __init__(self, config, model_id: str, projector_params: dict[str, Any] | None = None):
        super().__init__()
        self.config = config
        self.model_id = model_id
        self.baseline = VQABaselineModel(config)
        self._ready = False
        self._error = None
        self.tokenizer = None
        self.model = None

        exec_cfg = get_train_execution(config)
        xvars_cfg = _as_dict(exec_cfg.get("xvars"))
        hf_cfg = _as_dict(exec_cfg.get("hf"))
        projector_params = projector_params or {}
        self.video_token_len = int(xvars_cfg.get("video_token_len", projector_params.get("video_token_len", 356)))
        self.conv_mode = str(xvars_cfg.get("conv_mode", "video-chatgpt_v1"))
        self.feature_source = str(xvars_cfg.get("feature_source", "auto")).lower()
        self.raw_num_frames = resolve_xvars_raw_num_frames(config, xvars_cfg)
        self.raw_extractor = None

        local_files_only = bool(hf_cfg.get("local_files_only", False))
        prefer_cuda = bool(hf_cfg.get("prefer_cuda", True))
        adapter_path = get_model_load(config).get("checkpoint_path")
        projection_path = xvars_cfg.get("projection_path")
        mm_hidden_size = int(projector_params.get("input_dim", xvars_cfg.get("mm_hidden_size", 1024)))
        device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            with hf_offline_if_requested(local_files_only):
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only, use_fast=False)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                base_lm = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only)
            _ensure_video_special_tokens(self.tokenizer, base_lm)
            self.model = XVarsVideoChatGPTCausalLM.from_pretrained_projector(
                base_lm,
                projection_path,
                mm_hidden_size=mm_hidden_size,
            )
            if adapter_path:
                self.model, _status = load_peft_adapter_if_available(self.model, adapter_path)
            self.model = self.model.to(device).eval()
            self._ready = True
        except Exception as exc:
            self._error = str(exc)
            logger.warning("X-VARS VideoChatGPT backend unavailable | model_id=%s | reason=%s", model_id, self._error)

    def _build_prompt(self, sample: dict[str, Any], prompt_cfg: dict[str, Any] | None = None) -> str:
        prompt_cfg = prompt_cfg or {}
        system_prompt = str(
            prompt_cfg.get(
                "system_prompt",
                "You are an artificial intelligence assistant for visual football referee questions. Give short and helpful answers.",
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
        return build_xvars_prompt(
            system_prompt=system_prompt,
            question=str(sample.get("question", "")),
            prior_text=prior_text,
            video_token_len=token_len,
        )

    def _features_for_sample(self, sample: dict[str, Any], prompt_cfg: dict[str, Any] | None):
        features = sample.get("video_spatio_temporal_features")
        if features is None and self.feature_source in {"raw_video", "auto"}:
            video_path = sample.get("video_path")
            if video_path:
                if self.raw_extractor is None:
                    exec_cfg = get_train_execution(self.config)
                    hf_cfg = _as_dict(exec_cfg.get("hf"))
                    self.raw_extractor = XVarsRawVideoFeatureExtractor(prefer_cuda=bool(hf_cfg.get("prefer_cuda", True)))
                features = self.raw_extractor.extract(video_path, num_frames=self.raw_num_frames)
        if features is None:
            raise ValueError("Missing X-VARS video features and raw-video extraction was not available.")
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features, dtype=torch.float32)
        token_len = int((prompt_cfg or {}).get("video_token_len", self.video_token_len))
        if features.ndim != 2:
            raise ValueError(f"Expected X-VARS features [tokens, dim], got {tuple(features.shape)}")
        if int(features.shape[0]) != token_len:
            raise ValueError(
                f"X-VARS token mismatch: prompt video_token_len={token_len}, features rows={int(features.shape[0])}."
            )
        return features

    def generate_answer(self, sample: dict[str, Any], prompt_cfg=None, generation_cfg=None) -> str:
        generation_cfg = generation_cfg or {}
        fallback_policy = str(generation_cfg.get("fallback_policy", "none")).lower()
        if not self._ready:
            if fallback_policy == "baseline_on_failure":
                return self.baseline.generate_answer(sample, prompt_cfg=prompt_cfg, generation_cfg=generation_cfg)
            raise RuntimeError(self._error or "X-VARS VideoChatGPT backend is not ready")
        prompt = self._build_prompt(sample, prompt_cfg=prompt_cfg)
        features = self._features_for_sample(sample, prompt_cfg)
        encoded = self.tokenizer([prompt], return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        max_new_tokens = int(generation_cfg.get("max_new_tokens", 1024))
        temperature = float(generation_cfg.get("temperature", 0.2))
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    tokenizer=self.tokenizer,
                    attention_mask=attention_mask,
                    video_spatio_temporal_features=features.unsqueeze(0),
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            if output_ids.shape[-1] > input_ids.shape[-1]:
                decoded = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[-1]:], skip_special_tokens=True)[0]
            else:
                decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            return decoded.strip()
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
    "XVarsVideoChatGPTCausalLM",
    "XVarsVideoChatGPTModel",
]
