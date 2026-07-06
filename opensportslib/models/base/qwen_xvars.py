"""Qwen-backed VQA inference using the X-VARS visual feature contract."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

from opensportslib.core.config.accessors import (
    get_component_load_by_kind,
    get_component_params_by_kind,
    get_hf_cuda_device_index,
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
)
from opensportslib.models.base.xvars_videochatgpt import (
    XVarsRawVideoFeatureExtractor,
    XVarsStrictRawVideoFeatureExtractor,
    _BaselineFallback,
    _KeywordsStoppingCriteria,
    _build_direct_demo_parity_prompt_and_stop,
    _is_direct_demo_parity_sample,
    _module_execution_device,
    resolve_xvars_raw_num_frames,
    resolve_xvars_strict_sampling_cfg,
)
from opensportslib.models.utils.vqa_prompting import build_prior_text, build_xvars_prompt
from opensportslib.models.utils.xvars_clip_index import validate_xvars_feature_tensor

logger = logging.getLogger(__name__)


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items()}
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


class QwenXVarsCausalLM(nn.Module):
    """Causal LM wrapper that injects X-VARS visual features into a base Qwen LM."""

    def __init__(self, base_lm, *, mm_hidden_size: int = 1024):
        super().__init__()
        self.base_lm = base_lm
        config = getattr(base_lm, "config", None)
        hidden_size = int(getattr(config, "hidden_size", None) or getattr(config, "n_embd", 0) or 0)
        if hidden_size <= 0:
            raise ValueError("Could not infer decoder hidden size for Qwen X-VARS backend.")
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
                raise ValueError("Missing required <vid_start>/<vid_end> tokens for Qwen X-VARS prompt.")
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
                raise ValueError("The <vid_patch> tokens should be consecutive for Qwen X-VARS prompts.")
            video_start_pos = start_positions[0]
            video_end_pos = end_positions[0]
            if mask_index_start != video_start_pos + 1 or patch_positions[-1] + 1 != video_end_pos:
                raise ValueError("Qwen X-VARS <vid_patch> block must be between <vid_start> and <vid_end>.")
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
            raise ValueError("QwenXVarsCausalLM.forward requires tokenizer for video token ids.")
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
        return model_inputs


class QwenXVarsModel(nn.Module):
    """Inference-only Qwen VQA model reusing the X-VARS feature pipeline."""

    def __init__(self, config, model_id: str, projector_params: dict[str, Any] | None = None):
        super().__init__()
        self.config = config
        self.model_id = model_id
        self.baseline = _BaselineFallback()
        self._ready = False
        self._error = None
        self.tokenizer = None
        self.model = None
        self.inference_device = torch.device("cpu")

        exec_cfg = get_train_execution(config)
        xvars_cfg = _as_dict(exec_cfg.get("xvars"))
        hf_cfg = _as_dict(exec_cfg.get("hf"))
        quant_cfg = _as_dict(exec_cfg.get("quantization"))
        projector_params = projector_params or {}
        self.video_token_len = get_xvars_infer_video_token_len(config)
        self.feature_mode = get_vqa_xvars_feature_mode(config, default="strict_xvars")
        self.feature_source = get_vqa_feature_source(config, default="auto")
        self.raw_num_frames = resolve_xvars_raw_num_frames(config, xvars_cfg)
        self.strict_sampling_cfg = resolve_xvars_strict_sampling_cfg(config)
        self.raw_extractor = None

        encoder_params = get_component_params_by_kind(config, "encoder")
        encoder_load = get_component_load_by_kind(config, "encoder")
        self.vision_tower_name = str(encoder_params.get("vision_tower") or "openai/clip-vit-large-patch14")
        self.vision_weights_path = str(
            encoder_load.get("weights_path")
            or encoder_params.get("weights_path")
            or xvars_cfg.get("vision_weights_path")
            or ""
        )

        local_files_only = bool(hf_cfg.get("local_files_only", False))
        prefer_cuda = bool(hf_cfg.get("prefer_cuda", True))
        cuda_device_index = get_hf_cuda_device_index(config, hf_cfg)
        mm_hidden_size = int(projector_params.get("input_dim") or get_vqa_mm_hidden_size(config, default=1024))
        use_cuda = prefer_cuda and torch.cuda.is_available()
        if use_cuda and cuda_device_index is not None:
            torch.cuda.set_device(cuda_device_index)
        device = torch.device(
            f"cuda:{cuda_device_index}" if use_cuda and cuda_device_index is not None else ("cuda" if use_cuda else "cpu")
        )
        self.inference_device = device
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

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
                    get_xvars_infer_tokenizer_id(config, default=model_id),
                    local_files_only=local_files_only,
                    use_fast=False,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                base_lm = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            configure_generation_cache(base_lm, enabled=True)
            _ensure_video_special_tokens(self.tokenizer, base_lm)
            self.model = QwenXVarsCausalLM(base_lm, mm_hidden_size=mm_hidden_size)
            if bnb_config is None and not dispatched_model:
                self.model = self.model.to(device)
            self.model = self.model.eval()
            self._ready = True
            logger.info(
                "Initialized Qwen X-VARS backend | model_id=%s | feature_mode=%s | feature_source=%s | video_token_len=%s",
                model_id,
                self.feature_mode,
                self.feature_source,
                self.video_token_len,
            )
        except Exception as exc:
            self._error = str(exc)
            logger.warning("Qwen X-VARS backend unavailable | model_id=%s | reason=%s", model_id, self._error)

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
        if _is_direct_demo_parity_sample(sample):
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
            "</s>",
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
                    if self.feature_mode == "strict_xvars":
                        self.raw_extractor = XVarsStrictRawVideoFeatureExtractor(
                            weights_path=self.vision_weights_path,
                            vision_tower=self.vision_tower_name,
                            prefer_cuda=bool(hf_cfg.get("prefer_cuda", True)),
                            start_frame=self.strict_sampling_cfg.get("start_frame"),
                            end_frame=self.strict_sampling_cfg.get("end_frame"),
                            input_fps=self.strict_sampling_cfg.get("input_fps"),
                            target_fps=self.strict_sampling_cfg.get("target_fps"),
                            temporal_size=self.strict_sampling_cfg.get("temporal_size", 44),
                        )
                    else:
                        self.raw_extractor = XVarsRawVideoFeatureExtractor(
                            vision_tower=self.vision_tower_name,
                            prefer_cuda=bool(hf_cfg.get("prefer_cuda", True)),
                        )
                if isinstance(self.raw_extractor, XVarsStrictRawVideoFeatureExtractor):
                    features, classifier_prior = self.raw_extractor.extract_with_prior(video_path)
                    if classifier_prior:
                        sample["prior_prediction_text"] = classifier_prior
                else:
                    features = self.raw_extractor.extract(video_path, num_frames=self.raw_num_frames)
        if features is None:
            raise ValueError("Missing X-VARS video features and raw-video extraction was not available.")
        token_len = int((prompt_cfg or {}).get("video_token_len", self.video_token_len))
        return validate_xvars_feature_tensor(
            features,
            expected_tokens=token_len,
            context="Qwen X-VARS video_spatio_temporal_features",
        )

    def generate_answer(self, sample: dict[str, Any], prompt_cfg=None, generation_cfg=None) -> str:
        generation_cfg = generation_cfg or {}
        fallback_policy = str(generation_cfg.get("fallback_policy", "none")).lower()
        if not self._ready:
            if fallback_policy == "baseline_on_failure":
                return self.baseline.generate_answer(sample, prompt_cfg=prompt_cfg, generation_cfg=generation_cfg)
            raise RuntimeError(self._error or "Qwen X-VARS backend is not ready")

        resolved_sample = dict(sample)
        features = self._features_for_sample(resolved_sample, prompt_cfg)
        prompt, stop_str = self._build_prompt_and_stop(resolved_sample, prompt_cfg=prompt_cfg)
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
        temperature = float(generation_cfg.get("temperature", 0.0))
        generation_kwargs = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": float(generation_cfg.get("repetition_penalty", 1.0)),
            "no_repeat_ngram_size": int(generation_cfg.get("no_repeat_ngram_size", 0)),
            "stopping_criteria": [_KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)],
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        try:
            with torch.inference_mode():
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


__all__ = [
    "QwenXVarsCausalLM",
    "QwenXVarsModel",
]
