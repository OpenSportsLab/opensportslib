"""VQA model backends for OpenSportsLib."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from opensportslib.core.config.accessors import get_model_load
from opensportslib.core.utils.hf_runtime import HFCausalDecoderRuntime
from opensportslib.models.utils.vqa_prompting import build_prior_text, build_xvars_prompt
from opensportslib.models.utils.vqa_xvars_features import NumericProjector

logger = logging.getLogger(__name__)


class VQABaselineModel(nn.Module):
    """Deterministic baseline that builds short answers from question + optional priors."""

    def __init__(self, config):
        super().__init__()
        self.config = config

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
        offence = (((labels.get("offence") or {}).get("label")) or "").strip()
        action = (((labels.get("action") or {}).get("label")) or "").strip()
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

        if offence and action:
            base = f"{offence}. The action appears to be {action.lower()}."
        elif offence:
            base = offence
        elif action:
            base = f"The action appears to be {action.lower()}."
        else:
            refs = sample.get("references") or []
            base = str(refs[0]).strip() if refs else "Insufficient evidence to provide a definitive decision."

        if style == "detailed" and question:
            if priors:
                return f"Question: {question} Priors: {priors} Answer: {base}"
            return f"Question: {question} Answer: {base}"
        return base


class MultimodalHFVQAModel(nn.Module):
    """X-VARS multimodal VQA model backed by indexed CLIP feature tensors."""

    def __init__(self, config, model_id: str, projector_params: dict[str, Any] | None = None):
        super().__init__()
        self.config = config
        self.baseline = VQABaselineModel(config)

        projector_params = projector_params or {}
        in_dim = int(projector_params.get("input_dim", 1024))
        out_dim = int(projector_params.get("output_dim", 1024))
        self.projector = NumericProjector(in_dim=in_dim, out_dim=out_dim)

        exec_cfg_ns = getattr(getattr(config, "TRAIN", None), "execution", None)
        if exec_cfg_ns is None:
            exec_cfg = {}
        elif isinstance(exec_cfg_ns, dict):
            exec_cfg = exec_cfg_ns
        elif hasattr(exec_cfg_ns, "__dict__"):
            exec_cfg = vars(exec_cfg_ns)
        else:
            exec_cfg = {}

        hf_cfg = exec_cfg.get("hf", {})
        if hasattr(hf_cfg, "__dict__"):
            hf_cfg = vars(hf_cfg)
        if not isinstance(hf_cfg, dict):
            hf_cfg = {}

        local_files_only = bool(hf_cfg.get("local_files_only", False))
        prefer_cuda = bool(hf_cfg.get("prefer_cuda", True))
        cuda_device_index = hf_cfg.get("cuda_device_index")
        if cuda_device_index is not None:
            try:
                cuda_device_index = int(cuda_device_index)
            except Exception:
                cuda_device_index = None
        adapter_path = get_model_load(config).get("checkpoint_path")
        self.decoder = HFCausalDecoderRuntime(
            model_id=model_id,
            local_files_only=local_files_only,
            prefer_cuda=prefer_cuda,
            adapter_path=adapter_path,
            cuda_device_index=cuda_device_index,
        )

    def _build_prompt(self, sample: dict[str, Any], prompt_cfg: dict[str, Any] | None = None) -> str:
        prompt_cfg = prompt_cfg or {}
        system_prompt = str(
            prompt_cfg.get(
                "system_prompt",
                "You are a football refereeing assistant. Answer concisely and justify briefly.",
            )
        ).strip()
        question = str(sample.get("question", "")).strip()

        include_priors = bool(prompt_cfg.get("include_priors", True))
        prior_text = ""
        if include_priors:
            built_prior = build_prior_text(
                sample.get("labels", {}) or {},
                sample.get("metadata", {}) or {},
                include_fields=prompt_cfg.get("prior_fields"),
            )
            prior_text = str(sample.get("prior_prediction_text", "")).strip() or built_prior
        token_len = int(prompt_cfg.get("video_token_len", 300))
        return build_xvars_prompt(
            system_prompt=system_prompt,
            question=question,
            prior_text=prior_text,
            video_token_len=token_len,
        )

    def generate_answer(
        self,
        sample: dict[str, Any],
        prompt_cfg: dict[str, Any] | None = None,
        generation_cfg: dict[str, Any] | None = None,
    ) -> str:
        generation_cfg = generation_cfg or {}
        fallback_policy = str(generation_cfg.get("fallback_policy", "none")).lower()
        if not self.decoder.is_ready:
            if fallback_policy == "baseline_on_failure":
                logger.warning("Falling back to baseline VQA answer generation | reason=%s", self.decoder.error)
                return self.baseline.generate_answer(sample, prompt_cfg=prompt_cfg, generation_cfg=generation_cfg)
            raise RuntimeError(self.decoder.error or "HF decoder not ready and fallback_policy=none")

        prompt = self._build_prompt(sample, prompt_cfg=prompt_cfg)
        clip_features = sample.get("video_spatio_temporal_features")
        if clip_features is None:
            raise ValueError("Missing 'video_spatio_temporal_features' on sample. Ensure dataset uses xvars_clip backend.")
        if not isinstance(clip_features, torch.Tensor):
            clip_features = torch.as_tensor(clip_features, dtype=torch.float32)
        if clip_features.ndim == 1:
            clip_features = clip_features.unsqueeze(0)
        if clip_features.ndim != 2:
            raise ValueError(f"Expected 2D CLIP feature tensor [tokens, dim], got shape {tuple(clip_features.shape)}")
        video_vec = clip_features.mean(dim=0).to(torch.float32)
        token_len = int((prompt_cfg or {}).get("video_token_len", 300))
        hidden_size = int(getattr(self.decoder, "hidden_size", 0) or 0)
        projected_features = None
        if token_len > 0 and hidden_size > 0:
            projected_features = self.projector.to_patch_embeddings(
                video_vec if isinstance(video_vec, torch.Tensor) else torch.tensor(video_vec, dtype=torch.float32),
                patch_count=token_len,
                embed_dim=hidden_size,
            )
        try:
            out = self.decoder.generate(prompt, generation_cfg=generation_cfg, video_features=projected_features)
            if out:
                return out
            if fallback_policy == "baseline_on_failure":
                logger.warning("HF generation returned empty text, using baseline fallback")
                return self.baseline.generate_answer(sample, prompt_cfg=prompt_cfg, generation_cfg=generation_cfg)
        except Exception as exc:
            if fallback_policy == "baseline_on_failure":
                logger.warning("HF generation failed, using baseline fallback | reason=%s", str(exc))
                return self.baseline.generate_answer(sample, prompt_cfg=prompt_cfg, generation_cfg=generation_cfg)
            raise
        raise RuntimeError(
            "HF decoder returned an empty response and fallback_policy=none "
            "(set fallback_policy=baseline_on_failure to allow deterministic fallback)"
        )


__all__ = ["MultimodalHFVQAModel", "VQABaselineModel"]
