"""X-VARS-inspired feature extraction utilities for VQA."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class XVarsVideoEncoder:
    """Approximate X-VARS temporal+spatial tokenization from raw video."""

    def __init__(self, sampling_cfg: dict[str, Any] | None = None):
        sampling_cfg = sampling_cfg or {}
        self.start_frame = int(sampling_cfg.get("start_frame", 63))
        self.end_frame = int(sampling_cfg.get("end_frame", 87))
        self.input_fps = float(sampling_cfg.get("input_fps", 25))
        self.target_fps = float(sampling_cfg.get("target_fps", 17))
        self.temporal_tokens = int(sampling_cfg.get("temporal_tokens", 44))

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.numel() == 0:
            return frames

        t = frames.shape[0]
        start = min(max(self.start_frame, 0), max(t - 1, 0))
        end = self.end_frame if self.end_frame > 0 else t
        end = min(max(end, start + 1), t)
        window = frames[start:end]
        if window.shape[0] == 0:
            window = frames
        if window.shape[0] == 0:
            return window

        factor = self.input_fps / max(self.target_fps, 1e-6)
        desired = max(int(round(window.shape[0] / max(factor, 1e-6))), 1)
        idx = torch.linspace(0, window.shape[0] - 1, steps=desired).long()
        return window[idx]

    def encode(self, video_path: str | None) -> torch.Tensor:
        if not video_path:
            return torch.zeros((self.temporal_tokens + 1) * 6, dtype=torch.float32)
        try:
            from torchvision.io import read_video

            frames, _, _ = read_video(video_path, pts_unit="sec")
            if frames.numel() == 0:
                return torch.zeros((self.temporal_tokens + 1) * 6, dtype=torch.float32)

            sampled = self._sample_frames(frames.float())
            if sampled.numel() == 0:
                return torch.zeros((self.temporal_tokens + 1) * 6, dtype=torch.float32)

            means = sampled.mean(dim=(1, 2)) / 255.0
            stds = sampled.std(dim=(1, 2)) / 255.0
            frame_desc = torch.cat([means, stds], dim=1)

            temporal = frame_desc
            if temporal.shape[0] < self.temporal_tokens:
                pad = torch.zeros(self.temporal_tokens - temporal.shape[0], temporal.shape[1])
                temporal = torch.cat([temporal, pad], dim=0)
            else:
                temporal = temporal[: self.temporal_tokens]

            spatial = frame_desc.mean(dim=0, keepdim=True)
            combined = torch.cat([temporal, spatial], dim=0).reshape(-1)
            return combined.to(torch.float32)
        except Exception:
            return torch.zeros((self.temporal_tokens + 1) * 6, dtype=torch.float32)


class NumericProjector(nn.Module):
    """Project visual descriptor vector into compact multimodal prompt tokens."""

    def __init__(self, in_dim: int = 270, out_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)

    def _align(self, video_features: torch.Tensor) -> torch.Tensor:
        x = video_features
        if x.numel() < self.in_dim:
            x = torch.cat([x, torch.zeros(self.in_dim - x.numel(), dtype=x.dtype)], dim=0)
        elif x.numel() > self.in_dim:
            x = x[: self.in_dim]
        return x

    def to_prompt_tokens(self, video_features: torch.Tensor) -> str:
        x = self._align(video_features)
        with torch.no_grad():
            projected = self.proj(x.unsqueeze(0)).squeeze(0).tolist()
        return " ".join(f"<v{i}:{val:.3f}>" for i, val in enumerate(projected))

    def to_patch_embeddings(
        self,
        video_features: torch.Tensor,
        *,
        patch_count: int,
        embed_dim: int,
    ) -> torch.Tensor:
        """Create patch-aligned embeddings for multimodal token injection."""
        if patch_count <= 0:
            raise ValueError("patch_count must be > 0")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")

        x = self._align(video_features).to(torch.float32)
        with torch.no_grad():
            projected = self.proj(x.unsqueeze(0)).squeeze(0)

        repeats = (embed_dim + projected.numel() - 1) // projected.numel()
        base = projected.repeat(repeats)[:embed_dim]
        # Deterministic small variation across patch positions.
        scales = torch.linspace(0.98, 1.02, steps=patch_count, dtype=base.dtype)
        out = torch.stack([base * s for s in scales], dim=0)
        return out


def validate_xvars_feature_tensor(
    features: torch.Tensor,
    *,
    expected_tokens: int | None = None,
    context: str = "X-VARS features",
) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError(f"{context} must be a 2D tensor [tokens, dim], got shape {tuple(features.shape)}")
    if expected_tokens is not None and int(features.shape[0]) != int(expected_tokens):
        raise ValueError(
            f"{context} token count mismatch: expected {int(expected_tokens)}, got {int(features.shape[0])}. "
            "Check that the configured X-VARS feature mode matches the extracted feature files."
        )
    return features
