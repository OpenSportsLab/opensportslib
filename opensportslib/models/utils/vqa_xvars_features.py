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
