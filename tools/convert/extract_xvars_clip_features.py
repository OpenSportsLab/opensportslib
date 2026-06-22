#!/usr/bin/env python3
"""Extract X-VARS feature tensors for OSL-XFoul videos."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

from opensportslib.core.config.accessors import (
    get_xvars_feature_token_len_for_mode,
    normalize_xvars_feature_mode,
)

STRICT_START_FRAME = 63
STRICT_END_FRAME = 87
STRICT_TARGET_FPS = 17
STRICT_SOURCE_FPS = 25


def get_seq_frames(total_num_frames: int, desired_num_frames: int) -> list[int]:
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq


def load_video_frames(video_path: Path, num_frm: int = 100) -> list[Image.Image]:
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frame_num = len(vr)
        total_num_frm = min(total_frame_num, num_frm)
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
        img_array = vr.get_batch(frame_idx).asnumpy()
        target_h, target_w = 224, 224
        if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
            arr = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            arr = torch.nn.functional.interpolate(arr, size=(target_h, target_w))
            img_array = arr.permute(0, 2, 3, 1).to(torch.uint8).numpy()
        return [Image.fromarray(img_array[j]) for j in range(total_num_frm)]
    except Exception:
        from torchvision.io import read_video

        frames, _, _ = read_video(str(video_path), pts_unit="sec")
        if frames.numel() == 0:
            return []
        total_frame_num = int(frames.shape[0])
        total_num_frm = min(total_frame_num, num_frm)
        frame_idx = get_seq_frames(total_frame_num, total_num_frm)
        frames = frames[torch.tensor(frame_idx, dtype=torch.long)]
        arr = frames.permute(0, 3, 1, 2).float()
        arr = torch.nn.functional.interpolate(arr, size=(224, 224))
        img_array = arr.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        return [Image.fromarray(img_array[j]) for j in range(total_num_frm)]


def crop_strict_xvars_window(
    frames: list[Image.Image],
    *,
    start_frame: int = STRICT_START_FRAME,
    end_frame: int = STRICT_END_FRAME,
    target_fps: float = STRICT_TARGET_FPS,
    source_fps: float = STRICT_SOURCE_FPS,
) -> list[Image.Image]:
    window = frames[start_frame:end_frame]
    if not window:
        return frames
    frame_span = max(end_frame - start_frame, 1)
    factor = frame_span / (((frame_span / source_fps) * target_fps))
    final_frames: list[Image.Image] = []
    for index, frame in enumerate(window):
        if index % factor < 1:
            final_frames.append(frame)
    return final_frames or window


def get_spatio_temporal_features(features: torch.Tensor, num_temporal_tokens: int) -> torch.Tensor:
    temporal_tokens = torch.mean(features, dim=1)
    t, _s, c = features.shape
    if t < num_temporal_tokens:
        padding = torch.zeros(num_temporal_tokens - t, c, device=features.device, dtype=features.dtype)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)
    else:
        temporal_tokens = temporal_tokens[:num_temporal_tokens]
    spatial_tokens = torch.mean(features, dim=0)
    return torch.cat([temporal_tokens, spatial_tokens], dim=0).half()


class MVNetwork(nn.Module):
    """Clean-room implementation of the original X-VARS visual encoder."""

    def __init__(self, vision_tower_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, low_cpu_mem_usage=True)
        feat_dim = 1024
        self.inter = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )
        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4),
        )
        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8),
        )

    def forward(self, video: torch.Tensor):
        out = self.vision_tower(video, output_hidden_states=True)
        batch_features = out.hidden_states[-2][:, 1:]
        video_features = batch_features.detach().cpu()
        pooled = torch.mean(out.pooler_output, dim=0).unsqueeze(0)
        pooled = self.inter(pooled)
        out_off = self.fc_offence(pooled)
        out_act = self.fc_action(pooled)
        return out_off.squeeze(), out_act.squeeze(), video_features


def normalize_strict_xvars_state_dict(raw_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        new_key = str(key)
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("vision_model."):
            new_key = "vision_tower.vision_model." + new_key[len("vision_model.") :]
        if new_key.startswith("text_model.") or new_key in {"visual_projection.weight", "text_projection.weight", "logit_scale"}:
            continue
        normalized[new_key] = value
    return normalized


class ClipCompatExtractor:
    def __init__(self, vision_model: str, *, device: torch.device):
        self.device = device
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model)
        self.vision_tower = CLIPVisionModel.from_pretrained(
            vision_model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        self.vision_tower.eval()

    def extract(self, video_path: Path, *, max_frames: int) -> np.ndarray:
        frames = load_video_frames(video_path, num_frm=max_frames)
        if not frames:
            return np.zeros((356, 1024), dtype=np.float16)
        image_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        with torch.no_grad():
            image_forward_outs = self.vision_tower(image_tensor, output_hidden_states=True)
            frame_features = image_forward_outs.hidden_states[-2][:, 1:]
        st = get_spatio_temporal_features(frame_features, num_temporal_tokens=100)
        return st.detach().cpu().numpy().astype(np.float16)


class StrictXVarsExtractor:
    def __init__(
        self,
        *,
        weights_path: str,
        vision_model: str,
        device: torch.device,
        start_frame: int = STRICT_START_FRAME,
        end_frame: int = STRICT_END_FRAME,
        target_fps: float = STRICT_TARGET_FPS,
        source_fps: float = STRICT_SOURCE_FPS,
    ):
        if not weights_path:
            raise ValueError("strict_xvars mode requires --weights-path pointing to 14_model.pth.tar")
        self.device = device
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.target_fps = float(target_fps)
        self.source_fps = float(source_fps)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model)
        self.model = MVNetwork(vision_model).to(device)
        weights = Path(weights_path).expanduser().resolve()
        state = torch.load(str(weights), map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported strict_xvars checkpoint format at {weights}")
        normalized_state_dict = normalize_strict_xvars_state_dict(state_dict)
        missing, unexpected = self.model.load_state_dict(normalized_state_dict, strict=False)
        required_prefixes = ("vision_tower.", "inter.", "fc_offence.", "fc_action.")
        blocking_missing = [key for key in missing if key.startswith(required_prefixes)]
        if blocking_missing:
            raise RuntimeError(
                "strict_xvars checkpoint is incompatible after key normalization. "
                f"Missing required weights such as: {blocking_missing[:8]}"
            )
        if unexpected:
            print(f"[strict_xvars] ignored unexpected checkpoint keys: {unexpected[:8]}")
        self.model.eval()

    def extract(self, video_path: Path, *, max_frames: int) -> np.ndarray:
        del max_frames
        frames = load_video_frames(video_path, num_frm=1000)
        if not frames:
            return np.zeros((300, 1024), dtype=np.float16)
        frames = crop_strict_xvars_window(
            frames,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            target_fps=self.target_fps,
            source_fps=self.source_fps,
        )
        image_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        with torch.no_grad():
            _out_off, _out_act, video_features = self.model(image_tensor)
        st = get_spatio_temporal_features(video_features.to(self.device), num_temporal_tokens=44)
        return st.detach().cpu().numpy().astype(np.float16)


def iter_video_paths(dataset_root: Path, split: str, item: dict[str, object]) -> Iterable[Path]:
    for inp in item.get("inputs") or []:
        if str(inp.get("type", "")).lower() != "video":
            continue
        rel = str(inp.get("path", "")).strip()
        if not rel:
            continue
        path = Path(rel)
        if path.is_absolute():
            yield path
            continue
        candidates = [dataset_root / rel, dataset_root / split / rel]
        for candidate in candidates:
            if candidate.exists():
                yield candidate
                break
        else:
            yield candidates[0]


def resolve_split_annotation_path(dataset_root: Path, split: str) -> Path:
    candidates = [
        dataset_root / split / f"{split}.json",
        dataset_root / f"{split}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find annotation JSON for split '{split}'. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def extract_feature_for_video(
    video_path: Path,
    *,
    mode: str,
    clip_extractor: ClipCompatExtractor | None = None,
    strict_extractor: StrictXVarsExtractor | None = None,
    max_frames: int = 100,
) -> np.ndarray:
    normalized_mode = normalize_xvars_feature_mode(mode, default="clip_compat")
    if normalized_mode == "strict_xvars":
        if strict_extractor is None:
            raise ValueError("strict_xvars extraction requires a strict extractor instance")
        return strict_extractor.extract(video_path, max_frames=max_frames)
    if clip_extractor is None:
        raise ValueError("clip_compat extraction requires a clip extractor instance")
    return clip_extractor.extract(video_path, max_frames=max_frames)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract PRE_CLIP_feature_clip_{i}.pkl files for OSL-XFoul.")
    ap.add_argument("--dataset-root", required=True, help="Root path containing train/valid/test and split JSON files.")
    ap.add_argument("--dataset-output-root", required=True, help="Root path to write PRE_CLIP_feature_clip_{i}.pkl files.")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--mode", default="clip_compat", choices=["strict_xvars", "clip_compat"])
    ap.add_argument("--vision-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--weights-path", default="", help="Required for strict_xvars mode; path to 14_model.pth.tar.")
    ap.add_argument("--start-frame", type=int, default=STRICT_START_FRAME, help="Strict X-VARS crop start frame.")
    ap.add_argument("--end-frame", type=int, default=STRICT_END_FRAME, help="Strict X-VARS crop end frame.")
    ap.add_argument("--target-fps", type=float, default=STRICT_TARGET_FPS, help="Strict X-VARS target fps after window crop.")
    ap.add_argument("--source-fps", type=float, default=STRICT_SOURCE_FPS, help="Strict X-VARS source fps used for window resampling.")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.dataset_output_root).expanduser().resolve()
    mode = normalize_xvars_feature_mode(args.mode, default="clip_compat")
    expected_tokens = get_xvars_feature_token_len_for_mode(mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_extractor = None
    strict_extractor = None
    if mode == "strict_xvars":
        strict_extractor = StrictXVarsExtractor(
            weights_path=args.weights_path,
            vision_model=args.vision_model,
            device=device,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            target_fps=args.target_fps,
            source_fps=args.source_fps,
        )
    else:
        clip_extractor = ClipCompatExtractor(args.vision_model, device=device)

    processed = 0
    skipped = 0
    failed = 0

    for split in args.splits:
        ann_path = resolve_split_annotation_path(dataset_root, split)
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        for item in payload.get("data", []):
            if args.max_samples > 0 and processed >= args.max_samples:
                break
            sid = str(item.get("id", "")).strip()
            if not sid:
                continue
            action_dir = output_root / split / sid
            action_dir.mkdir(parents=True, exist_ok=True)
            for index, video_path in enumerate(sorted(iter_video_paths(dataset_root, split, item))[:3], start=1):
                print(f"Processing mode={mode} split={split} id={sid} video={video_path}")
                out_path = action_dir / f"PRE_CLIP_feature_clip_{index}.pkl"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                try:
                    feat = extract_feature_for_video(
                        video_path,
                        mode=mode,
                        clip_extractor=clip_extractor,
                        strict_extractor=strict_extractor,
                        max_frames=args.max_frames,
                    )
                    if tuple(feat.shape) != (expected_tokens, 1024):
                        raise ValueError(
                            f"Extractor mode {mode} produced shape {tuple(feat.shape)}; expected ({expected_tokens}, 1024)"
                        )
                    with out_path.open("wb") as f:
                        pickle.dump(feat, f)
                    processed += 1
                except Exception:
                    failed += 1
                    continue
        if args.max_samples > 0 and processed >= args.max_samples:
            break

    print(f"processed={processed} skipped={skipped} failed={failed} device={device.type} mode={mode}")


if __name__ == "__main__":
    main()
