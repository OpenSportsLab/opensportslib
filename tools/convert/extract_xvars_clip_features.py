#!/usr/bin/env python3
"""Extract X-VARS-style CLIP spatio-temporal features for OSL-XFoul videos."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel


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


def get_spatio_temporal_features_torch(features: torch.Tensor) -> torch.Tensor:
    # features: [t, s, c]
    t, _s, c = features.shape
    temporal_tokens = torch.mean(features, dim=1)
    padding_size = 100 - t
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device, dtype=features.dtype)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)
    spatial_tokens = torch.mean(features, dim=0)
    return torch.cat([temporal_tokens, spatial_tokens], dim=0).half()


def extract_feature_for_video(
    video_path: Path,
    vision_tower,
    image_processor,
    *,
    device: torch.device,
    max_frames: int = 100,
) -> np.ndarray:
    frames = load_video_frames(video_path, num_frm=max_frames)
    if not frames:
        return np.zeros((356, 1024), dtype=np.float16)
    image_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    image_tensor = image_tensor.to(device=device, dtype=torch.float16)
    with torch.no_grad():
        image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][:, 1:]
    st = get_spatio_temporal_features_torch(frame_features)
    return st.detach().cpu().numpy().astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract PRE_CLIP_feature_clip_{i}.pkl files for OSL-XFoul.")
    ap.add_argument("--dataset-root", required=True, help="Root path containing train/valid/test and split JSON files.")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--vision-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    vision_tower = CLIPVisionModel.from_pretrained(
        args.vision_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    vision_tower.eval()

    processed = 0
    skipped = 0
    failed = 0

    for split in args.splits:
        ann_path = dataset_root / f"{split}.json"
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        items = payload.get("data", [])
        for item in items:
            if args.max_samples > 0 and processed >= args.max_samples:
                break
            sid = str(item.get("id", "")).strip()
            if not sid:
                continue
            inputs = item.get("inputs") or []
            video_paths = []
            for inp in inputs:
                if str(inp.get("type", "")).lower() != "video":
                    continue
                rel = str(inp.get("path", "")).strip()
                if not rel:
                    continue
                p = Path(rel)
                video_paths.append(p if p.is_absolute() else (dataset_root / rel))
            if not video_paths:
                continue

            action_dir = dataset_root / split / sid
            action_dir.mkdir(parents=True, exist_ok=True)
            for i, vp in enumerate(sorted(video_paths)[:3], start=1):
                out_path = action_dir / f"PRE_CLIP_feature_clip_{i}.pkl"
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                try:
                    feat = extract_feature_for_video(
                        vp,
                        vision_tower,
                        image_processor,
                        device=device,
                        max_frames=args.max_frames,
                    )
                    with out_path.open("wb") as f:
                        pickle.dump(feat, f)
                    processed += 1
                except Exception:
                    failed += 1
                    continue

        if args.max_samples > 0 and processed >= args.max_samples:
            break

    print(f"processed={processed} skipped={skipped} failed={failed} device={device.type}")


if __name__ == "__main__":
    main()
