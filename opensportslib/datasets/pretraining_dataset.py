# opensportslib/datasets/pretraining_dataset.py

"""Dataset for self-supervised video pre-training.

Loads unlabeled video clips for SSL methods (MAE, DINO, SimCLR).
For MAE a single clip is returned; for contrastive and distillation
methods two augmented views of the same clip are produced.
"""

import os
import glob
import logging
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from opensportslib.core.utils.video_processing import (
    read_video,
    build_transform,
)


class PretrainingDataset(Dataset):
    """Base dataset for self-supervised pre-training on unlabeled video.

    Scans a directory tree for video files (.mp4, .avi, .mkv) or
    pre-extracted frame arrays (.npy) and returns clips suitable for
    the configured SSL method.

    Args:
        config: full configuration namespace.
        data_dir: root directory containing video files.
        split: "train" or "valid" (controls augmentation strength).
    """

    SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".webm")
    SUPPORTED_ARRAY_EXTENSIONS = (".npy",)

    def __init__(self, config, data_dir=None, split="train"):
        super().__init__()
        self.config = config
        self.split = split
        self.ssl_method = config.SSL.method.lower()

        data_dir = data_dir or config.DATA.data_dir
        self.data_dir = data_dir

        self.num_frames = config.DATA.num_frames
        self.frame_size = tuple(config.DATA.frame_size)
        self.input_fps = getattr(config.DATA, "input_fps", None)
        self.target_fps = getattr(config.DATA, "target_fps", None)

        # discover all video / npy files
        self.samples = self._discover_files(data_dir)
        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No video or .npy files found under {data_dir}"
            )
        logging.info(
            f"[PretrainingDataset] Found {len(self.samples)} clips "
            f"in {data_dir} (split={split})"
        )

        # build transform pipeline
        self.transform = build_transform(config, mode=split)

        # for contrastive / DINO methods, we need a second augmentation
        self.multi_view = self.ssl_method in ("dino", "simclr")

    def _discover_files(self, root):
        """Recursively find all supported media files."""
        all_exts = self.SUPPORTED_VIDEO_EXTENSIONS + self.SUPPORTED_ARRAY_EXTENSIONS
        files = []
        for ext in all_exts:
            files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        files.sort()
        return files

    def _load_clip(self, path):
        """Load and temporally sample a clip from a video or npy file.

        Returns:
            frames: numpy array of shape (T, H, W, C).
        """
        if path.endswith(".npy"):
            frames = np.load(path)  # expected (T, H, W, C) or (T, C, H, W)
            if frames.ndim == 4 and frames.shape[1] in (1, 3) and frames.shape[1] < frames.shape[2]:
                # (T, C, H, W) -> (T, H, W, C)
                frames = np.transpose(frames, (0, 2, 3, 1))
        else:
            frames = read_video(path)  # returns (T, H, W, C) numpy

        frames = frames.astype(np.float32)

        # temporal sub-sampling
        T = frames.shape[0]
        if T > self.num_frames:
            if self.split == "train":
                start = random.randint(0, T - self.num_frames)
            else:
                start = (T - self.num_frames) // 2
            frames = frames[start : start + self.num_frames]
        elif T < self.num_frames:
            pad = self.num_frames - T
            frames = np.concatenate(
                [frames, np.repeat(frames[-1:], pad, axis=0)], axis=0
            )

        return frames  # (T, H, W, C) numpy

    def _apply_transform(self, clip_np):
        """Apply spatial transforms to a clip and convert to tensor.

        Args:
            clip_np: numpy array (T, H, W, C).

        Returns:
            tensor of shape (C, T, H, W) ready for 3D convolutions.
        """
        # VideoTransform expects (T, H, W, C) numpy and returns same
        if self.transform is not None and self.split == "train":
            clip_np = self.transform(clip_np)

        # if transform returned numpy, convert; if returned as-is, convert
        if isinstance(clip_np, np.ndarray):
            frames = torch.from_numpy(clip_np).float()
        else:
            frames = clip_np.float()

        # ensure (T, H, W, C) shape
        if frames.ndim == 4 and frames.shape[-1] in (1, 3):
            # (T, H, W, C) -> (C, T, H, W)
            frames = frames.permute(3, 0, 1, 2)
        elif frames.ndim == 4 and frames.shape[1] in (1, 3):
            # already (T, C, H, W) -> (C, T, H, W)
            frames = frames.permute(1, 0, 2, 3)

        # normalize to [0, 1]
        if frames.max() > 1.0:
            frames = frames / 255.0

        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return a clip or pair of augmented clips.

        Returns:
            dict with keys:
                - "pixel_values": (C, T, H, W) for MAE, or
                  list of 2 tensors each (C, T, H, W) for DINO/SimCLR.
                - "id": sample identifier string.
        """
        path = self.samples[idx]
        clip = self._load_clip(path)

        sample_id = os.path.relpath(path, self.data_dir)

        if self.multi_view:
            view1 = self._apply_transform(clip)
            view2 = self._apply_transform(clip)
            return {
                "pixel_values": [view1, view2],
                "id": sample_id,
            }
        else:
            clip = self._apply_transform(clip)
            return {
                "pixel_values": clip,
                "id": sample_id,
            }


def build(config, data_dir=None, split="train"):
    """Factory function for the pretraining dataset.

    Args:
        config: full configuration namespace.
        data_dir: override for the data directory.
        split: "train" or "valid".

    Returns:
        PretrainingDataset instance.
    """
    return PretrainingDataset(config, data_dir=data_dir, split=split)
