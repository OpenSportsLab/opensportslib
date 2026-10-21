"""Frame-indexed T-DEED windows over OSL video manifests.

Training samples uniformly from the released sliding-window pool. Padding is
black, displacement signs are i-event, and colliding targets follow annotation
order, as in the pinned implementation. No labels are reconstructed from a
rounded millisecond clock when native frame indices are available.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SpottingVideo:
    identity: str
    path: Path
    frames: int
    fps: float
    events: list
    source_start: int = 0
    sha256: str | None = None


def load_videos(manifest, source_root=None):
    manifest = Path(manifest)
    doc = json.loads(manifest.read_text())
    root = Path(source_root) if source_root else manifest.parent
    videos = []
    for row in doc["data"]:
        meta = row.get("metadata", {})
        inputs = [x for x in row["inputs"] if x.get("type") == "video"]
        if len(inputs) != 1:
            raise ValueError("T-DEED requires exactly one video input per sample")
        inp = inputs[0]
        path = root / inp["path"]
        fps = meta.get("fps_raw") or inp.get("fps")
        n = meta.get("num_frames")
        if fps is None or n is None:
            import cv2

            probe = cv2.VideoCapture(str(path))
            try:
                fps = fps if fps is not None else probe.get(cv2.CAP_PROP_FPS)
                n = n if n is not None else int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                probe.release()
        fps, n = float(fps), int(n)
        if fps <= 0 or n <= 0:
            raise ValueError("Video FPS and frame count must be positive")
        intervals = meta.get("intervals", [])
        if len(intervals) > 1:
            raise ValueError("Split multiple intervals into separate OSL samples")
        start = int(meta.get("source_start_frame", 0)) if intervals else 0
        if intervals and "source_start_frame" not in meta:
            start = round(intervals[0]["start_time_ms"] * fps / 1000)
        events = []
        for event in row.get("events", []):
            frame = event.get("reference_frame", event.get("frame"))
            if frame is None:
                frame = round(event["position_ms"] * fps / 1000) - start
            if frame < 0 or (
                frame >= n
                and not meta.get("allow_events_beyond_reference_frames", False)
            ):
                raise ValueError(f'{row["id"]}: event outside video')
            events.append({**event, "frame": int(frame)})
        videos.append(
            SpottingVideo(
                meta.get("video") or row.get("id") or inp["path"],
                path,
                n,
                fps,
                events,
                start,
                meta.get("sha256"),
            )
        )
    if not videos or len({v.identity for v in videos}) != len(videos):
        raise ValueError("Empty manifest or duplicate video IDs")
    return videos


def displacement_targets(
    events, classes, start, clip_len, stride=1, radius=0, dilation=0
):
    labels = np.zeros(clip_len, np.int64)
    offsets = np.zeros(clip_len, np.float32)
    for event in events:
        index = (event["frame"] - start) // stride
        width = radius if radius > 0 else dilation
        for i in range(max(0, index - width), min(clip_len, index + width + 1)):
            labels[i] = classes[event["label"]]
            offsets[i] = i - index if radius > 0 else 0
    return labels, offsets


class RGBReader:
    """Bounded per-worker cache for short clips; seeks long source broadcasts."""

    def __init__(self, resize):
        self.height, self.width = map(int, resize)

    def __getstate__(self):
        return {"height": self.height, "width": self.width}

    @lru_cache(maxsize=4)
    def _short(self, path, expected):
        import cv2

        capture = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                frames.append(
                    cv2.cvtColor(
                        cv2.resize(image, (self.width, self.height)), cv2.COLOR_BGR2RGB
                    )
                )
        finally:
            capture.release()
        if len(frames) != expected:
            raise ValueError(
                f"{path}: decoded {len(frames)} frames; expected {expected}"
            )
        return np.stack(frames)

    def read(self, video, start, length, stride):
        import cv2

        indices = start + np.arange(length) * stride
        valid = (indices >= 0) & (indices < video.frames)
        frames = np.zeros((length, self.height, self.width, 3), np.uint8)
        if not valid.any():
            raise ValueError("Window contains no source frames")
        if video.frames <= 1000 and not video.source_start:
            frames[valid] = self._short(str(video.path), video.frames)[indices[valid]]
        else:
            capture = cv2.VideoCapture(str(video.path))
            first, last = int(indices[valid][0] + video.source_start), int(
                indices[valid][-1] + video.source_start
            )
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, first)
                positions = {
                    int(i + video.source_start): j
                    for j, i in enumerate(indices)
                    if valid[j]
                }
                for i in range(first, last + 1):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f"{video.path}: missing source frame {i}")
                    if i in positions:
                        frames[positions[i]] = cv2.cvtColor(
                            cv2.resize(image, (self.width, self.height)),
                            cv2.COLOR_BGR2RGB,
                        )
            finally:
                capture.release()
        return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()


class TDEEDDataset(Dataset):
    def __init__(
        self,
        videos,
        classes,
        *,
        clip_len=100,
        stride=1,
        radius=1,
        dilation=0,
        resize=(224, 398),
        epoch_num_frames=500000,
        training=True,
        mixup=True,
        overlap=0.9,
        foreground_only=False,
        pad=5,
        inference_overlap=None,
    ):
        self.videos, self.classes = videos, {c: i + 1 for i, c in enumerate(classes)}
        self.clip_len, self.stride, self.radius = clip_len, stride, radius
        self.dilation, self.training, self.mixup = dilation, training, mixup
        self.inference = inference_overlap is not None
        self.reader = RGBReader(resize)
        self.windows = []
        if min(clip_len, stride, epoch_num_frames) < 1 or not 0 <= overlap <= 1:
            raise ValueError("Invalid T-DEED window settings")
        for vi, video in enumerate(videos):
            if inference_overlap is None:
                # Retain the reference's integer conversion (including 0.9*100 rounding).
                step = max(1, int((1 - overlap) * clip_len)) if overlap != 1 else 1
                stop = max(0, video.frames - 1 + (2 * pad - clip_len) * stride)
            else:
                if not 0 <= inference_overlap < clip_len:
                    raise ValueError(
                        "Inference overlap must be smaller than clip length"
                    )
                step = (clip_len - inference_overlap) * stride
                stop = max(0, video.frames - inference_overlap * stride)
            for start in range(-pad * stride, stop, step):
                if start >= video.frames or start + (clip_len - 1) * stride < 0:
                    continue
                if foreground_only:
                    labels, _ = displacement_targets(
                        video.events,
                        self.classes,
                        start,
                        clip_len,
                        stride,
                        radius,
                        dilation,
                    )
                    if not labels.any():
                        continue
                self.windows.append((vi, start))
        if not self.windows:
            raise ValueError("No valid T-DEED windows")
        self.length = (
            epoch_num_frames // clip_len
            if inference_overlap is None
            else len(self.windows)
        )

    def __len__(self):
        return self.length

    def _one(self, index):
        vi, start = self.windows[index]
        video = self.videos[vi]
        labels, displ = displacement_targets(
            video.events,
            self.classes,
            start,
            self.clip_len,
            self.stride,
            self.radius,
            self.dilation,
        )
        sample = {
            "frame": self.reader.read(video, start, self.clip_len, self.stride),
            "label": torch.from_numpy(labels),
            "video_index": vi,
            "start": start // self.stride,
        }
        if self.radius > 0:
            sample["labelD"] = torch.from_numpy(displ)
        return sample

    def __getitem__(self, index):
        sampling = not self.inference
        first = self._one(random.randrange(len(self.windows)) if sampling else index)
        if self.training and self.mixup:
            second = self._one(random.randrange(len(self.windows)))
            for key in ["frame", "label", "labelD"]:
                if key in second:
                    first[key + "2"] = second[key]
        return first


class JointTDEEDDataset(Dataset):
    """Released 50/50 BAS/AS sampling, with twice the single-dataset epoch length."""

    def __init__(self, primary, auxiliary):
        self.primary, self.auxiliary = primary, auxiliary

    def __len__(self):
        return len(self.primary) + len(self.auxiliary)

    def __getitem__(self, index):
        head = 1 if random.random() < 0.5 else 2
        sample = (self.primary if head == 1 else self.auxiliary)[index]
        return {**sample, "dataset": head}
