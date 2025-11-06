import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
import decord
from decord import cpu


class ActionDataset(Dataset):
    """
    A modular video dataset loader supporting annotation-based metadata,
    FPS-based resampling, uniform frame sampling, and padding.
    """

    def __init__(self, config, split="train"):
        self.split = split
        print(type(self.split))
        self.config = config
        self.data_root = config.get("data_root")
        self.annotation_file = config.get("annotations")[split]
        self.num_frames = config.get("num_frames", 16)
        self.input_fps = config.get("input_fps")
        self.target_fps = config.get("target_fps", self.input_fps)
        self.transform = self._default_transform()

        # --- Load annotations once ---
        self.samples = self._load_annotations()
        print(f"Loaded {len(self.samples)} {split} samples")

    
    def resample_video_idx(self, num_frames, original_fps, new_fps):
        # https://github.com/HumamAlwassel/TSP/blob/e0b3f6ef41ac40dd8b33445a769ae769c9f6a1d5/train/untrimmed_video_dataset.py#L135
        # Credit: resampling the video with original_fps to match the new_fps
        """Resample indices to match desired fps."""
        step = float(original_fps) / new_fps
        if step.is_integer():
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

    # transform
    def _default_transform(self):
        mean = self.config.get("normalize").get("mean")
        std = self.config.get("normalize").get("std")
        size = self.config.get("frame_size")
    
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(size),
            T.Normalize(mean, std),
        ])

    # load annotations
    def _load_annotations(self):
        """Load annotation JSON file."""
        with open(self.annotation_file, "r") as f:
            data = json.load(f)
    
        # SoccerNet multiview style
        if "Actions" in data:
            # collect valid action classes
            class_names = sorted(list({
                v["Action class"] for v in data["Actions"].values()
                if v["Action class"] not in ["", "Challenge"]
            }))
            class_to_idx = {name: i for i, name in enumerate(class_names)}
    
            samples = []
            for _, action in data["Actions"].items():
                action_class = action["Action class"]
                if action_class in ["", "Challenge"]:
                    continue  # skip invalid actions
    
                label = class_to_idx[action_class]
    
                # get only first clip if available
                first_clip_key = next(iter(action["Clips"]), None)
                if first_clip_key is None:
                    continue
    
                clip_info = action["Clips"][0]
                if isinstance(clip_info, dict) and "Url" in clip_info:
                    clip_path = clip_info["Url"]
                elif isinstance(clip_info, str):
                    clip_path = clip_info
                else:
                    continue
    
                # replace Dataset paths to match your local folder
                clip_path = (
                    clip_path.replace("Dataset/Train", "train")
                             .replace("Dataset/Test", "test")
                             .replace("Dataset/Valid", "valid")
                )
    
                samples.append({"videos": [clip_path + ".mp4"], "label": label})
    
            #print(f"Loaded {len(samples)} valid samples from {self.annotation_file}")
        return samples



    # Read frames
    def _read_video(self, video_path):
        """Read full video into a tensor using Decord."""
        try:
            vr = decord.VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            raise RuntimeError(f"Error reading video {video_path}: {e}")

        video_data = vr.get_batch(range(len(vr)))  # (T, H, W, C)
        video_data = torch.from_numpy(video_data.asnumpy()).permute(0, 3, 1, 2)  # (T, C, H, W)
        return video_data

    # frame processing
    def _process_frames(self, video_data):
        """
        Adjust number of frames:
        - If too short: resample based on FPS ratio
        - If too long: uniformly sample
        - If still short: pad with last frame
        """
        num_frames = video_data.size(0)
        real_duration = num_frames / self.input_fps

        # Too short → compute new FPS
        if num_frames < self.num_frames:
            new_fps = np.ceil(self.num_frames / real_duration)
            new_idxs = self.resample_video_idx(self.num_frames, self.input_fps, new_fps)
            new_idxs = np.clip(new_idxs, 0, num_frames - 1)
            video_data = video_data[new_idxs]

        # Too long → uniform sampling
        elif num_frames > self.num_frames:
            idxs = np.linspace(0, num_frames - 1, num=self.num_frames)
            video_data = video_data[idxs.astype(int)]

        # Still short → pad with last frame
        if video_data.size(0) < self.num_frames:
            pad = self.num_frames - video_data.size(0)
            last_frame = video_data[-1:].repeat(pad, 1, 1, 1)
            video_data = torch.cat([video_data, last_frame], dim=0)
        return video_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        view_tensors = []

        for video_rel in sample["videos"]:
            video_path = os.path.join(self.data_root, video_rel)
            if not os.path.exists(video_path):
                print(f"Missing: {video_path}")
                continue

            frames = self._read_video(video_path)
            frames = self._process_frames(frames)

            if self.transform:
                frames = torch.stack([self.transform(f) for f in frames])

            # Reorder to (C, N, H, W)
            frames = frames.permute(1, 0, 2, 3)
            view_tensors.append(frames)

        if len(view_tensors) == 0:
            raise RuntimeError(f"No valid videos for index {idx}")
        
        # Ensure shape consistency: always (V, C, N, H, W)
        if len(view_tensors) > 1:
            frames = torch.stack(view_tensors)  # (V, C, N, H, W)
        else:
            frames = view_tensors[0].unsqueeze(0)  # (1, C, N, H, W)

        return frames, label, str(idx)



if __name__ == "__main__":
    from utils.config_utils import load_config
    # Step 1: Load config file
    config_path = "configs/default.yaml"  # example path
    config = load_config(config_path)
    #print(config)
    
    # Check train dataset 
    dataset = ActionDataset(config["DATA"], split="train")
    print(f"Dataset length: {len(dataset)}")
    frames, label, idx = dataset[0]
    print(f"Frames shape: {frames.shape}")  # 
    print(f"Label: {label}")
    print(f"idx: {idx}")

    # Check valid dataset
    dataset = ActionDataset(config["DATA"], split="valid") 
    print(f"Dataset length: {len(dataset)}")
    frames, label, idx = dataset[0]
    print(f"Frames shape: {frames.shape}")  # 
    print(f"Label: {label}")
    print(f"idx: {idx}")

    # Check test dataset
    dataset = ActionDataset(config["DATA"], split="test") 
    print(f"Dataset length: {len(dataset)}")
    frames, label, idx = dataset[0]
    print(f"Frames shape: {frames.shape}")  # 
    print(f"Label: {label}")
    print(f"idx: {idx}")
