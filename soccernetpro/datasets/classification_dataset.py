import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from soccernetpro.core.utils.video_processing import *
from soccernetpro.core.utils.load_annotations import load_annotations

class ClassificationDataset(Dataset):
    def __init__(self, config, annotations_path, processor, split="train"):
        self.config = config
        self.split = split
        self.exclude_labels = ["Unknown", "Dont know"]
        self.samples, self.label_map = load_annotations(annotations_path, 
                                        exclude_labels=self.exclude_labels, 
                                        multiview=config.DATA.view_type == "multi")

        self.label_map = {v: k for k, v in self.label_map.items()}
        self.config.DATA.classes = list(self.label_map.values())
        self.config.DATA.num_classes = len(self.label_map)
        print(self.config.DATA.num_classes, "classes:", self.config.DATA.classes)
        print("Label Map : ", self.label_map)
        #print(self.samples)
        #self.HOME_DIR = os.path.dirname(annotations_path)
        self.processor = processor
        #print(processor.do_normalize)  # should be True
        #print(processor.image_mean, processor.image_std)  # only used internally
        self.view_type = config.DATA.view_type
        self.num_frames = config.DATA.num_frames
        self.input_fps = config.DATA.input_fps
        self.transform = build_transform(config, mode=self.split)

    def get_sample_weights(self):
        labels = [item["label"] for item in self.samples]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.float)
        return sample_weights

    def get_class_weights(self, num_classes=None, normalize=True, sqrt=False):
    
        labels = torch.tensor([item["label"] for item in self.samples])

        if num_classes is None:
            num_classes = int(labels.max().item() + 1)

        # Count samples per class
        counts = torch.bincount(labels, minlength=num_classes).float()

        # Avoid division by zero
        counts[counts == 0] = 1.0

        if sqrt:
            weights = 1.0 / torch.sqrt(counts)
        else:
            weights = 1.0 / counts

        if normalize:
            weights = weights / weights.sum() * num_classes

        return weights

        
    def _select_views(self, video_paths):
        if self.view_type == "single":
            return [video_paths[0]]

        if self.split.lower() == "train" and self.view_type == "multi":
            return random.sample(video_paths, min(2, len(video_paths)))

        return video_paths

    def _load_and_sample_clip(self, path):
        v = read_video(os.path.join(self.config.DATA.data_dir, path))

        v = process_frames(
            v,
            self.num_frames,
            self.input_fps,
            self.config.DATA.target_fps,
            start_frame=self.config.DATA.start_frame,
            end_frame=self.config.DATA.end_frame
        )

        if isinstance(v, list):
            v = np.stack(v)  # (T, H, W, C)

        if self.transform is not None:
            v = self.transform(v)

        return v  # (T, H, W, C)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = torch.tensor(item["label"], dtype=torch.long)
        video_paths = item["video_paths"]
        # take parent folder name
        first_path = video_paths[0]["path"] if isinstance(video_paths[0], dict) else video_paths[0]
        sample_id = first_path.split("/")[-2]   # action_0
        split = first_path.split("/")[0]        # test

        sample_id = f"{split}_{sample_id}"
        # --- Choose which clips to load ---
        if not video_paths:
            raise ValueError(f"No video paths found for item {idx}")
    
        selected_paths = self._select_views(video_paths)
        
        # --- Load and process frames for selected clips ---
        if self.config.MODEL.type == "huggingface":
            path = selected_paths[0]
            v = self._load_and_sample_clip(path)
            # convert clip -> list of frames
            v = list(v)   # each element is (H, W, C)
        
            #print(type(v), v)
            v = self.processor(v, return_tensors="pt")#, do_rescale=False) 
            pixel_values = v["pixel_values"].float()
            pixel_values = pixel_values.squeeze(0)
            return {"pixel_values": pixel_values, "labels": label, "id": sample_id}
        
        else:
            view_tensors=[]
            for path in selected_paths:
                v = self._load_and_sample_clip(path)
                
                # (T, H, W, C) → (T, C, H, W)
                v = torch.from_numpy(v).permute(0, 3, 1, 2)

                # (T, C, H, W) → (C, T, H, W)
                v_t = get_transforms_model(self.config.MODEL.pretrained_model)(v)

                view_tensors.append(v_t)

            # Stack → (V, C, T, H, W)
            videos = torch.stack(view_tensors, dim=0)
            #print("VIDEOS:", videos.shape)
            return {"pixel_values": videos, "labels": label, "id": sample_id}




        # # --- Load and process frames for selected clips ---
        # for path in selected_paths:
        #     v = read_video(os.path.join(self.HOME_DIR, self.config.data_path, path))  # -> list or tensor of frames
        #     v = process_frames(v, self.num_frames, self.input_fps)
        #     v = torch.stack([self.transform(f) for f in v]).permute(1, 0, 2, 3)
        #     clip_tensors.append(v)
    
        # # --- Stack cameras along a new dimension ---
        # # Always return tensor of shape [num_cams, C, T, H, W]
        # video_tensor = torch.stack(clip_tensors)
    
        # label = torch.tensor(item["label"], dtype=torch.long)
        # return video_tensor, label


    