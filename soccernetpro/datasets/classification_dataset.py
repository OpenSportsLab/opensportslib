import os
import torch
import random
from torch.utils.data import Dataset
from soccernetpro.core.utils.video_processing import *
from soccernetpro.core.utils.load_annotations import load_annotations

class ClassificationDataset(Dataset):
    def __init__(self, config, annotations_path, processor, split="train"):
        self.config = config
        self.split = split
        self.samples = load_annotations(annotations_path)
        # print(self.samples)
        #self.HOME_DIR = os.path.dirname(annotations_path)
        self.processor = processor
        #print(processor.do_normalize)  # should be True
        #print(processor.image_mean, processor.image_std)  # only used internally
        self.view_type = config.DATA.view_type
        self.num_frames = config.DATA.num_frames
        self.input_fps = config.DATA.input_fps
        self.transform = build_transform(config, mode=self.split)
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        clip_tensors = []
        video_paths = item["video_paths"]
        # --- Choose which clips to load ---
        if len(video_paths) == 0:
            raise ValueError(f"No video paths found for item {idx}")
    
        elif self.view_type=="single":
            # Single-camera: use as is
            selected_paths = [video_paths[0]]
    
        else:
            if self.view_type=="multiple":
                # Multi-camera: always include main (first), randomly pick 1 extra
                selected_paths = [video_paths[0]]
                remaining = video_paths[1:]
                if len(remaining) > 0:
                    extra = random.choice(remaining)
                    selected_paths.append(extra)
        
        
        # --- Load and process frames for selected clips ---
        for path in selected_paths:
            v = read_video(os.path.join(self.config.DATA.data_dir, path))  # -> list or tensor of frames
            v = process_frames(v, self.num_frames, self.input_fps, self.config.DATA.target_fps, start_frame=self.config.DATA.start_frame, end_frame=self.config.DATA.end_frame)
            
            #print(v.shape)

            # ensure numpy clip
            if isinstance(v, list):
                v = np.stack(v)  # (T, H, W, C), uint8

            # clip-level augmentation (optional)
            if self.transform is not None:
                v = self.transform(v) # (T, H, W, C), uint8

            # convert clip -> list of frames
            v = list(v)   # each element is (H, W, C)
            
            if self.config.MODEL.type == "huggingface":
                #print(type(v), v)
                v = self.processor(v, return_tensors="pt")#, do_rescale=False) 
                pixel_values = v["pixel_values"].float()
                pixel_values = pixel_values.squeeze(0)
                label = torch.tensor(item["label"], dtype=torch.long)
                output = {"pixel_values": pixel_values, "labels": label}

        return output

    def get_sample_weights(self):
        labels = [item["label"] for item in self.samples]
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.float)
        return sample_weights

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


    