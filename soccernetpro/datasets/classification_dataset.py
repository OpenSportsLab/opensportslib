import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from soccernetpro.core.utils.video_processing import *
from soccernetpro.core.utils.load_annotations import load_annotations
from tqdm import tqdm
import json

from soccernetpro.datasets.utils.tracking import (
            build_edge_index,
            HorizontalFlip,
            VerticalFlip,
            TeamFlip,
            NUM_OBJECTS,
            FEATURE_DIM,
            PITCH_HALF_LENGTH,
            PITCH_HALF_WIDTH,
            MAX_DISPLACEMENT,
            MAX_BALL_HEIGHT,
        )

import pandas as pd

def build(config, annotations_path, processor=None, split="train"):
    modality = config.DATA.data_modality.lower()

    if modality == "tracking_parquet":
        return TrackingDataset(config, annotations_path, split)
    elif modality == "video":
        return VideoDataset(config, annotations_path, processor, split)
    else:
        raise ValueError(f"Unknown data_modality: {modality}")
    

class ClassificationDataset(Dataset):
    """Base class for classification datasets."""

    def __init__(self, config, annotations_path, processor, split="train"):
        self.config = config
        self.split = split
        self.data_dir = config.DATA.data_dir
        self.processor = processor
        self.samples, self.label_map = load_annotations(annotations_path, 
                                        exclude_labels=["Unknown", "Dont know"], 
                                        multiview=getattr(config.DATA, "view_type", None) == "multi",
                                        input_type=config.DATA.data_modality)
        #print(self.samples)

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

        counts = torch.bincount(labels, minlength=num_classes).float()
        counts[counts == 0] = 1.0

        if sqrt:
            weights = 1.0 / torch.sqrt(counts)
        else:
            weights = 1.0 / counts

        if normalize:
            weights = weights / weights.sum() * num_classes

        return weights


    def __len__(self):
        return len(self.samples)


    def num_classes(self):
        return len(self.label_map)



class VideoDataset(ClassificationDataset):
    """Video-based classification dataset."""

    def __init__(self, config, annotations_path, processor, split="train"):
        super().__init__(config, annotations_path, split)

        self.processor = processor
        self.view_type = config.DATA.view_type
        self.num_frames = config.DATA.num_frames
        self.input_fps = config.DATA.input_fps
        self.transform = build_transform(config, mode=self.split)

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

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = torch.tensor(item["label"], dtype=torch.long)
        video_paths = item["video_paths"]
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
            return {"pixel_values": pixel_values, "labels": label}
        
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
            return {"pixel_values": videos, "labels": label}


class TrackingDataset(ClassificationDataset):
    """Tracking-based classification dataset for player position data."""
    
    def __init__(self, config, annotations_path, split="train"):
        super().__init__(config, annotations_path, split)
        
        # tracking-specific config
        self.num_frames = config.DATA.num_frames
        self.normalize = config.DATA.normalize
        self.edge_type = config.MODEL.edge
        self.k = config.MODEL.k
        self.preload_data = config.DATA.preload_data
        
        self.transforms = self._build_transforms(config, split)

        # preload data if enabled
        self.processed_samples = None
        if self.preload_data:
            self._preload_all_data()


    def _build_transforms(self, config, split):
        if split != "train":
            return []
        
        transforms = []
        aug_config = config.DATA.augmentations
        
        if getattr(aug_config, "horizontal_flip", False):
            transforms.append(HorizontalFlip(probability=0.5))
        
        if getattr(aug_config, "vertical_flip", False):
            transforms.append(VerticalFlip(probability=0.5))
        
        if getattr(aug_config, "team_flip", False):
            transforms.append(TeamFlip(probability=0.5))
        
        return transforms


    def _preload_all_data(self):
        """
        preload all tracking clips into memory.
        parses frames and builds edge indices during loading.
        """
        print(f"Preloading {len(self.samples)} {self.split} samples into memory...")
        
        self.processed_samples = []
        
        for idx, item in enumerate(tqdm(self.samples, desc=f"Loading {self.split}")):
            clip_paths = item["video_paths"]
            if not clip_paths:
                continue
            
            clip_path = clip_paths[0]
            
            # load and parse
            df = self._load_tracking_clip(clip_path)
            
            num_frames = len(df)
            all_features = np.zeros((num_frames, NUM_OBJECTS, FEATURE_DIM), dtype=np.float32)
            all_positions = []
            
            for t, (_, row) in enumerate(df.iterrows()):
                features, positions = self._parse_frame(row)
                all_features[t] = features
                all_positions.append(positions)
            
            # compute velocity deltas
            all_features = self._compute_deltas(all_features)
            
            # build edge indices for all frames
            edge_indices = []
            for t in range(num_frames):
                edge_index = build_edge_index(
                    all_features[t],
                    all_positions[t],
                    self.edge_type,
                    self.k
                )
                edge_indices.append(edge_index)
            
            self.processed_samples.append({
                "features": all_features,
                "positions": all_positions,
                "edge_indices": edge_indices,
                "label": item["label"]
            })
        
        print(f"Loaded {len(self.processed_samples)} {self.split} samples")
    
    def _load_tracking_clip(self, path):
        """load a parquet clip and return raw dataframe."""
        full_path = os.path.join(self.data_dir, path)
        return pd.read_parquet(full_path)
    
    def _parse_frame(self, row):
        """
        parse a single frame row from parquet into node features and positions.
        
        Returns:
            features: np.array of shape (NUM_OBJECTS, FEATURE_DIM)
            positions: list of position group strings
        """
        features = np.full((NUM_OBJECTS, FEATURE_DIM), -200.0, dtype=np.float32)
        positions = [''] * NUM_OBJECTS
        
        obj_idx = 0
        
        # Ball (always index 0)
        ball_str = row.get('balls', 'null')
        if pd.notna(ball_str) and ball_str not in ['null', '']:
            try:
                ball_list = json.loads(ball_str)
                if ball_list:
                    ball = ball_list[0]
                    x, y = ball.get('x'), ball.get('y')
                    z = ball.get('z', 0)
                    if x is not None and y is not None:
                        features[obj_idx] = [float(x), float(y), 1, 0, 0, 0, 0, float(z)]
                        positions[obj_idx] = 'BALL'
            except (json.JSONDecodeError, TypeError):
                pass
        obj_idx += 1
        
        # home players (indices 1-11)
        home_str = row.get('homePlayers', '[]')
        if pd.notna(home_str) and home_str not in ['null', '']:
            try:
                home_players = json.loads(home_str)
                home_players = sorted(home_players, key=lambda p: int(p.get('jerseyNum', 0)))[:11]
                
                for player in home_players:
                    x, y = player.get('x'), player.get('y')
                    if x is not None and y is not None:
                        features[obj_idx] = [float(x), float(y), 0, 1, 0, 0, 0, -200.0]
                        positions[obj_idx] = player.get('positionGroup', '')
                    obj_idx += 1
                
                # fill remaining home slots
                while obj_idx < 12:
                    obj_idx += 1
            except (json.JSONDecodeError, TypeError):
                obj_idx = 12
        else:
            obj_idx = 12
        
        # away players (indices 12-22)
        away_str = row.get('awayPlayers', '[]')
        if pd.notna(away_str) and away_str not in ['null', '']:
            try:
                away_players = json.loads(away_str)
                away_players = sorted(away_players, key=lambda p: int(p.get('jerseyNum', 0)))[:11]
                
                for player in away_players:
                    x, y = player.get('x'), player.get('y')
                    if x is not None and y is not None:
                        features[obj_idx] = [float(x), float(y), 0, 0, 1, 0, 0, -200.0]
                        positions[obj_idx] = player.get('positionGroup', '')
                    obj_idx += 1
            except (json.JSONDecodeError, TypeError):
                pass
        
        return features, positions
    
    def _compute_deltas(self, all_features):
        """compute velocity (delta x, delta y) features across frames."""
        for t in range(1, all_features.shape[0]):
            for obj in range(NUM_OBJECTS):
                if all_features[t, obj, 0] != -200.0 and all_features[t-1, obj, 0] != -200.0:
                    all_features[t, obj, 5] = all_features[t, obj, 0] - all_features[t-1, obj, 0]
                    all_features[t, obj, 6] = all_features[t, obj, 1] - all_features[t-1, obj, 1]
        return all_features
    
    def _normalize_features(self, features):
        """normalize features to [-1, 1] range. uses -2.0 as sentinel for missing values."""
        features_norm = features.copy()
        valid_mask = features_norm[:, :, 0] != -200.0
        
        features_norm[valid_mask, 0] /= PITCH_HALF_LENGTH
        features_norm[valid_mask, 1] /= PITCH_HALF_WIDTH
        features_norm[valid_mask, 5] /= MAX_DISPLACEMENT
        features_norm[valid_mask, 6] /= MAX_DISPLACEMENT
        features_norm[valid_mask, 7] /= MAX_BALL_HEIGHT
        
        # sentinel value for missing data
        features_norm[~valid_mask, 0] = -2.0
        features_norm[~valid_mask, 1] = -2.0
        features_norm[~valid_mask, 5] = -2.0
        features_norm[~valid_mask, 6] = -2.0
        features_norm[~valid_mask, 7] = -2.0
        
        return features_norm

    def __getitem__(self, idx):
        if self.preload_data:
            return self._getitem_preloaded(idx)
        else:
            return self._getitem_on_the_fly(idx)
    
    def _getitem_preloaded(self, idx):
        """get item from preloaded data."""
        sample = self.processed_samples[idx]
        
        # copy features to avoid modifying cached data
        features = sample["features"].copy()
        
        # apply augmentations (training only)
        for transform in self.transforms:
            features = transform(features)
        
        # normalize
        if self.normalize:
            features = self._normalize_features(features)
        
        # convert to tensors
        node_features = torch.tensor(features, dtype=torch.float32)
        edge_indices = [
            torch.tensor(ei, dtype=torch.long) 
            for ei in sample["edge_indices"]
        ]
        label = torch.tensor(sample["label"], dtype=torch.long)
        
        return {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "labels": label
        }
    
    def _getitem_on_the_fly(self, idx):
        """load and process item on-the-fly."""
        item = self.samples[idx]
        label = torch.tensor(item["label"], dtype=torch.long)
        
        # get parquet path
        clip_paths = item["video_paths"]
        if not clip_paths:
            raise ValueError(f"No tracking paths found for item {idx}")
        
        clip_path = clip_paths[0]
        
        # load parquet clip
        df = self._load_tracking_clip(clip_path)
        
        # parse all frames
        num_frames = len(df)
        all_features = np.zeros((num_frames, NUM_OBJECTS, FEATURE_DIM), dtype=np.float32)
        all_positions = []
        
        for t, (_, row) in enumerate(df.iterrows()):
            features, positions = self._parse_frame(row)
            all_features[t] = features
            all_positions.append(positions)
        
        # compute velocity deltas
        all_features = self._compute_deltas(all_features)
        
        # apply augmentations (training only)
        for transform in self.transforms:
            all_features = transform(all_features)
        
        # normalize
        if self.normalize:
            all_features = self._normalize_features(all_features)
        
        # build edge indices for each frame
        edge_indices = []
        for t in range(num_frames):
            edge_index = build_edge_index(
                all_features[t],
                all_positions[t],
                self.edge_type,
                self.k
            )
            edge_indices.append(torch.tensor(edge_index, dtype=torch.long))
        
        # convert features to tensor
        node_features = torch.tensor(all_features, dtype=torch.float32)
        
        return {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "labels": label
        }