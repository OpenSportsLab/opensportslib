# opensportslib/datasets/classification_dataset.py

"""classification dataset implmentations for video and tracking modalities.

provides three concrete dataset classes:

- VideoDataset (MVFoul, SN-GAR video)
- TrackingDataset (SN-GAR tracking / parquet)

both inherit from ClassificationDataset, which handles annotation loading,
label mapping, and class-weight computation.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from opensportslib.core.utils.load_annotations import load_annotations
from opensportslib.core.utils.video_processing import *


# -------------------------------------------------------------
# factory
# -------------------------------------------------------------

def build(config, annotations_path, processor=None, split="train"):
    """construct the appropriate dataset for the configured modality.

    Args:
        config: the loaded YAML configuration.
        annotations_path: path to the annotation JSON file.
        processor: HuggingFace image processor (video modality only).
        split: one of "train", "valid", "test".

    Returns:
        a Dataset instance (VideoDataset or TrackingDataset).

    Raises:
        ValueError: if the data_modality is not recognized.
    """
    modality = config.DATA.data_modality.lower()

    if modality == "tracking_parquet":
        return TrackingDataset(config, annotations_path, split)
    elif modality in ("video", "frames_npy"):
        return VideoDataset(config, annotations_path, processor, split)
    else:
        raise ValueError(f"Unknown data_modality: {modality}")
    

# -------------------------------------------------------------
# base class
# -------------------------------------------------------------

class ClassificationDataset(Dataset):
    """shared base for all classification datasets.
    
    loads annotations, builds a label map, and exposes helpers for
    computing sample-level and class-level weights (useful for
    balanced sampling and weighted loss respectively).

    Args:
        config: the loaded YAML configuration.
        annotations_path: path to the annotation JSON file.
        split: one of "train", "valid", "test".
    """

    def __init__(self, config, annotations_path, processor, split="train"):
        self.config = config
        self.split = split
        self.exclude_labels = ["Unknown", "Dont know"]
        self.data_dir = config.DATA.data_dir
        self.processor = None

        # view_type is optional; only MVFoul uses it as of now
        is_multiview = getattr(config.DATA, "view_type", None) == "multi"

        allow_missing_labels = split in ["test", "infer"]

        # these lines of code are used for data scaling experiments.
        # if you want to check how the model performance changes with different 
        # number of games in the training set, you can use this code.
        # to use this, you need to add the following to the config:
        # DATA:
        #   data_slicing:
        #     enabled: true
        #     training_matches: <number of games to include in the training set>
        # we will refer to this as "data slicing" in the rest of the code.
        max_games = None
        slicing_cfg = getattr(config.DATA, "data_slicing", None)
        if slicing_cfg and getattr(slicing_cfg, "enabled", False) and split == "train":
            max_games = getattr(slicing_cfg, "training_matches", None)

        self.samples, self.label_map = load_annotations(
            annotations_path, 
            exclude_labels=self.exclude_labels, 
            multiview=is_multiview,
            input_type=config.DATA.data_modality,
            allow_missing_labels=allow_missing_labels,
            max_games=max_games
        )

        # this is used for quick testing of the model.
        # we can only use a small subset of data (ideally 100 samples) to 
        # test the overall integration quickly.
        # to use this, you need to add the following to the config:
        # DATA:
        #   max_samples: <number of samples to include in the training set>
        max_samples = getattr(config.DATA, 'max_samples', None)
        if max_samples:
            self.samples = self.samples[:max_samples]

        # invert to id -> name and propagate into the config so
        # downstream components (metrics, logging) can look it up.
        self.label_map = {v: k for k, v in self.label_map.items()}
        self.config.DATA.classes = list(self.label_map.values())
        self.config.DATA.num_classes = len(self.label_map)

        print(self.config.DATA.num_classes, "classes:", self.config.DATA.classes)
        print("Label Map : ", self.label_map)

        self.has_labels = len(self.samples) > 0 and "label" in self.samples[0]

    # -- Sampling / loss weights ------------------------------------------
    
    def get_sample_weights(self):
        """per-sample inverse-frequency weights for WeightedRandomSampler.

        Returns:
            torch.Tensor of length len(self) with one weight per sample.
        """
        labels = [item["label"] for item in self.samples]

        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = torch.tensor(
            [class_weights[label] for label in labels], 
            dtype=torch.float
        )
        
        return sample_weights


    def get_class_weights(self, num_classes=None, normalize=True, sqrt=False):
        """per-class inverse-frequency weights for WeightedRandomSampler.

        Args:
            num_classes: if None, inferred from the label tensor.
            normalize: if True, weights are scaled so they sum to num_classes.
            sqrt: if True, use inverse square-root frequency instead of raw counts.

        Returns:
            torch.Tensor of shape (num_classes,).
        """
        labels = torch.tensor([item["label"] for item in self.samples])

        if num_classes is None:
            num_classes = int(labels.max().item() + 1)

        counts = torch.bincount(labels, minlength=num_classes).float()
        counts[counts == 0] = 1.0  # avoid division by zero for unseen classes

        weights = 1.0 / torch.sqrt(counts) if sqrt else 1.0 / counts
        
        if normalize:
            weights = weights / weights.sum() * num_classes
        
        return weights

    def __len__(self):
        return len(self.samples)

    def num_classes(self):
        return len(self.label_map)


# -------------------------------------------------------------
# video modality
# -------------------------------------------------------------

class VideoDataset(ClassificationDataset):
    """frame-sampled video clips for classification.
    
    for MVFoul: supports single-view and multi-view modes. In multi-view 
    training, two views are randomly sampled per clip; at test time all available
    views are returned and stacked along a view dimension.

    Args:
        config: the loaded YAML configuration.
        annotations_path: path to the annotation JSON file.
        processor: HuggingFace image processor (used only for HuggingFace models).
        split: one of "train", "valid", "test".
    """

    def __init__(self, config, annotations_path, processor, split="train"):
        super().__init__(config, annotations_path, split)

        self.processor = processor
        self.view_type = getattr(config.DATA, "view_type", "single")
        self.num_frames = getattr(config.DATA, "num_frames", None)
        self.input_fps = getattr(config.DATA, "input_fps", None)
        self.transform = build_transform(config, mode=self.split)

    def _select_views(self, video_paths):
        """ choose which camera views to laod for this sample.

        Args:
            video_paths: list of available view paths for the clip.

        Returns:
            a (possibly subsampled) list of paths.
        
        """
        if self.view_type == "single":
            return [video_paths[0]]

        if self.split.lower() == "train" and self.view_type == "multi":
            return random.sample(video_paths, min(2, len(video_paths)))

        return video_paths

    def _load_and_sample_clip(self, path):
        """read a video file, temporally sub-sample, and apply transforms.

        Args:
            path: realtive path (under data_dir) to the video file.

        Returns:
            numpy.ndarray of shape (T, H, W, C).
        """
        full_path = os.path.join(self.config.DATA.data_dir, path)

        if full_path.endswith(".npy"):
            frames = np.load(full_path).astype(np.float32) / 255.0
            if self.transform is not None:
                frames = self.transform(frames)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            frames = (frames - mean) / std
            return frames

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
        label = item.get("label", None)
        if label is not None:
            label = torch.tensor(label, dtype=torch.long)
        video_paths = item["video_paths"]
        sample_id = item["id"]

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
            out = {"pixel_values": pixel_values, "id": sample_id}
            if label is not None:
                out["labels"] = label
            return out
        
        else:
            view_tensors = []
            for path in selected_paths:
                v = self._load_and_sample_clip(path)

                if path.endswith(".npy"):
                    # frames_npy
                    v = torch.from_numpy(v)  # (T, H, W, C)
                else:
                    # existing raw video path: apply torchvision model transforms
                    v = torch.from_numpy(v).permute(0, 3, 1, 2)  # (T, C, H, W)
                    v = get_transforms_model(self.config.MODEL.pretrained_model)(v)  # (C, T, H, W)

                view_tensors.append(v)

            if selected_paths[0].endswith(".npy"):
                # frames_npy: single view, return (T, H, W, C) matching pixels_vs_positions
                out = {"pixel_values": view_tensors[0], "id": sample_id}
            else:
                # existing multi-view path: stack to (V, C, T, H, W)
                videos = torch.stack(view_tensors, dim=0)
                out = {"pixel_values": videos, "id": sample_id}
                
            if label is not None:
                out["labels"] = label
            return out


# -------------------------------------------------------------
# tracking modality
# -------------------------------------------------------------

class TrackingDataset(ClassificationDataset):
    """graph-based classification dataset built from player tracking data.
    
    each sample is a temporal sequence of per-frame graphs where nodes
    represent the ball and 22 players, and edges encode spatial or
    tactical relationships (see build_edge_index).

    supports optional preloading of all clips into memory and 
    training-time augmentation (horizontal/vertical/team flip).

    Args:
        config: the loaded YAML configuration.
        annotations_path: path to the annotation JSON file.
        split: one of "train", "valid", "test".
    """
        
    def __init__(self, config, annotations_path, split="train"):
        super().__init__(config, annotations_path, split)

        from opensportslib.datasets.utils.tracking import (
            FEATURE_DIM,
            NUM_OBJECTS,
            HorizontalFlip,
            TeamFlip,
            VerticalFlip,
            build_edge_index,
        )

        # storing references for the constants without repeating the import.
        self._NUM_OBJECTS = NUM_OBJECTS
        self._FEATURE_DIM = FEATURE_DIM
        self._build_edge_index = build_edge_index

        self.num_frames = config.DATA.num_frames
        self.normalize = config.DATA.normalize
        self.edge_type = config.MODEL.edge
        self.k = config.MODEL.k
        self.r = config.MODEL.r
        self.preload_data = config.DATA.preload_data
        
        self.transforms = self._build_transforms(
            config, split, HorizontalFlip, VerticalFlip, TeamFlip
        )

        self.processed_samples = None
        if self.preload_data:
            self._preload_all_data()

    @staticmethod
    def _build_transforms(config, split, HorizontalFlip, VerticalFlip, TeamFlip):
        """assmeble the list of training-time augmentations.

        Args:
            config: the loaded YAML configuration.
            split: dataset split; augmentations are only applied during training.
            HorizontalFlip: augmentation class (passed to avoid re-importing).
            VerticalFlip: augmentation class.
            TeamFlip: augmentation class.

        Returns:
            list of callable augmentation transforms (empty for
            non-training splits).
        """
        if split != "train":
            return []
        
        transforms = []
        aug_config = config.DATA.augmentations

        # augmentation flags are optional in the config; default to off.
        if getattr(aug_config, "horizontal_flip", False):
            transforms.append(HorizontalFlip(probability=0.5))
        
        if getattr(aug_config, "vertical_flip", False):
            transforms.append(VerticalFlip(probability=0.5))
        
        if getattr(aug_config, "team_flip", False):
            transforms.append(TeamFlip(probability=0.5))
        
        return transforms

    def _preload_all_data(self):
        """parse every clip and cache features and edge indices in memory.

        edge indices are built on raw (un-augmented, un-normalized)
        features so that the graph topology is deterministic and 
        augmentation-independent.
        """
        from tqdm import tqdm

        from opensportslib.datasets.utils.tracking import (
            compute_deltas,
            parse_frame
        )
        
        print(f"Preloading {len(self.samples)} {self.split} samples into memory...")
        
        self.processed_samples = []
        
        for item in tqdm(self.samples, desc=f"Loading {self.split}"):
            clip_paths = item["video_paths"]
            if not clip_paths:
                continue
            
            clip_path = clip_paths[0]
            df = self._load_tracking_clip(clip_path)

            num_frames = len(df)
            all_features = np.zeros(
                (num_frames, self._NUM_OBJECTS, self._FEATURE_DIM), 
                dtype=np.float32
            )
            all_positions = []
            
            for t, (_, row) in enumerate(df.iterrows()):
                features, positions = parse_frame(row)
                all_features[t] = features
                all_positions.append(positions)
            
            all_features = compute_deltas(all_features)

            # build edge indices on raw features (before any augmentation
            # or normalization) so the graph topology stays consistent
            # regardless of transforms.
            edge_indices = []
            for t in range(num_frames):
                edge_index = self.build_edge_index(
                    all_features[t],
                    all_positions[t],
                    self.edge_type,
                    self.k,
                    self.r
                )
                edge_indices.append(edge_index)
            
            self.processed_samples.append({
                "features": all_features,
                "positions": all_positions,
                "edge_indices": edge_indices,
                "label": item["label"],
                "id": item["id"]
            })
        
        print(f"Loaded {len(self.processed_samples)} {self.split} samples")
    
    def _load_tracking_clip(self, path):
        """read a single parquet tracking clip.
        
        Args:
            path: Relative path (under ``data_dir``) to the parquet
                file.

        Returns:
            ``pandas.DataFrame`` with one row per frame.
        """
        import pandas as pd

        full_path = os.path.join(self.data_dir, path)
        return pd.read_parquet(full_path)
    
    def __getitem__(self, idx):
        if self.preload_data:
            return self._getitem_preloaded(idx)
        else:
            return self._getitem_on_the_fly(idx)
    
    def _getitem_preloaded(self, idx):
        """return a sample from the in-memory cache.

        a copy of the feature array is made before augmentation and 
        normalization so the cached data is never mutated.
        """
        from torch_geometric.data import Data

        from opensportslib.datasets.utils.tracking import normalize_features
        
        sample = self.processed_samples[idx]
        features = sample["features"].copy()
        
        for transform in self.transforms:
            features = transform(features)
        
        if self.normalize:
            features = normalize_features(features)
        
        # build one PyG Data object per frame. The downstream collate function
        # (tracking_collate) uses PyG Batch.from_data_list to merge these across
        # the batch dimension.
        graphs = []
        for t in range(features.shape[0]):
            data = Data(
                x=torch.tensor(features[t], dtype=torch.float),
                edge_index=torch.tensor(
                    sample["edge_indices"][t], dtype=torch.long
                ),
            )
            graphs.append(data)
        
        out = {
            "graphs": graphs,
            "seq_len": len(graphs),
            "id": sample["id"]
        }
        if "label" in sample:
            out["label"] = sample["label"]
        return out

    def _getitem_on_the_fly(self, idx):
        """load, parse, and process a single sample from disk."""
        from torch_geometric.data import Data
        
        from opensportslib.datasets.utils.tracking import (
            compute_deltas,
            normalize_features,
            parse_frame,
        )

        item = self.samples[idx]
        label = item["label"]
        
        clip_paths = item["video_paths"]
        if not clip_paths:
            raise ValueError(f"No tracking paths found for item {idx}")
        
        clip_path = clip_paths[0]
        df = self._load_tracking_clip(clip_path)
        
        num_frames = len(df)
        all_features = np.zeros(
            (num_frames, self._NUM_OBJECTS, self._FEATURE_DIM), 
            dtype=np.float32
        )
        all_positions = []
        
        for t, (_, row) in enumerate(df.iterrows()):
            features, positions = parse_frame(row)
            all_features[t] = features
            all_positions.append(positions)
        
        all_features = compute_deltas(all_features)
        
        # edge indices are built on raw features (before any augmentation /
        # normalization) so the graph structure is augmentation-invariant.
        edge_indices = []
        for t in range(num_frames):
            edge_index = self._build_edge_index(
                all_features[t],
                all_positions[t],
                self.edge_type,
                self.k,
                self.r
            )
            edge_indices.append(edge_index)
        
        for transform in self.transforms:
            all_features = transform(all_features)
        
        if self.normalize:
            all_features = normalize_features(all_features)
        
        graphs = []
        for t in range(num_frames):
            data = Data(
                x=torch.tensor(all_features[t], dtype=torch.float),
                edge_index=torch.tensor(edge_indices[t], dtype=torch.long),
            )
            graphs.append(data)
        
        out = {
            "graphs": graphs,
            "seq_len": len(graphs),
            "id": item["id"]
        }
        if label is not None:
            out["label"] = label
        return out
        