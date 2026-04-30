import os
import torch
import random
from torch.utils.data import Dataset
import tempfile
import copy
import math
import numpy as np
import json
import logging
import tqdm
from opensportslib.core.utils.default_args import get_default_args_dataset
from opensportslib.core.utils.load_annotations import get_repartition_gpu
from opensportslib.core.utils.video_processing import feats2clip, getChunks_anchors, getTimestampTargets, oneHotToShifts
from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import (
    AverageMeter,
    EVENT_DICTIONARY_V2,
    INVERSE_EVENT_DICTIONARY_V2,
)
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1

try:
    from nvidia.dali import pipeline_def, backend
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    DALI_AVAILABLE = True

except ImportError:
    DALI_AVAILABLE = False
    # Optional: placeholders (prevents NameError)
    pipeline_def = None
    backend = None
    fn = None
    types = None
    DALIGenericIterator = object

if DALI_AVAILABLE:
    def dali_pipeline_def(func):
        return pipeline_def(func)
else:
    def dali_pipeline_def(func):
        return func  # dummy decorator
    
class LocalizationDataset(Dataset):
    def __init__(self, config, annotations_path=None, processor=None, split="train"):
        self.config = config
        self.split = split
        self.config.TRAIN.repartitions = get_repartition_gpu(self.config.SYSTEM.GPU)
        if split == "train":
            self.cfg = self.config.DATA.train
            self.default_args = get_default_args_dataset("train", self.config)
        elif split == "valid":
            self.cfg = self.config.DATA.valid
            self.default_args = get_default_args_dataset("valid", self.config)
        elif split == "test":
            self.cfg = self.config.DATA.test
            self.default_args = get_default_args_dataset("test", self.config)
        elif split == "valid_data_frames":
            self.cfg = self.config.DATA.valid_data_frames
            self.default_args = get_default_args_dataset("valid_data_frames", self.config)
        #self.built_dataset = self.building_dataset(cfg=cfg, default_args=default_args)
        #self.data_loader = self.building_dataloader(self.built_dataset, cfg=cfg.dataloader, gpu=0, dali=True)

    
    def building_dataset(self, cfg, gpu=None, default_args=None):
        if cfg.type == "SoccerNetClips" or cfg.type == "SoccerNetGames":
            if cfg.split == None:
                dataset = SoccerNetGameClips(
                    path=cfg.video_path,
                    features=cfg.features,
                    version=cfg.version,
                    framerate=cfg.framerate,
                    window_size=cfg.window_size,
                )
            else:
                dataset = SoccerNetClips(
                    path=cfg.video_path,
                    features=cfg.features,
                    split=cfg.split,
                    version=cfg.version,
                    framerate=cfg.framerate,
                    window_size=cfg.window_size,
                    train=True if cfg.type == "SoccerNetClips" else False,
                )
        elif cfg.type == "SoccerNetClipsCALF" or cfg.type == "SoccerNetClipsTestingCALF":
            if cfg.split == None:
                dataset = SoccerNetGameClipsChunks(
                    path=cfg.video_path,
                    features=cfg.features,
                    framerate=cfg.framerate,
                    chunk_size=cfg.chunk_size,
                    receptive_field=cfg.receptive_field,
                )
            else:
                dataset = SoccerNetClipsChunks(
                    path=cfg.video_path,
                    features=cfg.features,
                    split=cfg.split,
                    framerate=cfg.framerate,
                    chunk_size=cfg.chunk_size,
                    receptive_field=cfg.receptive_field,
                    chunks_per_epoch=cfg.chunks_per_epoch,
                    gpu=gpu,
                    train=True if cfg.type == "SoccerNetClipsCALF" else False,
                )
        elif cfg.type == "FeatureClipsfromJSON":
            dataset = FeatureClipsfromJSON(
                path=cfg.path,
                features_dir=cfg.video_path,
                classes=cfg.classes,
                framerate=cfg.framerate,
                window_size=cfg.window_size,
            )
        elif cfg.type == "FeatureVideosfromJSON":
            dataset = FeatureClipsfromJSON(
                path=cfg.path,
                features_dir=cfg.video_path,
                classes=cfg.classes,
                framerate=cfg.framerate,
                window_size=cfg.window_size,
                train=False,
            )
            # dataset = FeatureVideosfromJSON(path=cfg.path,
            #     framerate=cfg.framerate,
            #     window_size=cfg.window_size)
        elif cfg.type == "FeatureClipChunksfromJson":
            dataset = FeatureClipChunksfromJson(
                path=cfg.path,
                features_dir=cfg.video_path,
                classes=cfg.classes,
                framerate=cfg.framerate,
                chunk_size=cfg.chunk_size,
                receptive_field=cfg.receptive_field,
                chunks_per_epoch=cfg.chunks_per_epoch,
                gpu=gpu,
            )
        elif cfg.type == "FeatureVideosChunksfromJson":
            dataset = FeatureClipChunksfromJson(
                path=cfg.path,
                features_dir=cfg.video_path,
                classes=cfg.classes,
                framerate=cfg.framerate,
                chunk_size=cfg.chunk_size,
                receptive_field=cfg.receptive_field,
                chunks_per_epoch=cfg.chunks_per_epoch,
                gpu=gpu,
                train=False,
            )
        elif cfg.type == "VideoGameWithOpencv":
            dataset_len = self.config.DATA.epoch_num_frames // self.config.DATA.clip_len
            dataset = ActionSpotDataset(
                default_args["classes"],
                cfg.path,
                cfg.video_path,
                self.config.DATA.modality,
                self.config.DATA.clip_len,
                self.config.DATA.input_fps,
                self.config.DATA.extract_fps,
                dataset_len if default_args["train"] else dataset_len // 4,
                is_eval=not default_args["train"],
                crop_dim=self.config.DATA.crop_dim,
                dilate_len=self.config.DATA.dilate_len,
                mixup=self.config.DATA.mixup,
                IMAGENET_MEAN=self.config.DATA.imagenet_mean,
                IMAGENET_STD=self.config.DATA.imagenet_std,
                TARGET_HEIGHT=self.config.DATA.target_height,
                TARGET_WIDTH=self.config.DATA.target_width,
            )
        elif cfg.type == "VideoGameWithOpencvVideo":
            dataset = ActionSpotVideoDataset(
                default_args["classes"],
                cfg.path,
                cfg.video_path,
                self.config.DATA.modality,
                self.config.DATA.clip_len,
                self.config.DATA.input_fps,
                self.config.DATA.extract_fps,
                crop_dim=self.config.DATA.crop_dim,
                overlap_len=getattr(cfg, "overlap_len", self.config.DATA.clip_len // 2),
                IMAGENET_MEAN=self.config.DATA.imagenet_mean,
                IMAGENET_STD=self.config.DATA.imagenet_std,
                TARGET_HEIGHT=self.config.DATA.target_height,
                TARGET_WIDTH=self.config.DATA.target_width,
            )
        elif cfg.type == "VideoGameWithDali":
            if not DALI_AVAILABLE:
                raise ImportError(
                    "NVIDIA DALI is required. "
                    "Install it or use another dataset type."
                )
            loader_batch_size = cfg.dataloader.batch_size // default_args["acc_grad_iter"]
            dataset_len = self.config.DATA.epoch_num_frames // self.config.DATA.clip_len
            dataset = DaliDataSet(
                epochs=default_args["num_epochs"],
                batch_size=loader_batch_size,
                output_map=cfg.output_map,
                devices=(
                    default_args["repartitions"][0]
                    if default_args["train"]
                    else default_args["repartitions"][1]
                ),
                #devices=list(range(gpu)),
                classes=default_args["classes"],
                label_file=cfg.path,
                modality=self.config.DATA.modality,
                clip_len=self.config.DATA.clip_len,
                dataset_len=dataset_len if default_args["train"] else dataset_len // 4,
                video_dir=cfg.video_path,
                input_fps=self.config.DATA.input_fps,
                extract_fps=self.config.DATA.extract_fps,
                IMAGENET_MEAN=self.config.DATA.imagenet_mean,
                IMAGENET_STD=self.config.DATA.imagenet_std,
                TARGET_HEIGHT=self.config.DATA.target_height,
                TARGET_WIDTH=self.config.DATA.target_width,
                is_eval=False if default_args["train"] else True,
                crop_dim=self.config.DATA.crop_dim,
                dilate_len=self.config.DATA.dilate_len,
                mixup=self.config.DATA.mixup,
            )
        elif cfg.type == "VideoGameWithDaliVideo":
            if not DALI_AVAILABLE:
                raise ImportError(
                    "NVIDIA DALI is required. "
                    "Install it or use another dataset type."
                )
            dataset = DaliDataSetVideo(
                batch_size=cfg.dataloader.batch_size,
                output_map=cfg.output_map,
                #devices=list(range(gpu)),
                devices=default_args["repartitions"][1],
                classes=default_args["classes"],
                label_file=cfg.path,
                modality=self.config.DATA.modality,
                clip_len=self.config.DATA.clip_len,
                video_dir=cfg.video_path,
                input_fps=self.config.DATA.input_fps,
                extract_fps=self.config.DATA.extract_fps,
                IMAGENET_MEAN=self.config.DATA.imagenet_mean,
                IMAGENET_STD=self.config.DATA.imagenet_std,
                TARGET_HEIGHT=self.config.DATA.target_height,
                TARGET_WIDTH=self.config.DATA.target_width,
                overlap_len=cfg.overlap_len,
                crop_dim=self.config.DATA.crop_dim,
            )
        else:
            dataset = None
        return dataset


    def building_dataloader(self, dataset, cfg, gpu, dali):
        """Build a dataloader from config dict.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            default_args (dict | None, optional): Default initialization arguments.
                Default: None.

        Returns:
            Dataloader: The constructed dataloader.
        """
        def worker_init_fn(id):
            random.seed(id + 100 * 100)
        if dali:
            return dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers if gpu >= 0 else 0,
            pin_memory=cfg.pin_memory if gpu >= 0 else False,
            prefetch_factor=(
                getattr(cfg, "prefetch_factor", None)
            ),
            worker_init_fn=worker_init_fn
        )
        return dataloader


class FrameReader:
    """Class used to read a video and create a clip of frames by applying some transformations.

    Args:
        modality (string): Modality of the frames.
        crop_transform : Transformations that apply to frame for cropping.
        img_transform : Transformations that apply to frame like GaussianBlur and Normalization.
        same_transform (bool): Whether to apply same trasnforms to every frame of a same clip.
        sample_fps (int): Fps at which we want to extract frames from the video.
    """

    def __init__(
        self,
        modality,
        crop_transform,
        img_transform,
        same_transform,
        sample_fps=2,
        TARGET_HEIGHT=224,
        TARGET_WIDTH=398,
    ):
        self._is_flow = modality == "flow"
        self._crop_transform = crop_transform
        self._img_transform = img_transform
        self._same_transform = same_transform
        self._sample_fps = sample_fps
        self.TARGET_HEIGHT = TARGET_HEIGHT
        self.TARGET_WIDTH = TARGET_WIDTH

    def adapt_frame_ocv(self, frame):
        """Apply some modifications to the frame to have the expected shape and format.

        Args:
            frame (np.array).

        Returns:
            img (torch.tensor).
        """
        import cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(frame).float() / 255
        img = img.permute(2, 0, 1)
        if self._is_flow:
            img = img[1:, :, :]  # GB channels contain data
        return img

    def load_frames_ocv(self, video_name, start, end, pad=False):
        """Load frames from a video to create a clip of frames.

        Args:
            video_name (string): The path of the video. This is the full path of the file if we infer a single video
            or the partial path if we infer from json file.
            start (int): Start frame at which we load the clip.
            end (int): End frame at which we finish the clip.
            pad (bool): Whether to apply padding to the clip or not.
        """
        import cv2
        import torch.nn as nn

        def get_stride(src_fps):
            """Get stride to apply based on the input and output fps.

            Args:
                src_fps (int): The input fps of the video.
            """
            if self._sample_fps <= 0:
                stride_extract = 1
            else:
                stride_extract = int(src_fps / self._sample_fps)
            return stride_extract

        # if self.infer:
        #     video_path = video_name
        # else:
        #     video_path = os.path.join(self._video_dir, video_name + self.extension)
        vc = cv2.VideoCapture(video_name)
        fps = vc.get(cv2.CAP_PROP_FPS)

        oh = self.TARGET_HEIGHT
        ow = self.TARGET_WIDTH

        frames = []
        rand_crop_state = None
        rand_state_backup = None
        ret = []
        n_pad_start = 0
        n_pad_end = 0
        stride_extract = get_stride(fps)
        vc.set(cv2.CAP_PROP_POS_FRAMES, start * stride_extract)
        out_frame_num = 0
        i = 0
        while True:
            ret, frame = vc.read()
            if ret:
                if i % stride_extract == 0:
                    if frame.shape[0] != oh or frame.shape[1] != ow:
                        frame = cv2.resize(frame, (ow, oh))
                    img = self.adapt_frame_ocv(frame)
                    if self._crop_transform:
                        if self._same_transform:
                            if rand_crop_state is None:
                                rand_crop_state = random.getstate()
                            else:
                                rand_state_backup = random.getstate()
                                random.setstate(rand_crop_state)

                        img = self._crop_transform(img)

                        if rand_state_backup is not None:
                            # Make sure that rand state still advances
                            random.setstate(rand_state_backup)
                            rand_state_backup = None
                    if not self._same_transform:
                        img = self._img_transform(img)
                    frames.append(img)
                    out_frame_num += 1
                i += 1
                if out_frame_num == (end - start):
                    break
            else:
                n_pad_end = (end - start) - out_frame_num
                break
        vc.release()
        # In the multicrop case, the shape is (B, T, C, H, W)
        frames = torch.stack(frames, dim=int(len(frames[0].shape) == 4))
        if self._same_transform:
            frames = self._img_transform(frames)

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            frames = nn.functional.pad(
                frames, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0)
            )
        return frames


class ActionSpotDataset(Dataset):
    """Class that overrides Dataset class. This class is to prepare training data using opencv.
    Training data consists of frames, associated labels and a boolean indicating if the clip of frames contains an event.
    A training sample can be mixed up with another one if mixup is used or not.
    In particular, a training sample contains the following informations without mixup:
        "frame": The frames.
        "contains_event": True if event occurs within these frames, False otherwise.
        "label": The labels associated to the frames.
    and the following informations with mixup:
        "frame": A combination of the frames of the first video and the second one.
        "contains_event": True if event occurs within these frames, False otherwise.
        "label": Rearrangement of the labels of each video.
        "mix_frame": Frames of the second video.
        "mix_weight": The weight that have been used for mixing frames and labels.

    Args:
        classes (dict): dict of class names to idx.
        label_file (list[string]|string): Path to label json files. Can be a single file or a list or a json files.
        video_dir (list[string]|string): Path to folders where videos are located. Can be a single folder or a list of folders. Must match the number of json files.
        modality (string): [rgb] Modality of the frame.
        clip_len (int): Length of a clip of frames.
        input_fps (int): Fps of the input videos.
        extract_fps (int): Fps at which we want to extract frames.
        dataset_len (int): Number of clips.
        is_eval (bool): Disable random augmentation
            Default: True.
        crop_dim (int): The dimension for cropping frames.
            Default: None.
        same_transform (bool): Apply the same random augmentation to each frame in a clip.
            Default: True.
        dilate_len (int): Dilate ground truth labels.
            Default: 0.
        mixup (bool): Whether to mixup clips of two videos or not.
            Default: False.
        pad_len (int): Number of frames to pad the start and end of videos.
            Default: DEFAULT_PAD_LEN.
        fg_upsample: Sample foreground explicitly.
            Default: -1.
    """

    def __init__(
        self,
        classes,
        label_file,
        video_dir,
        modality,
        clip_len,
        input_fps,
        extract_fps,
        dataset_len,
        is_eval=True,
        crop_dim=None,
        # stride=1,  # Downsample frame rate
        same_transform=True,
        dilate_len=0,
        mixup=False,
        pad_len=5,
        fg_upsample=-1,
        IMAGENET_MEAN=[0.485, 0.456, 0.406],
        IMAGENET_STD=[0.229, 0.224, 0.225],
        TARGET_HEIGHT=224,
        TARGET_WIDTH=398,
    ):
        import random
        from opensportslib.core.utils.load_annotations import annotationstoe2eformat
        from opensportslib.core.utils.video_processing import _get_deferred_rgb_transform, _get_img_transforms

        self._src_file = label_file
        self._labels, self.task_name = annotationstoe2eformat(
            label_file, video_dir, input_fps, extract_fps, False
        )
        # self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x["video"]: i for i, x in enumerate(self._labels)}
        # Sample videos weighted by their length
        num_frames = [v["num_frames"] for v in self._labels]
        self._weights_by_length = np.array(num_frames) / np.sum(num_frames)

        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = 1
        assert self._stride > 0
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0
        self._is_eval = is_eval

        # Label modifications
        self._dilate_len = dilate_len
        self._fg_upsample = fg_upsample

        # Sample based on foreground labels
        if self._fg_upsample > 0:
            self._flat_labels = []
            for i, x in enumerate(self._labels):
                for event in x["events"]:
                    if event["frame"] < x["num_frames"]:
                        self._flat_labels.append((i, event["frame"]))

        self._mixup = mixup

        self.IMAGENET_MEAN = IMAGENET_MEAN
        self.IMAGENET_STD = IMAGENET_STD
        self.TARGET_HEIGHT = TARGET_HEIGHT
        self.TARGET_WIDTH = TARGET_WIDTH
        # Try to do defer the latter half of the transforms to the GPU
        self._gpu_transform = None
        if not is_eval and same_transform:
            if modality == "rgb":
                print("=> Deferring some RGB transforms to the GPU!")
                self._gpu_transform = _get_deferred_rgb_transform(self.IMAGENET_MEAN, self.IMAGENET_STD)


        crop_transform, img_transform = _get_img_transforms(
            self.IMAGENET_MEAN,
            self.IMAGENET_STD,
            is_eval,
            crop_dim,
            modality,
            same_transform,
            defer_transform=self._gpu_transform is not None,
        )

        self._frame_reader = FrameReader(
            modality,
            crop_transform,
            img_transform,
            same_transform,
            extract_fps,
            self.TARGET_HEIGHT,
            self.TARGET_WIDTH
        )

    def load_frame_gpu(self, batch, device):
        """Load frame to gpu by appliyng some transformations or not.

        Args:
            batch: Batch containing data.
            device: The device on which to load the frames.
        """
        from opensportslib.core.utils.video_processing import _load_frame_deferred
        if self._gpu_transform is None:
            frame = batch["frame"].to(device)
        else:
            frame = _load_frame_deferred(self._gpu_transform, batch, device)
        return frame

    def _sample_uniform(self):
        """Sample video metadata and a base start index uniformly based on video lengths and weights.
        Returns:
            video_meta: metadata of a video.
            base_idx: base start index for the video processing.
        """
        video_meta = random.choices(self._labels, weights=self._weights_by_length)[0]

        video_len = video_meta["num_frames"]
        base_idx = -self._pad_len * self._stride + random.randint(
            0,
            max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride),
        )
        return video_meta, base_idx

    def _sample_foreground(self):
        """Samples video metadata and a base start index focusing on foreground labels.
        Returns:
            video_meta: metadata of a video.
            base_idx: base start index for the video processing.
        """
        video_idx, frame_idx = random.choices(self._flat_labels)[0]
        video_meta = self._labels[video_idx]
        video_len = video_meta["num_frames"]

        lower_bound = max(
            -self._pad_len * self._stride, frame_idx - self._clip_len * self._stride + 1
        )
        upper_bound = min(
            video_len - 1 + (self._pad_len - self._clip_len) * self._stride, frame_idx
        )

        base_idx = (
            random.randint(lower_bound, upper_bound)
            if upper_bound > lower_bound
            else lower_bound
        )

        assert base_idx <= frame_idx
        assert base_idx + self._clip_len > frame_idx
        return video_meta, base_idx

    def _get_one(self):
        """Get a training sample for one video."""
        if self._fg_upsample > 0 and random.random() >= self._fg_upsample:
            video_meta, base_idx = self._sample_foreground()
        else:
            video_meta, base_idx = self._sample_uniform()

        labels = np.zeros(self._clip_len, np.int64)
        for event in video_meta["events"]:
            event_frame = event["frame"]

            # Index of event in label array
            label_idx = (event_frame - base_idx) // self._stride
            if (
                label_idx >= -self._dilate_len
                and label_idx < self._clip_len + self._dilate_len
            ):
                label = self._class_dict[event["label"]]
                for i in range(
                    max(0, label_idx - self._dilate_len),
                    min(self._clip_len, label_idx + self._dilate_len + 1),
                ):
                    labels[i] = label
        frames = self._frame_reader.load_frames_ocv(
            video_meta["video"],
            base_idx,
            base_idx + self._clip_len * self._stride,
            pad=True,
        )

        return {
            "frame": frames,
            "contains_event": int(np.sum(labels) > 0),
            "label": labels,
        }

    def __getitem__(self, unused):
        """Get a training sample based on one video without mixup, two otherwise."""
        ret = self._get_one()

        if self._mixup:
            mix = self._get_one()  # Sample another clip
            l = random.betavariate(0.2, 0.2)
            label_dist = np.zeros((self._clip_len, len(self._class_dict) + 1))
            label_dist[range(self._clip_len), ret["label"]] = l
            label_dist[range(self._clip_len), mix["label"]] += 1.0 - l

            if self._gpu_transform is None:
                ret["frame"] = l * ret["frame"] + (1.0 - l) * mix["frame"]
            else:
                ret["mix_frame"] = mix["frame"]
                ret["mix_weight"] = l

            ret["contains_event"] = max(ret["contains_event"], mix["contains_event"])
            ret["label"] = label_dist

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        from core.utils.config import _print_info_helper
        _print_info_helper(self._src_file, self._labels)



class DatasetVideoSharedMethods:
    def get_labels(self, video):
        """Get labels of a video.

        Args:
            video (string): Name of the video.

        Returns:
            labels (np.array): Array of length being the number of frame with elements being the index of the class.
        """
        meta = self._labels[self._video_idxs[video]]
        num_frames = meta["num_frames"]
        num_labels = num_frames // self._stride
        if num_frames % self._stride != 0:
            num_labels += 1
        labels = np.zeros(num_labels, np.int64)
        for event in meta["events"]:
            frame = event["frame"]
            if frame < num_frames:
                labels[frame // self._stride] = self._class_dict[event["label"]]
        return labels

    @property
    def augment(self):
        """Whether flip or multi cropping have been applied to frames or not."""
        return self._flip or self._multi_crop

    @property
    def videos(self):
        """Return a list containing metadatas of videos sorted by their names."""
        # return [
        #     (v['video'], v['num_frames_dali'] // self._stride,
        #      v['fps'] / self._stride) for v in self._labels]
        return sorted(
            [
                (
                    v["path"],
                    # os.path.splitext(v["path"])[0],
                    v["num_frames"] // self._stride,
                    v["fps"] / self._stride,
                )
                for v in self._labels
            ]
        )

    @property
    def labels(self):
        """Return the metadatas containing in the json file."""
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                x_copy["fps"] /= self._stride
                x_copy["num_frames"] //= self._stride
                for e in x_copy["events"]:
                    e["frame"] //= self._stride
                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x["num_frames"] for x in self._labels])
        num_events = sum([len(x["events"]) for x in self._labels])
        print(
            "{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg".format(
                self._src_file,
                len(self._labels),
                num_frames,
                self._stride,
                num_events / num_frames * 100,
            )
        )

class ActionSpotVideoDataset(Dataset, DatasetVideoSharedMethods):
    """Class that overrides Dataset class. This class is to prepare testing data.
    Testing data consists of frames, the name of the video and index of the first frame in the video.
    This class can process as input a json file containing metadatas of video or just a video.
    Args:
        classes (dict): dict of class names to idx.
        label_file (string): Can be path to label json or path of a video.
        video_dir (string): path to folder where videos are located.
        modality (string): [rgb] Modality of the frame.
        clip_len (int): Length of a clip of frames.
        input_fps (int): Fps of the input videos.
        extract_fps (int): Fps at which we want to extract frames.
        overlap_len (int): The number of overlapping frames between consecutive clips.
        crop_dim (int): The dimension for cropping frames.
            Default: None.
        flip (bool): Whether to flip or not the frames.
            Default: False.
        multi_crop (bool): Whether multi croping or not
            Default: False.
        skip_partial_end (bool): Whether to skip a partial number of clips at the end.
            Default: True.
        pad_len (int): Number of frames to pad the start and end of videos.
            Default: DEFAULT_PAD_LEN.
    """

    def __init__(
        self,
        classes,
        label_file,
        video_dir,
        modality,
        clip_len,
        input_fps,
        extract_fps,
        overlap_len=0,
        crop_dim=None,
        pad_len=5,
        flip=False,
        multi_crop=False,
        skip_partial_end=True,
        IMAGENET_MEAN=[0.485, 0.456, 0.406],
        IMAGENET_STD=[0.229, 0.224, 0.225],
        TARGET_HEIGHT=224,
        TARGET_WIDTH=398,
    ):
        from opensportslib.core.utils.load_annotations import annotationstoe2eformat, construct_labels
        from opensportslib.core.utils.video_processing import _get_img_transforms
    
        self._src_file = label_file
        if label_file.endswith(".json"):
            self._labels, self.task_name = annotationstoe2eformat(
                label_file, video_dir, input_fps, extract_fps, False
            )
            # self._labels = load_json(label_file)
        else:
            self._labels, _ = construct_labels(label_file, extract_fps)
        # self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x["path"]: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        stride = 1
        self._stride = stride
        self.IMAGENET_MEAN = IMAGENET_MEAN
        self.IMAGENET_STD = IMAGENET_STD
        self.TARGET_HEIGHT = TARGET_HEIGHT
        self.TARGET_WIDTH = TARGET_WIDTH
        crop_transform, img_transform = _get_img_transforms(
            self.IMAGENET_MEAN,
            self.IMAGENET_STD,
            is_eval=True,
            crop_dim=crop_dim,
            modality=modality,
            same_transform=True,
            multi_crop=multi_crop,
        )

        # No need to enforce same_transform since the transforms are
        # deterministic
        self._frame_reader = FrameReader(
            modality,
            crop_transform,
            img_transform,
            False,
            extract_fps,
            self.TARGET_HEIGHT,
            self.TARGET_WIDTH
        )

        self._flip = flip
        self._multi_crop = multi_crop

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(
                    0, l["num_frames"] - (overlap_len * stride) * int(skip_partial_end)
                ),  # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride,
            ):
                has_clip = True
                self._clips.append((l["path"], l["video"], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        """Get a dict of metadata containing the name of the video, the index of the first frame and the clip of frame.

        Args:
            idx (int): The index of the clip in the list of clips.

        Returns:
            dict :{"video","start","frame"}.
        """
        video_path, video_name, start = self._clips[idx]
        frames = self._frame_reader.load_frames_ocv(
            video_name, start, start + self._clip_len * self._stride, pad=True
        )
        # ,stride=self._stride)

        if self._flip:
            frames = torch.stack((frames, frames.flip(-1)), dim=0)

        return {"video": video_path, "start": start // self._stride, "frame": frames}

if DALI_AVAILABLE:
    class DaliDataSet(DALIGenericIterator):
        """Class that overrides DALIGenericIterator class. This class is to prepare training data using nvidia dali.
        Training data consists of frames, associated labels and a boolean indicating if the clip of frames contains an event.
        A training sample can be mixed up with another one if mixup is used or not.
        In particular, a training sample contains the following informations without mixup:
            "frame": The frames.
            "contains_event": True if event occurs within these frames, False otherwise.
            "label": The labels associated to the frames.
        and the following informations with mixup:
            "frame": A combination of the frames of the first video and the second one.
            "contains_event": True if event occurs within these frames, False otherwise.
            "label": Rearrangement of the labels of each video.
            "mix_frame": Frames of the second video.
            "mix_weight": The weight that have been used for mixing frames and labels.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int).
            output_map (List[string]): List of strings which maps consecutive outputs of DALI pipelines to user specified name. Outputs will be returned from iterator as dictionary of those names. Each name should be distinct.
            devices (list[int]): List of indexes of gpu to use.
            classes (dict): dict of class names to idx.
            label_file (list[string]|string): Paths to label jsons. Can be a single json file or a list of json files.
            clip_len (int): Length of a clip of frames.
            dataset_len (int): Number of clips.
            video_dir (list[string]|string): Paths to folder where videos are located. Can be a single folder file or a list of folders. Must match the number of json files.
            input_fps (int): Fps of the input videos.
            extract_fps (int): Fps at which we extract the frames.
            is_eval (bool): Disable random augmentation
                Default: True.
            crop_dim (int): The dimension for cropping frames.
                Default: None.
            dilate_len (int): Dilate ground truth labels.
                Default: 0.
            mixup (bool): Whether to mixup clips of two videos or not.
                Default: False.
        """

        def __init__(
            self,
            epochs,
            batch_size,
            output_map,
            devices,
            classes,
            label_file,
            modality,
            clip_len,
            dataset_len,
            video_dir,
            input_fps,
            extract_fps,
            IMAGENET_MEAN,
            IMAGENET_STD,
            TARGET_HEIGHT,
            TARGET_WIDTH,
            is_eval=True,
            crop_dim=None,
            dilate_len=0,
            mixup=False,
        ):
            if not DALI_AVAILABLE:
                raise ImportError(
                    "NVIDIA DALI is required for VideoGameWithDali. "
                    "Install it or use another dataset type."
                )
            import random
            from opensportslib.core.utils.load_annotations import annotationstoe2eformat
            from opensportslib.core.utils.video_processing import distribute_elements, _get_deferred_rgb_transform, get_stride

            self._src_file = label_file
            # self._labels = load_json(label_file)
            self._labels, self.task_name = annotationstoe2eformat(
                label_file, video_dir, input_fps, extract_fps, True
            )
            self._class_dict = classes
            self.original_batch_size = batch_size

            if mixup:
                self.batch_size = 2 * batch_size
            else:
                self.batch_size = batch_size

            self.batch_size_per_pipe = distribute_elements(self.batch_size, len(devices))

            self.batch_size = batch_size
            self.nb_videos = dataset_len * 2 if mixup else dataset_len
            self.mixup = mixup
            self.output_map = output_map
            self.devices = devices
            self.is_eval = is_eval
            self.crop_dim = crop_dim
            self.dilate_len = dilate_len
            self.clip_len = clip_len
            self.IMAGENET_MEAN = IMAGENET_MEAN
            self.IMAGENET_STD = IMAGENET_STD
            self.TARGET_HEIGHT = TARGET_HEIGHT
            self.TARGET_WIDTH = TARGET_WIDTH

            self._stride = get_stride(input_fps, extract_fps)

            if is_eval:
                nb_clips_per_video = math.ceil(dataset_len / len(self._labels)) * epochs
            else:
                nb_clips_per_video = math.ceil(dataset_len / len(self._labels)) * epochs

            if mixup:
                nb_clips_per_video = nb_clips_per_video * 2

            file_list_txt = ""
            for index, video in enumerate(self._labels):
                video_path = video["video"]
                #print("video_path :", video_path)
                # video_path = os.path.join(video_dir, video["video"] + extension)
                for _ in range(nb_clips_per_video):
                    #print(video["num_frames"], (clip_len + 1))
                    random_start = random.randint(1, video["num_frames"] - (clip_len + 1))
                    file_list_txt += f"{video_path} {index} {random_start * self._stride} {(random_start+clip_len) * self._stride}\n"

            tf = tempfile.NamedTemporaryFile()
            tf.write(str.encode(file_list_txt))
            tf.flush()

            self.pipes = [
                self.video_pipe(
                    batch_size=self.batch_size_per_pipe[index],
                    sequence_length=self.clip_len,
                    stride_dali=self._stride,
                    step=-1,
                    num_threads=8,
                    device_id=i,
                    file_list=tf.name,
                    shard_id=index,
                    num_shards=len(devices),
                )
                for index, i in enumerate(devices)
            ]

            for pipe in self.pipes:
                pipe.build()

            # Pipeline returns (video, label_idx, frame_num) - label processing
            # is done post-hoc in get_attr to avoid DALI 2.0 fn.python_function issues
            internal_output_map = ['data', 'label_idx', 'frame_num']
            super().__init__(self.pipes, internal_output_map, size=self.nb_videos)

            self.device = torch.device(
                "cuda:{}".format(self.devices[1 if len(self.devices) > 1 else 0])
            )

            self.gpu_transform = None
            if not self.is_eval:
                self.gpu_transform = _get_deferred_rgb_transform(self.IMAGENET_MEAN, self.IMAGENET_STD)
                # self.gpu_transform = self.get_deferred_rgb_transform()

        def __next__(self):
            out = super().__next__()
            ret = self.getitem(out)
            if self.is_eval:
                frame = ret["frame"]
            else:
                frame = self.load_frame_deferred(self.gpu_transform, ret)
            return {"frame": frame, "label": ret["label"]}

        def delete(self):
            """Useful method to free memory used by gpu when the dataset is no longer needed."""
            for pipe in self.pipes:
                pipe.__del__()
                del pipe
            backend.ReleaseUnusedMemory()

        def get_attr(self, batch):
            """Return a dictionnary containing attributes of the batch.

            Args:
                batch (dict).

            Returns:
                dict :{"frames","contains_event","labels"}.
            """
            batch_label_idx = batch["label_idx"]
            batch_frame_num = batch["frame_num"]
            batch_images = batch["data"]

            batch_size = batch_label_idx.shape[0]
            batch_labels = torch.zeros(batch_size, self.clip_len, dtype=torch.int64)
            for b in range(batch_size):
                video_idx = int(batch_label_idx[b].item())
                frame_num = int(batch_frame_num[b].item())
                batch_labels[b] = torch.from_numpy(
                    self._compute_labels(video_idx, frame_num)
                )

            sum_labels = torch.sum(
                batch_labels, dim=1 if len(batch_labels.shape) == 2 else 0
            )
            contains_event = (sum_labels > 0).int()
            return {
                "frame": batch_images,
                "contains_event": contains_event,
                "label": batch_labels,
            }

        def move_to_device(self, batch):
            """Move all tensors of the batch to a device. Useful since samples are handled by different gpus in a first time.

            Args:
                batch : Batch containing samples that are located on different gpus.
            """
            for key, tensor in batch.items():
                batch[key] = tensor.to(self.device)

        def getitem(self, data):
            """Construct and return a batch. Mixup clips of two videos if mixup is true.

            Args:
                data: List of samples that are located on different gpus.
            """
            nb_devices = len(self.devices)
            if nb_devices == 1:
                ret = self.get_attr(data[0])
            if nb_devices >= 2:
                ret = self.get_attr(data[0])
                mix = self.get_attr(data[1])
                self.move_to_device(ret)
                self.move_to_device(mix)

            if nb_devices >= 4:
                ret2 = self.get_attr(data[2])
                mix2 = self.get_attr(data[3])
                self.move_to_device(ret2)
                self.move_to_device(mix2)

            if self.mixup:
                if nb_devices == 1:
                    mix = {}
                    for key, tensor in ret.items():
                        ret[key], mix[key] = torch.chunk(tensor, 2, dim=0)
                if nb_devices >= 4:
                    for key, tensor in ret.items():
                        ret[key] = torch.cat((tensor, ret2[key]))
                    for key, tensor in mix.items():
                        mix[key] = torch.cat((tensor, mix2[key]))

                l = [random.betavariate(0.2, 0.2) for i in range(ret["frame"].shape[0])]
                l = torch.tensor(l)
                label_dist = torch.zeros(
                    (ret["frame"].shape[0], self.clip_len, len(self._class_dict) + 1),
                    device=self.device,
                )
                for i in range(ret["frame"].shape[0]):
                    label_dist[i, range(self.clip_len), ret["label"][i]] = l[i].item()
                    label_dist[i, range(self.clip_len), mix["label"][i]] += (
                        1.0 - l[i].item()
                    )

                if self.gpu_transform is None:
                    for i in range(ret["frame"].shape[0]):
                        ret["frame"][i] = (
                            l[i].item() * ret["frame"][i]
                            + (1.0 - l[i].item()) * mix["frame"][i]
                        )
                else:
                    ret["mix_frame"] = mix["frame"]
                    ret["mix_weight"] = l

                ret["contains_event"] = torch.max(
                    ret["contains_event"], mix["contains_event"]
                )
                ret["label"] = label_dist
            else:
                if nb_devices >= 4:
                    for key, tensor in ret.items():
                        ret[key] = torch.cat((tensor, mix[key], ret2[key], mix2[key]))
                elif nb_devices >= 2:
                    for key, tensor in ret.items():
                        ret[key] = torch.cat((tensor, mix[key]))
            return ret

        def load_frame_deferred(self, gpu_transform, batch):
            """Load frames on the device and applying some transforms.

            Args:
                gpu_transform : Transform to apply to the frames.
                batch : Batch containing the frames and possibly some other datas as
                "mix_weight" and "mix_frame" is mixup is applied while processing videos.
                device : The device on which we load the data.

            Returns:
                frame (torch.tensor).
            """
            frame = batch["frame"]
            with torch.no_grad():
                for i in range(frame.shape[0]):
                    frame[i] = gpu_transform(frame[i])

                if "mix_weight" in batch:
                    weight = batch["mix_weight"].to(self.device)
                    # weight = batch['mix_weight'].to(torch.device('cuda:0'))
                    frame *= weight[:, None, None, None, None]

                    frame_mix = batch["mix_frame"]
                    for i in range(frame.shape[0]):
                        frame[i] += (1.0 - weight[i]) * gpu_transform(frame_mix[i])

            return frame

        @dali_pipeline_def
        def video_pipe(
            self, file_list, sequence_length, stride_dali, step, shard_id, num_shards
        ):
            """Construct the pipeline to process a video. This pipeline process a clip with specified arguments such as stride,step and sequence length.
            The first step returns clip of frames with associated labels (index of the clip in the list of clips) and the index of the first frame.
            The second step is the cropping, mirroring (only if non eval) and normalizing the frames.
            The last step is to construct the list of labels (corresponding to events) corresponding with the extracted frames.

            Args:
                file_list (string): Path to the file with a list of <file label [start_frame [end_frame]]> values.
                sequence_length (int): Frames to load per sequence.
                stride_dali (int): Distance between consecutive frames in the sequence.
                step(int): Frame interval between each sequence.
                shard_id (int): Index of the shard to read.
                num_shards (int): Partitions the data into the specified number of parts.

            Returns:
                video (torch.tensor): The frames processed.
                label : the list of labels (corresponding to events) corresponding with the extracted frames.
            """
            video, label, frame_num = fn.readers.video_resize(
                device="gpu",
                size=(self.TARGET_HEIGHT, self.TARGET_WIDTH),
                file_list=file_list,
                sequence_length=sequence_length,
                random_shuffle=True,
                shard_id=shard_id,
                num_shards=num_shards,
                image_type=types.RGB,
                file_list_include_preceding_frame=True,
                file_list_frame_num=True,
                enable_frame_num=True,
                stride=stride_dali,
                step=step,
                pad_sequences=True,
                skip_vfr_check=True,
            )
            if self.is_eval:
                video = fn.crop_mirror_normalize(
                    video,
                    dtype=types.FLOAT,
                    # crop = self.crop_dim,
                    crop=(self.crop_dim, self.crop_dim) if self.crop_dim != None else None,
                    out_of_bounds_policy="trim_to_shape",
                    output_layout="FCHW",
                    mean=[self.IMAGENET_MEAN[i] * 255.0 for i in range(len(self.IMAGENET_MEAN))],
                    std=[self.IMAGENET_STD[i] * 255.0 for i in range(len(self.IMAGENET_STD))],
                )
            else:
                video = fn.crop_mirror_normalize(
                    video,
                    dtype=types.FLOAT,
                    output_layout="FCHW",
                    # crop = self.crop_dim,
                    crop=(self.crop_dim, self.crop_dim) if self.crop_dim != None else None,
                    out_of_bounds_policy="trim_to_shape",
                    # crop_w=self.crop_dim, crop_h=self.crop_dim,
                    std=[255, 255, 255],
                    mirror=fn.random.coin_flip(),
                )
            return video, label, frame_num

        def _compute_labels(self, video_idx, frame_num):
            """Construct a label array for a clip. Each element is the class index
            (starting at 1) where an event occurs, 0 otherwise.

            Args:
                video_idx (int): Index of the video in self._labels.
                frame_num (int): Raw start frame number from the reader.

            Returns:
                labels (np.ndarray): Label array of shape (clip_len,).
            """
            video_meta = self._labels[video_idx]
            base_idx = frame_num // self._stride
            labels = np.zeros(self.clip_len, np.int64)

            for event in video_meta["events"]:
                event_frame = event["frame"]
                # Index of event in label array
                label_idx = (event_frame - base_idx) // 1
                if (
                    label_idx >= self.dilate_len
                    and label_idx < self.clip_len + self.dilate_len
                ):
                    label_val = self._class_dict[event["label"]]
                    for i in range(
                        max(0, label_idx - self.dilate_len),
                        min(self.clip_len, label_idx + self.dilate_len + 1),
                    ):
                        labels[i] = label_val
            return labels

        def print_info(self):
            from core.utils.config import _print_info_helper
            _print_info_helper(self._src_file, self._labels)


    class DaliDataSetVideo(DALIGenericIterator, DatasetVideoSharedMethods):
        """Class that overrides DALIGenericIterator class. This class is to prepare testing data using nvidia dali.
        Testing data consists of frames, the name of the video and index of the first frame in the video.
        This class can process as input a json file containing metadatas of video or just a video.

        Args:
            batch_size (int).
            output_map (List[string]): List of strings which maps consecutive outputs of DALI pipelines to user specified name. Outputs will be returned from iterator as dictionary of those names. Each name should be distinct.
            devices (list[int]): List of indexes of gpu to use.
            classes (dict): dict of class names to idx.
            label_file (string): Can be path to label json or path of a video.
            clip_len (int): Length of a clip of frames.
            video_dir (string): path to folder where videos are located.
            input_fps (int): Fps of the input videos.
            extract_fps (int): The fps at which we extract frames. This variable is used if dataset is a single video.
            overlap_len (int): The number of overlapping frames between consecutive clips.
                Default: 0.
            crop_dim (int): The dimension for cropping frames.
                Default: None.
            flip (bool): Whether to flip or not the frames.
                Default: False.
            multi_crop (bool): Whether multi croping or not
                Default: False.
        """

        def __init__(
            self,
            batch_size,
            output_map,
            devices,
            classes,
            label_file,
            modality,
            clip_len,
            video_dir,
            input_fps,
            extract_fps,
            IMAGENET_MEAN,
            IMAGENET_STD,
            TARGET_HEIGHT,
            TARGET_WIDTH,
            overlap_len=0,
            crop_dim=None,
            flip=False,
            multi_crop=False,
        ):
            if not DALI_AVAILABLE:
                raise ImportError(
                    "NVIDIA DALI is required for VideoGameWithDali. "
                    "Install it or use another dataset type."
                )
            import random
            from opensportslib.core.utils.load_annotations import annotationstoe2eformat, construct_labels
            from opensportslib.core.utils.video_processing import distribute_elements, _get_deferred_rgb_transform, get_stride, get_remaining
            self._src_file = label_file
            # self.infer = False
            if label_file.endswith(".json"):
                self._labels, self.task_name = annotationstoe2eformat(
                    label_file, video_dir, input_fps, extract_fps, True
                )
                stride_dali = get_stride(input_fps, extract_fps)
                # self._labels = load_json(label_file)
            else:
                # self.infer = True
                self._labels, stride_dali = construct_labels(label_file, extract_fps)
            # self._labels = self._labels[:3]
            self._class_dict = classes
            self._video_idxs = {x["path"]: i for i, x in enumerate(self._labels)}
            self._clip_len = clip_len
            self.crop_dim = crop_dim
            stride = 1
            self._stride = stride
            self._flip = flip
            self._multi_crop = multi_crop
            self.batch_size = batch_size // len(devices)
            self.devices = devices
            self._clips = []
            self.IMAGENET_MEAN = IMAGENET_MEAN
            self.IMAGENET_STD = IMAGENET_STD
            self.TARGET_HEIGHT = TARGET_HEIGHT
            self.TARGET_WIDTH = TARGET_WIDTH
            file_list_txt = ""
            cmp = 0
            for l in self._labels:
                has_clip = False
                for i in range(
                    1,
                    l[
                        "num_frames"
                    ],  # Need to ensure that all clips have at least one frame
                    (clip_len - overlap_len) * self._stride,
                ):
                    if i + clip_len > l["num_frames"]:
                        end = l["num_frames_base"]
                    else:
                        end = (i + clip_len) * stride_dali
                    has_clip = True
                    self._clips.append((l["path"], l["video"], i))
                    # if self.infer:
                    #     video_path = l["video"]
                    # else:
                    #     video_path = os.path.join(video_dir, l["video"] + extension)
                    video_path = l["video"]
                    file_list_txt += f"{video_path} {cmp} {i * stride_dali} {end}\n"
                    # if cmp2 <5:
                    #     print(file_list_txt)
                    #     cmp2+=1
                    cmp += 1
                last_video = l["video"]
                last_path = l["path"]
                assert has_clip, l

            x = get_remaining(len(self._clips), batch_size)
            for _ in range(x):
                self._clips.append((last_path, last_video, i))
                # if self.infer:
                #     video_path = l["video"]
                # else:
                #     video_path = os.path.join(video_dir, l["video"] + extension)
                video_path = l["video"]
                file_list_txt += f"{video_path} {cmp} {i * stride_dali} {end}\n"
                cmp += 1
            # print(file_list_txt)
            tf = tempfile.NamedTemporaryFile()
            tf.write(str.encode(file_list_txt))
            tf.flush()

            self.pipes = [
                self.video_pipe(
                    batch_size=self.batch_size,
                    sequence_length=self._clip_len,
                    stride_dali=stride_dali,
                    step=-1,
                    num_threads=8,
                    device_id=i,
                    file_list=tf.name,
                    shard_id=index,
                    num_shards=len(devices),
                )
                for index, i in enumerate(devices)
            ]

            for pipe in self.pipes:
                pipe.build()

            size = len(self._clips)

            super().__init__(self.pipes, output_map, size=size)

        def __next__(self):
            import cupy

            out = super().__next__()
            video_names = []
            starts = cupy.zeros(len(self.devices) * self.batch_size, np.int64)
            cmp = 0
            for j in range(len(out)):
                for i in range(out[j]["label"].shape[0]):
                    video_path, video_name, start = self._clips[out[j]["label"][i]]
                    video_names.append(video_path)
                    starts[cmp] = start
                    cmp += 1
            return {
                "video": video_names,
                "start": torch.as_tensor(starts),
                "frame": torch.cat(
                    ([data["data"].to(torch.device("cuda")) for data in out])
                ),
            }

        def delete(self):
            """Useful method to free memory used by gpu when the dataset is no longer needed."""
            for pipe in self.pipes:
                pipe.__del__()
                del pipe
            backend.ReleaseUnusedMemory()

        @dali_pipeline_def
        def video_pipe(
            self, file_list, sequence_length, stride_dali, step, shard_id, num_shards
        ):
            """Construct the pipeline to process a video. This pipeline process a clip with specified arguments such as stride,step and sequence length.
            The first step returns clip of frames with associated labels (index of the clip in the list of clips) and the index of the first frame.
            The second step is the cropping, mirroring (only if non eval) and normalizing the frames.

            Args:
                file_list (string): Path to the file with a list of <file label [start_frame [end_frame]]> values.
                sequence_length (int): Frames to load per sequence.
                stride_dali (int): Distance between consecutive frames in the sequence.
                step(int): Frame interval between each sequence.
                shard_id (int): Index of the shard to read.
                num_shards (int): Partitions the data into the specified number of parts.

            Returns:
                video (torch.tensor): The frames processed.
                label : the index of the clip in the list of clips.
            """
            video, label = fn.readers.video_resize(
                device="gpu",
                size=(self.TARGET_HEIGHT, self.TARGET_WIDTH),
                file_list=file_list,
                sequence_length=sequence_length,
                random_shuffle=False,
                shard_id=shard_id,
                num_shards=num_shards,
                image_type=types.RGB,
                file_list_include_preceding_frame=True,
                file_list_frame_num=True,
                stride=stride_dali,
                step=step,
                pad_sequences=True,
                skip_vfr_check=True,
            )

            video = fn.crop_mirror_normalize(
                video,
                dtype=types.FLOAT,
                output_layout="FCHW",
                crop=(self.crop_dim, self.crop_dim) if self.crop_dim != None else None,
                out_of_bounds_policy="trim_to_shape",
                mean=[self.IMAGENET_MEAN[i] * 255.0 for i in range(len(self.IMAGENET_MEAN))],
                std=[self.IMAGENET_STD[i] * 255.0 for i in range(len(self.IMAGENET_STD))],
            )

            return video, label

        def get_dims(video):
            print(video.shape)

K_V2 = torch.FloatTensor(
    [
        [
            -100,
            -98,
            -20,
            -40,
            -96,
            -5,
            -8,
            -93,
            -99,
            -31,
            -75,
            -10,
            -97,
            -75,
            -20,
            -84,
            -18,
        ],
        [
            -50,
            -49,
            -10,
            -20,
            -48,
            -3,
            -4,
            -46,
            -50,
            -15,
            -37,
            -5,
            -49,
            -38,
            -10,
            -42,
            -9,
        ],
        [50, 49, 60, 10, 48, 3, 4, 46, 50, 15, 37, 5, 49, 38, 10, 42, 9],
        [100, 98, 90, 20, 96, 5, 8, 93, 99, 31, 75, 10, 97, 75, 20, 84, 18],
    ]
)


class FeaturefromJson(Dataset):
    """Parent class that is used to modularise common operations between the classes that prepare features data.
    In particular, this class loads the input, creates dictionnaries of classes and has a method to process the annotations.

    Args:
        path (str|List(str)): Path of the input. Can be a json file, a features file or a list of json files (list only for training purposes).
        features_dir (str|List(str)): Path where the features are located. Must match the number of input paths.
        classes (path): Path of the file containing the classes. Only used if input is feature file, otherwise list of classes is in the json file.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
    """

    def __init__(self, path, features_dir, classes, framerate=2):
        self.classes = classes
        self.path = path
        self.framerate = framerate
        self.features_dir = features_dir

        self.is_json = True

        if isinstance(path, list):
            self.data_json = []
            self.classes = []
            for single_path in path:
                assert os.path.isfile(single_path)
                with open(single_path) as f:
                    tmp = json.load(f)
                    self.data_json.append(tmp)
                    for task_name, task_data in tmp["labels"].items():
                        self.classes.append(task_data.get("labels", {}))
            assert all(x == self.classes[0] for x in self.classes) == True
            self.classes = self.classes[0]

        else:
            self.features_dir = [features_dir]

            assert os.path.isfile(path)
            if path.endswith(".json"):
                with open(path) as f:
                    tmp = json.load(f)
                    self.data_json = [tmp]
                
                for task_name, task_data in tmp["labels"].items():
                    self.classes = task_data.get("labels", {})
            else:
                self.is_json = False
                self.data_json = [
                    {
                        "data": [
                            {
                                "inputs": [{
                                    "path": path
                                }],
                                "events": [],
                            }
                        ]
                    }
                ]
                assert isinstance(self.classes, list) or os.path.isfile(self.classes)

                from opensportslib.core.utils.config import load_text
                if not isinstance(self.classes, list):
                    self.classes = load_text(self.classes)

        self.num_classes = len(self.classes)
        print(self.num_classes)
        self.event_dictionary = {cls: i_cls for i_cls, cls in enumerate(self.classes)}
        self.inverse_event_dictionary = {
            i_cls: cls for i_cls, cls in enumerate(self.classes)
        }
        logging.info("Pre-compute clips")

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def annotation(self, annotation):
        """Process an annotation to derive the frame number, the class index and a boolean.
        Args:
            annotation (dict): It must contains the keys "gameTime" and "label".

        Returns:
            label (int): The index of the class.
            frame (int): The index of the frame.
            cont (bool): Whether to continue in the loop or not.
        """
        # time = annotation["gameTime"]
        event = annotation["label"]

        if "position_ms" in annotation.keys():
            frame = int(self.framerate * (int(annotation["position_ms"]) / 1000))
        else:
            time = annotation["gameTime"]

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            frame = self.framerate * (seconds + 60 * minutes)

        cont = False

        if event not in self.classes:
            cont = True
        else:
            label = self.classes.index(event)

        return label, frame, cont


class FeatureClipsfromJSON(FeaturefromJson):
    """Class that inherits from FeaturefromJson to prepare features data as clips of features.
    This class is used for the pooling methods.
    The class has 2 behaviours for processing the data depending if it is for training or testing purposes.

    Args:
        path (str|List(str)): Path of the input. Can be a json file, a features file or a list of json files (list only for training purposes).
        features_dir (str|List(str)): Path where the features are located. Must match the number of input paths.
        classes (path): Path of the file containing the classes. Only used if input is feature file, otherwise list of classes is in the json file.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        window_size (int): The length of the window that will be used for the length of the clips.
            Default: 15.
        train (bool): Whether we prepare data for training or for testing purposes.
            Default: True.
    """

    def __init__(
        self, path, features_dir, classes, framerate=2, window_size=15, train=True
    ):
        super().__init__(path, features_dir, classes, framerate)
        self.window_size_frame = window_size * framerate
        self.train = train
        self.features_clips = list()
        self.labels_clips = list()

        if self.train:
            for i, single_data_json in enumerate(self.data_json):
                if isinstance(path, list):
                    logging.info("Processing " + path[i])
                else:
                    logging.info("Processing " + path)
                # loop over videos
                for video in tqdm.tqdm(single_data_json["data"]):
                    # for video in tqdm(self.data_json["videos"]):
                    # Load features
                    features = np.load(
                        os.path.join(self.features_dir[i], video["inputs"][0]["path"])
                    )
                    features = features.reshape(-1, features.shape[-1])

                    # convert video features into clip features
                    features = feats2clip(
                        torch.from_numpy(features),
                        stride=self.window_size_frame,
                        clip_length=self.window_size_frame,
                    )

                    # Load labels
                    labels = np.zeros((features.shape[0], self.num_classes + 1))
                    labels[:, 0] = 1  # those are BG classes

                    # loop annotation for that video
                    for annotation in video.get("events", []):

                        label, frame, cont = self.annotation(annotation)

                        if cont:
                            continue

                        # if label outside temporal of view
                        if frame // self.window_size_frame >= labels.shape[0]:
                            continue

                        labels[frame // self.window_size_frame][0] = 0  # not BG anymore
                        labels[frame // self.window_size_frame][
                            label + 1
                        ] = 1  # that's my class

                    self.features_clips.append(features)
                    self.labels_clips.append(labels)

            self.features_clips = np.concatenate(self.features_clips)
            self.labels_clips = np.concatenate(self.labels_clips)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            If training:
                clip_feat (np.array): clip of features.
                clip_labels (np.array): clip of labels.
            If testing:
                Name of the feature file.
                features (np.array): clip of features.
                labels (np.array): clip of labels.
        """
        if self.train:
            return self.features_clips[index, :, :], self.labels_clips[index, :]
        else:
            video = self.data_json[0]["data"][index]
            video_path = video["inputs"][0]["path"]
            # Load features
            if self.is_json:
                features = np.load(os.path.join(self.features_dir[0], video_path))
            else:
                features = np.load(os.path.join(video_path))
            features = features.reshape(-1, features.shape[-1])

            # Load labels
            labels = np.zeros((features.shape[0], self.num_classes))

            if "events" in video.keys():
                for annotation in video.get("events", []):

                    label, frame, cont = self.annotation(annotation)

                    if cont:
                        continue

                    frame = min(frame, features.shape[0] - 1)
                    labels[frame][label] = 1

            features = feats2clip(
                torch.from_numpy(features),
                stride=1,
                off=int(self.window_size_frame / 2),
                clip_length=self.window_size_frame,
            )

            return video_path, features, labels

    def __len__(self):
        if self.train:
            return len(self.features_clips)
        else:
            return len(self.data_json[0]["data"])


class FeatureClipChunksfromJson(FeaturefromJson):
    """Class that inherits from FeaturefromJson to prepare features data as clips of features based on a chunk approach.
    This class is used for the CALF method.
    The class has 2 behaviours for processing the data depending if it is for training or testing purposes.

    Args:
        path (str|List(str)): Path of the input. Can be a json file, a features file or a list of json files (list only for training purposes).
        features_dir (str|List(str)): Path where the features are located. Must match the number of input paths.
        classes (path): Path of the file containing the classes. Only used if input is feature file, otherwise list of classes is in the json file.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        chunk_size (int): Size of the chunk.
            Default: 240.
        receptive_field (int):  temporal receptive field of x seconds on both sides of the central frame in the temporal dimension of the 3D convolutions
            Default: 80.
        chunks_per_epoch (int): Number of chunks per epoch.
            Default: 6000.
        gpu (bool): Whether gpu is used or not.
            Default: True.
        train (bool): Whether we prepare data for training or for testing purposes.
            Default: True.
    """

    def __init__(
        self,
        path,
        features_dir,
        classes,
        framerate=2,
        chunk_size=240,
        receptive_field=80,
        chunks_per_epoch=6000,
        gpu=True,
        train=True,
    ):
        super().__init__(path, features_dir, classes, framerate)
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.chunks_per_epoch = chunks_per_epoch
        self.gpu = gpu
        self.train = train
        global K_V2
        if self.gpu >= 0:
            K_V2 = K_V2.cuda()
        self.K_parameters = K_V2 * framerate
        self.num_detections = 15

        if self.train:
            self.features_clips = list()
            self.labels_clips = list()
            self.anchors_clips = list()
            for i in np.arange(self.num_classes + 1):
                self.anchors_clips.append(list())

            game_counter = 0
            # loop over videos
            for i, single_data_json in enumerate(self.data_json):
                if isinstance(path, list):
                    logging.info("Processing " + path[i])
                else:
                    logging.info("Processing " + path)
                # loop over videos
                for video in tqdm.tqdm(single_data_json["data"]):
                    # for video in tqdm(self.data_json["videos"]):
                    # Load features
                    features = np.load(
                        os.path.join(self.features_dir[i], video["inputs"][0]["path"])
                    )

                    # Load labels
                    labels = np.zeros((features.shape[0], self.num_classes))

                    # loop annotation for that video
                    for annotation in video.get("events", []):

                        label, frame, cont = self.annotation(annotation)

                        if cont:
                            continue

                        frame = min(frame, features.shape[0] - 1)
                        labels[frame][label] = 1

                    shift_half = oneHotToShifts(labels, self.K_parameters.cpu().numpy())

                    anchors_half = getChunks_anchors(
                        shift_half,
                        game_counter,
                        self.K_parameters.cpu().numpy(),
                        self.chunk_size,
                        self.receptive_field,
                    )

                    game_counter = game_counter + 1

                    self.features_clips.append(features)
                    self.labels_clips.append(shift_half)

                    for anchor in anchors_half:
                        self.anchors_clips[anchor[2]].append(anchor)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            If training:
                clip_feat (np.array): clip of features.
                clip_labels (np.array): clip of labels.
                clip_targets (np.array): clip of targets.
            If testing:
                features (np.array): clip of features.
                labels (np.array): clip of labels.
        """
        if self.train:
            # Retrieve the game index and the anchor
            class_selection = random.randint(0, self.num_classes)
            event_selection = random.randint(
                0, len(self.anchors_clips[class_selection]) - 1
            )
            game_index = self.anchors_clips[class_selection][event_selection][0]
            anchor = self.anchors_clips[class_selection][event_selection][1]

            # Compute the shift for event chunks
            if class_selection < self.num_classes:
                shift = np.random.randint(
                    -self.chunk_size + self.receptive_field, -self.receptive_field
                )
                start = anchor + shift
            # Compute the shift for non-event chunks
            else:
                start = random.randint(anchor[0], anchor[1] - self.chunk_size)
            if start < 0:
                start = 0
            if start + self.chunk_size >= self.features_clips[game_index].shape[0]:
                start = self.features_clips[game_index].shape[0] - self.chunk_size - 1

            # Extract the clips
            clip_feat = self.features_clips[game_index][start : start + self.chunk_size]
            clip_labels = self.labels_clips[game_index][start : start + self.chunk_size]

            # Put loss to zero outside receptive field
            clip_labels[0 : int(np.ceil(self.receptive_field / 2)), :] = -1
            clip_labels[-int(np.ceil(self.receptive_field / 2)) :, :] = -1

            # Get the spotting target
            clip_targets = getTimestampTargets(
                np.array([clip_labels]), self.num_detections
            )[0]

            return (
                torch.from_numpy(clip_feat),
                torch.from_numpy(clip_labels),
                torch.from_numpy(clip_targets),
            )
        else:
            video = self.data_json[0]["data"][index]
            video_path = video["inputs"][0]["path"]
            # Load features
            if self.is_json:
                features = np.load(os.path.join(self.features_dir[0], video_path))
            else:
                features = np.load(os.path.join(video_path))

            # Load labels
            labels = np.zeros((features.shape[0], self.num_classes))

            if "annotations" in video.keys():
                for annotation in video.get("events",[]):

                    label, frame, cont = self.annotation(annotation)
                    if cont:
                        continue

                    value = 1
                    if "visibility" in annotation.keys():
                        if annotation["visibility"] == "not shown":
                            value = -1

                    frame = min(frame, features.shape[0] - 1)
                    labels[frame][label] = value

            features = feats2clip(
                torch.from_numpy(features),
                stride=self.chunk_size - self.receptive_field,
                clip_length=self.chunk_size,
                modif_last_index=True,
            )

            return features, torch.from_numpy(labels)

    def __len__(self):
        if self.train:
            return self.chunks_per_epoch
        else:
            return len(self.data_json[0]["data"])



class SoccerNetGame(Dataset):
    """Parent class that is used to modularise common operations between the classes that prepare features of a single game of the soccernet dataset.

    Args:
        path (string): Path of the game.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
    """

    def __init__(self, path, features="ResNET_PCA512.npy", version=1, framerate=2):
        self.path = path
        self.framerate = framerate
        self.version = version
        self.features = features
        self.listGames = [self.path]
        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17

    def __getitem__(self, index):
        pass

    def __len__(self):
        return 2

    def load_features(self):
        self.feat_half1 = np.load(os.path.join(self.path, "1_" + self.features))
        self.feat_half2 = np.load(os.path.join(self.path, "2_" + self.features))


class SoccerNetGameClips(SoccerNetGame):
    """Class that inherits from SoccerNetGame to prepare features data of a single game as clips of features.
    This class is used for the pooling methods.

    Args:
        path (string): Path of the game.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        window_size (int): The length of the window that will be used for the length of the clips.
            Default: 15.
    """

    def __init__(
        self, path, features="ResNET_PCA512.npy", version=1, framerate=2, window_size=15
    ):

        super().__init__(path, features, version, framerate)
        self.window_size_frame = window_size * self.framerate

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            path: path of the game.
            clip_feat_1 (np.array): clip of features for the first half.
            clip_feat_2 (np.array): clip of features for the second half.
            empty list
            empty list
        """
        # Load features
        feat_half1, feat_half2 = self.load_features()

        feat_half1 = feats2clip(
            torch.from_numpy(feat_half1),
            stride=1,
            off=int(self.window_size_frame / 2),
            clip_length=self.window_size_frame,
        )

        feat_half2 = feats2clip(
            torch.from_numpy(feat_half2),
            stride=1,
            off=int(self.window_size_frame / 2),
            clip_length=self.window_size_frame,
        )

        return self.path, feat_half1, feat_half2, [], []

    def load_features(self):
        super().load_features()
        self.feat_half1 = self.feat_half1.reshape(-1, self.feat_half1.shape[-1])
        self.feat_half2 = self.feat_half2.reshape(-1, self.feat_half2.shape[-1])
        return self.feat_half1, self.feat_half2


class SoccerNetGameClipsChunks(SoccerNetGame):
    """Class that inherits from SoccerNetGame to prepare features data of a single game as clips of features based on a chunk approach.
    This class is used for the CALF method.

    Args:
        path (string): Path of the game.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        chunk_size (int): Size of the chunk.
            Default: 240.
        receptive_field (int):  temporal receptive field of x seconds on both sides of the central frame in the temporal dimension of the 3D convolutions
            Default: 80.
    """

    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        framerate=2,
        chunk_size=240,
        receptive_field=80,
    ):
        super().__init__(path, features, 2, framerate)
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat_1 (np.array): clip of features for the first half.
            clip_feat_2 (np.array): clip of features for the second half.
            clip_label_1 (np.array): clip of labels for the first half.
            clip_label_2 (np.array): clip of labels for the second half.
        """
        # Load features
        feat_half1, feat_half2 = self.load_features()

        # Load labels
        label_half1, label_half2 = self.load_labels(feat_half1, feat_half2)

        feat_half1 = feats2clip(
            torch.from_numpy(feat_half1),
            stride=self.chunk_size - self.receptive_field,
            clip_length=self.chunk_size,
            modif_last_index=True,
        )

        feat_half2 = feats2clip(
            torch.from_numpy(feat_half2),
            stride=self.chunk_size - self.receptive_field,
            clip_length=self.chunk_size,
            modif_last_index=True,
        )

        return (
            feat_half1,
            feat_half2,
            torch.from_numpy(label_half1),
            torch.from_numpy(label_half2),
        )

    def load_features(
        self,
    ):
        super().load_features()
        return self.feat_half1, self.feat_half2

    def load_labels(self, feat_half1, feat_half2):
        self.label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        self.label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))
        return self.label_half1, self.label_half2


class SoccerNet(Dataset):
    """Parent class that is used to modularise common operations between the classes that prepare features of a split of a soccernet dataset.

    Args:
        path (string): Data root of the feature files.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        split (List(string)): List of splits.
            Default: ["train"].
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
    """

    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        split=["train"],
        version=1,
        framerate=2,
    ):
        self.path = path
        # split=["train"] if clips else ["test"]
        self.listGames = getListGames(split)
        self.features = features
        self.framerate = framerate
        self.split = split
        self.version = version

        if version == 1:
            self.dict_event = EVENT_DICTIONARY_V1
            self.num_classes = 3
            self.labels = "Labels.json"
        elif version == 2:
            self.dict_event = EVENT_DICTIONARY_V2
            self.num_classes = 17
            self.labels = "Labels-v2.json"

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.listGames)

    def load_features(self, index=0, game=""):
        """Load features from files.

        Args:
            index (int): Used for testing purpose to retrieve the game based on the index.
                Default: 0.
            game (string): Used for training purpose, this is the name of the game.
                Default: "".
        """
        if self.train:
            game = game
        else:
            game = self.listGames[index].replace(" ", "_")
        self.feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
        self.feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))

    def load_labels(self, feat_half1, feat_half2, number_classes):
        # Load labels
        self.label_half1 = np.zeros((feat_half1.shape[0], number_classes))
        self.label_half2 = np.zeros((feat_half2.shape[0], number_classes))

    def annotation(self, annotation):
        """Process an annotation to derive the frame number, the class index and a boolean.
        Args:
            annotation (dict): It must contains the keys "gameTime" and "label".

        Returns:
            label (int): The index of the class.
            half (int): Which half of the game.
            frame (int): The index of the frame.
            cont (bool): Whether to continue in the loop or not.
        """
        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        frame = self.framerate * (seconds + 60 * minutes)

        cont = False
        if self.version == 1:
            if "card" in event:
                label = 0
            elif "subs" in event:
                label = 1
            elif "soccer" in event:
                label = 2
            else:
                cont = True
        elif self.version == 2:
            if event not in self.dict_event:
                cont = True
            else:
                label = self.dict_event[event]

        return label, half, frame, cont


class SoccerNetClips(SoccerNet):
    """Class that inherits from SoccerNetGame to prepare features data of a split as clips of features.
    This class is used for the pooling methods.

    Args:
        path (string): Data root of the feature files.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        split (List(string)): List of splits.
            Default: ["train"].
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        window_size (int): The length of the window that will be used for the length of the clips.
            Default: 15.
        train (bool): Whether training or testing.
            Default: True.
    """

    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        split=["train"],
        version=1,
        framerate=2,
        window_size=15,
        train=True,
    ):

        super().__init__(path, features, split, version, framerate)
        self.window_size_frame = window_size * self.framerate
        self.train = train

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)

        # if train :
        #     downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False,randomized=True)
        # else :
        #     for s in split:
        #         if s == "challenge":
        #             downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)
        #         else:
        #             downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False,randomized=True)

        if train:
            logging.info("Pre-compute clips")

            self.game_feats = list()
            self.game_labels = list()

            for game in tqdm.tqdm(self.listGames):
                game = game.replace(" ", "_")
                # Load features
                feat_half1, feat_half2 = self.load_features(game=game)

                feat_half1 = feats2clip(
                    torch.from_numpy(feat_half1),
                    stride=self.window_size_frame,
                    clip_length=self.window_size_frame,
                )
                feat_half2 = feats2clip(
                    torch.from_numpy(feat_half2),
                    stride=self.window_size_frame,
                    clip_length=self.window_size_frame,
                )

                # Load labels
                labels = json.load(open(os.path.join(self.path, game, self.labels)))

                label_half1, label_half2 = self.load_labels(feat_half1, feat_half2)

                for annotation in labels["annotations"]:

                    label, half, frame, cont = self.annotation(annotation)
                    if cont:
                        continue

                    # if label outside temporal of view
                    if (
                        half == 1
                        and frame // self.window_size_frame >= label_half1.shape[0]
                    ):
                        continue
                    if (
                        half == 2
                        and frame // self.window_size_frame >= label_half2.shape[0]
                    ):
                        continue

                    if half == 1:
                        label_half1[frame // self.window_size_frame][
                            0
                        ] = 0  # not BG anymore
                        label_half1[frame // self.window_size_frame][
                            label + 1
                        ] = 1  # that's my class

                    if half == 2:
                        label_half2[frame // self.window_size_frame][
                            0
                        ] = 0  # not BG anymore
                        label_half2[frame // self.window_size_frame][
                            label + 1
                        ] = 1  # that's my class
                self.game_feats.append(feat_half1)
                self.game_feats.append(feat_half2)
                self.game_labels.append(label_half1)
                self.game_labels.append(label_half2)

            self.game_feats = np.concatenate(self.game_feats)
            self.game_labels = np.concatenate(self.game_labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            if train :
                clip_feat (np.array): clip of features.
                clip_labels (np.array): clip of labels for the segmentation.
            if testing:
                name of the game.
                clip_feat_1 (np.array): clip of features for the first half.
                clip_feat_2 (np.array): clip of features for the second half.
                clip_label_1 (np.array): clip of labels for the first half.
                clip_label_2 (np.array): clip of labels for the second half.
        """
        if self.train:
            return self.game_feats[index, :, :], self.game_labels[index, :]
        else:
            # Load features
            feat_half1, feat_half2 = self.load_features(index=index)

            # Load labels
            label_half1, label_half2 = self.load_labels(feat_half1, feat_half2)

            # check if annotation exists
            if os.path.exists(
                os.path.join(self.path, self.listGames[index], self.labels)
            ):
                labels = json.load(
                    open(os.path.join(self.path, self.listGames[index], self.labels))
                )

                for annotation in labels["annotations"]:

                    label, half, frame, cont = self.annotation(annotation)
                    if cont:
                        continue

                    value = 1
                    if "visibility" in annotation.keys():
                        if annotation["visibility"] == "not shown":
                            value = -1

                    if half == 1:
                        frame = min(frame, feat_half1.shape[0] - 1)
                        label_half1[frame][label] = value

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0] - 1)
                        label_half2[frame][label] = value

            feat_half1 = feats2clip(
                torch.from_numpy(feat_half1),
                stride=1,
                off=int(self.window_size_frame / 2),
                clip_length=self.window_size_frame,
            )

            feat_half2 = feats2clip(
                torch.from_numpy(feat_half2),
                stride=1,
                off=int(self.window_size_frame / 2),
                clip_length=self.window_size_frame,
            )

            return (
                self.listGames[index],
                feat_half1,
                feat_half2,
                label_half1,
                label_half2,
            )

    def __len__(self):
        if self.train:
            return len(self.game_feats)
        else:
            return super().__len__()

    def load_features(self, index=0, game=""):
        super().load_features(index, game)
        self.feat_half1 = self.feat_half1.reshape(-1, self.feat_half1.shape[-1])
        self.feat_half2 = self.feat_half2.reshape(-1, self.feat_half2.shape[-1])
        return self.feat_half1, self.feat_half2

    def load_labels(self, feat_half1, feat_half2):
        super().load_labels(
            feat_half1,
            feat_half2,
            self.num_classes + 1 if self.train else self.num_classes,
        )
        if self.train:
            self.label_half1[:, 0] = 1  # those are BG classes
            self.label_half2[:, 0] = 1  # those are BG classes
        return self.label_half1, self.label_half2


class SoccerNetClipsChunks(SoccerNet):
    """Class that inherits from SoccerNetGame to prepare features data of a split as clips of features with a chunk approach.
    This class is used for the CALF method.

    Args:
        path (str): Data root of the feature files.
        features (str): Name of the features.
            Default: ResNET_PCA512.npy.
        split (string): split.
            Default: "train".
        version (int): Version of the dataset.
            Default: 1.
        framerate (int): The framerate at which the features have been extracted.
            Default: 2.
        chunk_size (int): Size of the chunk.
            Default: 240.
        receptive_field (int):  temporal receptive field of x seconds on both sides of the central frame in the temporal dimension of the 3D convolutions
            Default: 80.
        chunks_per_epoch (int): Number of chunks per epoch.
            Default: 6000.
        gpu (bool): Whether gpu is used or not.
            Default: True.
        train (bool): Whether we prepare data for training or for testing purposes.
            Default: True.
    """

    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        split="train",
        framerate=2,
        chunk_size=240,
        receptive_field=80,
        chunks_per_epoch=6000,
        gpu=True,
        train=True,
    ):
        super().__init__(path, features, split, 2, framerate)
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.chunks_per_epoch = chunks_per_epoch
        self.gpu = gpu
        self.train = train
        global K_V2
        if self.gpu >= 0:
            K_V2 = K_V2.cuda()
        self.K_parameters = K_V2 * framerate
        self.num_detections = 15

        logging.info("Checking/Download features and labels locally")
        downloader = SoccerNetDownloader(path)

        # if train:
        #     downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=split, verbose=False)
        # else:
        #     for s in split:
        #         if s == "challenge":
        #             downloader.downloadGames(files=[f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)
        #         else:
        #             downloader.downloadGames(files=[self.labels, f"1_{self.features}", f"2_{self.features}"], split=[s], verbose=False)

        if train:
            logging.info("Pre-compute clips")

            self.game_feats = list()
            self.game_labels = list()
            self.game_anchors = list()
            for i in np.arange(self.num_classes + 1):
                self.game_anchors.append(list())

            game_counter = 0
            for game in tqdm.tqdm(self.listGames):
                game = game.replace(" ", "_")
                # Load features
                feat_half1, feat_half2 = self.load_features(game=game)
                # Load labels
                labels = json.load(open(os.path.join(self.path, game, self.labels)))

                label_half1, label_half2 = self.load_labels(feat_half1, feat_half2)

                for annotation in labels["annotations"]:

                    label, half, frame, cont = self.annotation(annotation)
                    if cont:
                        continue
                    if half == 1:
                        frame = min(frame, feat_half1.shape[0] - 1)
                        label_half1[frame][label] = 1

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0] - 1)
                        label_half2[frame][label] = 1

                shift_half1 = oneHotToShifts(
                    label_half1, self.K_parameters.cpu().numpy()
                )
                shift_half2 = oneHotToShifts(
                    label_half2, self.K_parameters.cpu().numpy()
                )

                anchors_half1 = getChunks_anchors(
                    shift_half1,
                    game_counter,
                    self.K_parameters.cpu().numpy(),
                    self.chunk_size,
                    self.receptive_field,
                )

                game_counter = game_counter + 1

                anchors_half2 = getChunks_anchors(
                    shift_half2,
                    game_counter,
                    self.K_parameters.cpu().numpy(),
                    self.chunk_size,
                    self.receptive_field,
                )

                game_counter = game_counter + 1
                self.game_feats.append(feat_half1)
                self.game_feats.append(feat_half2)
                self.game_labels.append(shift_half1)
                self.game_labels.append(shift_half2)

                for anchor in anchors_half1:
                    self.game_anchors[anchor[2]].append(anchor)
                for anchor in anchors_half2:
                    self.game_anchors[anchor[2]].append(anchor)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            If training:
                clip_feat (np.array): clip of features.
                clip_labels (np.array): clip of labels.
                clip_targets (np.array): clip of targets.
            If testing:
                clip_feat_1 (np.array): clip of features for the first half.
                clip_feat_2 (np.array): clip of features for the second half.
                clip_labels_1 (np.array): clip of labels for the first half.
                clip_labels_2 (np.array): clip of labels for the second half.

        """
        if self.train:
            # Retrieve the game index and the anchor
            class_selection = random.randint(0, self.num_classes)
            event_selection = random.randint(
                0, len(self.game_anchors[class_selection]) - 1
            )
            game_index = self.game_anchors[class_selection][event_selection][0]
            anchor = self.game_anchors[class_selection][event_selection][1]

            # Compute the shift for event chunks
            if class_selection < self.num_classes:
                shift = np.random.randint(
                    -self.chunk_size + self.receptive_field, -self.receptive_field
                )
                start = anchor + shift
            # Compute the shift for non-event chunks
            else:
                start = random.randint(anchor[0], anchor[1] - self.chunk_size)
            if start < 0:
                start = 0
            if start + self.chunk_size >= self.game_feats[game_index].shape[0]:
                start = self.game_feats[game_index].shape[0] - self.chunk_size - 1

            # Extract the clips
            clip_feat = self.game_feats[game_index][start : start + self.chunk_size]
            clip_labels = self.game_labels[game_index][start : start + self.chunk_size]

            # Put loss to zero outside receptive field
            clip_labels[0 : int(np.ceil(self.receptive_field / 2)), :] = -1
            clip_labels[-int(np.ceil(self.receptive_field / 2)) :, :] = -1

            # Get the spotting target
            clip_targets = getTimestampTargets(
                np.array([clip_labels]), self.num_detections
            )[0]

            return (
                torch.from_numpy(clip_feat),
                torch.from_numpy(clip_labels),
                torch.from_numpy(clip_targets),
            )
        else:
            # Load features
            feat_half1, feat_half2 = self.load_features(index=index)

            # Load labels
            label_half1, label_half2 = self.load_labels(feat_half1, feat_half2)

            # check if annotation exists
            if os.path.exists(
                os.path.join(
                    self.path, self.listGames[index].replace(" ", "_"), self.labels
                )
            ):
                labels = json.load(
                    open(
                        os.path.join(
                            self.path,
                            self.listGames[index].replace(" ", "_"),
                            self.labels,
                        )
                    )
                )

                for annotation in labels["annotations"]:

                    label, half, frame, cont = self.annotation(annotation)
                    if cont:
                        continue

                    value = 1
                    if "visibility" in annotation.keys():
                        if annotation["visibility"] == "not shown":
                            value = -1

                    if half == 1:
                        frame = min(frame, feat_half1.shape[0] - 1)
                        label_half1[frame][label] = value

                    if half == 2:
                        frame = min(frame, feat_half2.shape[0] - 1)
                        label_half2[frame][label] = value

            feat_half1 = feats2clip(
                torch.from_numpy(feat_half1),
                stride=self.chunk_size - self.receptive_field,
                clip_length=self.chunk_size,
                modif_last_index=True,
            )

            feat_half2 = feats2clip(
                torch.from_numpy(feat_half2),
                stride=self.chunk_size - self.receptive_field,
                clip_length=self.chunk_size,
                modif_last_index=True,
            )

            return (
                feat_half1,
                feat_half2,
                torch.from_numpy(label_half1),
                torch.from_numpy(label_half2),
            )

    def __len__(self):
        if self.train:
            return self.chunks_per_epoch
        else:
            return super().__len__()

    def load_features(self, index=0, game=""):
        super().load_features(index, game)
        return self.feat_half1, self.feat_half2

    def load_labels(self, feat_half1, feat_half2):
        super().load_labels(feat_half1, feat_half2, self.num_classes)
        return self.label_half1, self.label_half2
    
if __name__ == "__main__":
    LocalizationDataset(config="/home/vorajv/opensportslib-ml/opensportslib/opensportslib/config/localization.yaml")
