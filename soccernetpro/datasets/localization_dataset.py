import os
import torch
import random
from torch.utils.data import Dataset
from nvidia.dali import pipeline_def, backend
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import tempfile
import cupy
import copy
import math
import numpy as np
from soccernetpro.core.utils.default_args import get_default_args_dataset
from soccernetpro.core.utils.load_annotations import get_repartition_gpu


class LocalizationDataset(Dataset):
    def __init__(self, config, annotations_path=None, processor=None, split="train"):
        self.config = config
        self.split = split
        self.config.TRAIN.repartitions = get_repartition_gpu()
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
        print(cfg)
        if cfg.type == "VideoGameWithDali":
            loader_batch_size = cfg.dataloader.batch_size // default_args["acc_grad_iter"]
            dataset_len = self.config.DATA.epoch_num_frames // self.config.DATA.clip_len
            dataset = DaliDataSet(
                default_args["num_epochs"],
                loader_batch_size,
                cfg.output_map,
                (
                    default_args["repartitions"][0]
                    if default_args["train"]
                    else default_args["repartitions"][1]
                ),
                default_args["classes"],
                cfg.path,
                self.config.DATA.modality,
                self.config.DATA.clip_len,
                dataset_len if default_args["train"] else dataset_len // 4,
                cfg.video_path,
                self.config.DATA.input_fps,
                self.config.DATA.extract_fps,
                self.config.DATA.IMAGENET_MEAN,
                self.config.DATA.IMAGENET_STD,
                self.config.DATA.TARGET_HEIGHT,
                self.config.DATA.TARGET_WIDTH,
                is_eval=False if default_args["train"] else True,
                crop_dim=self.config.DATA.crop_dim,
                dilate_len=self.config.DATA.dilate_len,
                mixup=self.config.DATA.mixup,
            )
        elif cfg.type == "VideoGameWithDaliVideo":
            dataset = DaliDataSetVideo(
                cfg.dataloader.batch_size,
                cfg.output_map,
                default_args["repartitions"][1],
                default_args["classes"],
                cfg.path,
                self.config.DATA.modality,
                self.config.DATA.clip_len,
                cfg.video_path,
                self.config.DATA.input_fps,
                self.config.DATA.extract_fps,
                self.config.DATA.IMAGENET_MEAN,
                self.config.DATA.IMAGENET_STD,
                self.config.DATA.TARGET_HEIGHT,
                self.config.DATA.TARGET_WIDTH,
                crop_dim=self.config.DATA.crop_dim,
                overlap_len=cfg.overlap_len,

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
                cfg.prefetch_factor if "prefetch_factor" in cfg.keys() else None
            ),
            worker_init_fn=worker_init_fn
        )
        return dataloader


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
        import random
        from soccernetpro.core.utils.load_annotations import annotationstoe2eformat
        from soccernetpro.core.utils.video_processing import distribute_elements, _get_deferred_rgb_transform, get_stride

        self._src_file = label_file
        # self._labels = load_json(label_file)
        self._labels = annotationstoe2eformat(
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
            print(video_path)
            # video_path = os.path.join(video_dir, video["video"] + extension)
            for _ in range(nb_clips_per_video):
                #print(video["num_frames"], (clip_len + 1))
                random_start = random.randint(1, abs(video["num_frames"] - (clip_len + 1)))
                file_list_txt += f"{video_path} {index} {random_start * self._stride} {(random_start+clip_len) * self._stride}\n"

        tf = tempfile.NamedTemporaryFile()
        tf.write(str.encode(file_list_txt))
        tf.flush()

        self.pipes = [
            self.video_pipe(
                batch_size=self.batch_size_per_pipe[index],
                sequence_length=clip_len,
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

        super().__init__(self.pipes, output_map, size=self.nb_videos)

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
        batch_labels = batch["label"]
        batch_images = batch["data"]
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

    @pipeline_def
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
        label = fn.python_function(
            label, frame_num, function=self.edit_labels, device="gpu"
        )
        return video, label

    def edit_labels(self, label, frame_num):
        """Construct a list having the same length as the number of frames. The elements of the list are the indexes (starting at 1) of the class where an event occurs, 0 otherwise.

        Args:
            label :index of the video to get the metadata.
            frame_num :index of start frame.

        Returns:
            labels (cupy.array): the list of labels (corresponding to events) corresponding with the extracted frames.
        """
        video_meta = self._labels[label.item()]
        base_idx = frame_num.item() // self._stride
        labels = cupy.zeros(self.clip_len, np.int64)

        for event in video_meta["events"]:
            event_frame = event["frame"]
            # Index of event in label array
            label_idx = (event_frame - base_idx) // 1
            if (
                label_idx >= self.dilate_len
                and label_idx < self.clip_len + self.dilate_len
            ):
                label = self._class_dict[event["label"]]
                for i in range(
                    max(0, label_idx - self.dilate_len),
                    min(self.clip_len, label_idx + self.dilate_len + 1),
                ):
                    labels[i] = label
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
        import random
        from soccernetpro.core.utils.load_annotations import annotationstoe2eformat, construct_labels
        from soccernetpro.core.utils.video_processing import distribute_elements, _get_deferred_rgb_transform, get_stride, get_remaining
        self._src_file = label_file
        # self.infer = False
        if label_file.endswith(".json"):
            self._labels = annotationstoe2eformat(
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
                sequence_length=clip_len,
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

    @pipeline_def
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


if __name__ == "__main__":
    LocalizationDataset(config="/home/vorajv/soccernetpro-ml/soccernetpro/soccernetpro/config/localization.yaml")
