import torch
import numpy as np
import math

try:
    import decord
    from decord import cpu
    USE_DECORD = True
except:
    import av
    USE_DECORD = False

def read_video(video_path):
    """Read video frames into list of HxWxC uint8 arrays for VideoMAE."""
    if USE_DECORD:
        vr = decord.VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(range(len(vr)))  # (T, H, W, C)
        frames = frames.asnumpy().astype(np.uint8)  # ensure uint8 for VideoMAE
        frames_list = [frame for frame in frames]  # list of T frames
    else:
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24").astype(np.uint8))
        return frames
    return frames_list


def resample_video_idx(num_frames, original_fps, new_fps):
    """Return frame indices to match new fps"""
    step = float(original_fps) / new_fps
    if step.is_integer():
        step = int(step)
        return slice(None, None, step)
    idxs = torch.arange(num_frames, dtype=torch.float32) * step
    idxs = idxs.floor().to(torch.int64)
    return idxs

def process_frames(frames, target_num_frames, input_fps, target_fps, start_frame=0, end_frame=-1, uniform_sample=False):
    """
    frames: list of np arrays (H, W, C)
    target_num_frames: int
    Returns: list of np arrays (H, W, C) ready for processor
    """

    
    target_fps = target_fps or input_fps
    num_frames = len(frames)
    end_frame = end_frame if end_frame != -1 else num_frames
    duration = num_frames / input_fps

    # unfiorm sampling throughout the video
    if uniform_sample:
        # Too short → resample with new fps
        if num_frames < target_num_frames:
            new_fps = np.ceil(target_num_frames / duration)
            idxs = resample_video_idx(target_num_frames, input_fps, new_fps)
            idxs = np.clip(idxs, 0, num_frames - 1)
            frames = [frames[i] for i in idxs]

        # Too long → uniform sampling
        elif num_frames > target_num_frames:
            idxs = np.linspace(0, num_frames - 1, target_num_frames).astype(int)
            frames = [frames[i] for i in idxs]

        # Pad if still short
        if len(frames) < target_num_frames:
            pad = target_num_frames - len(frames)
            frames.extend([frames[-1]] * pad)
    else:
        window = frames[start_frame:end_frame]          # 24 frames
        assert len(window) > 0, "Empty temporal window"
        factor = input_fps / target_fps
        idxs = [int(i * factor) for i in range(target_num_frames)]
        idxs = [min(i, len(window) - 1) for i in idxs]
        frames = [window[i] for i in idxs]
        assert len(frames) == target_num_frames, f"Expected {target_num_frames} frames, got {len(frames)}"

    return frames


def get_stride(src_fps, sample_fps):
    """Get stride to apply based on the input and output fps.

    Args:
        src_fps (int): The input fps of the video.
        sample_fps (int): The output fps.
    Returns:
        stride (int): The stride to apply.
    """
    if sample_fps <= 0:
        stride = 1
    else:
        stride = int(src_fps / sample_fps)
    return stride


def read_fps(fps, sample_fps):
    """Computes the exact output fps based on input fps and wanted output fps.
    Example: if input fps is 25 and wanted output fps is 2, the exact output fps is 2.0833333333333335.

    Args:
        fps (int): The input fps.
        sample_fps (int): The wanted output fps.

    Returns:
        est_out_fps (float): The exact output fps.

    """
    stride = get_stride(fps, sample_fps)
    est_out_fps = fps / stride
    return est_out_fps

def get_num_frames(num_frames, fps, sample_fps):
    """Compute the number of frames of a video after fps changes.

    Args:
        num_frames (int): Number of frames in the base video.
        fps (int): The input fps.
        sample_fps (int): The output fps.

    Returns:
        (int): The number of frames with the output fps.
    """
    return math.ceil(num_frames / get_stride(fps, sample_fps))


def distribute_elements(batch_size, len_devices):
    """Return a list containing the distribution of the batch along the devices.

    Args:
        batch_size (int).
        len_device (int).

    Returns:
        distribution (list): For example if batch size is 8 and there is 4 gpus, the distribution is [2,2,2,2], meaning that each gpu will process 2 samples.
    """
    quotient, remainder = divmod(batch_size, len_devices)
    distribution = [quotient] * len_devices
    if remainder > 0:
        for i in range(len(distribution)):
            distribution[i] += 1

    return distribution

def _load_frame_deferred(gpu_transform, batch, device):
    """Load frames on the device and applying some transforms.

    Args:
        gpu_transform : Transform to apply to the frames.
        batch : Batch containing the frames and possibly some other datas as
        "mix_weight" and "mix_frame" is mixup is applied while processing videos.
        device : The device on which we load the data.

    Returns:
        frame (torch.tensor).
    """
    frame = batch["frame"].to(device)
    with torch.no_grad():
        for i in range(frame.shape[0]):
            frame[i] = gpu_transform(frame[i])

        if "mix_weight" in batch:
            weight = batch["mix_weight"].to(device)
            frame *= weight[:, None, None, None, None]

            frame_mix = batch["mix_frame"]
            for i in range(frame.shape[0]):
                frame[i] += (1.0 - weight[i]) * gpu_transform(frame_mix[i].to(device))
    return frame


def _get_deferred_rgb_transform(IMAGENET_MEAN, IMAGENET_STD):
    import torchvision.transforms as T
    import torch.nn as nn
    img_transforms = [
        # Jittering separately is faster (low variance)
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(hue=0.2)]), p=0.25
        ),
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(saturation=(0.7, 1.2))]), p=0.25
        ),
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(brightness=(0.7, 1.2))]), p=0.25
        ),
        T.RandomApply(
            nn.ModuleList([T.ColorJitter(contrast=(0.7, 1.2))]), p=0.25
        ),
        # Jittering together is slower (high variance)
        # transforms.RandomApply(
        #     nn.ModuleList([
        #         transforms.ColorJitter(
        #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
        #             saturation=(0.7, 1.2), hue=0.2)
        #     ]), 0.8),
        T.RandomApply(nn.ModuleList([T.GaussianBlur(5)]), p=0.25),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return torch.jit.script(nn.Sequential(*img_transforms))


def _get_img_transforms(
    IMAGENET_MEAN, IMAGENET_STD, is_eval, crop_dim, modality, same_transform, defer_transform=False, multi_crop=False,
):

    """Get the cropping transformations and some images transformations that will be applied.

    Args:
        is_eval (bool): Whether we want train or eval transformations.
        crop_dim (int): Dimension for cropping.
        modality (string): Modality of the frame.
        same_transform (bool): Whether to apply same transform to each frame.
        defer_transform (bool): Whether some transforms have been defered to gpu.
            Default: False.
        multi_crop (bool): Whether multi cropping is applied.
            Default: False.

    Returns:
        crop_transform
        img_transform
    """
    import torchvision.transforms as transforms
    import torch.nn as nn

    crop_transform = None
    if crop_dim is not None:
        if multi_crop:
            assert is_eval
            crop_transform = ThreeCrop(crop_dim)
        elif is_eval:
            crop_transform = transforms.CenterCrop(crop_dim)
        elif same_transform:
            print("=> Using seeded crops!")
            crop_transform = SeedableRandomSquareCrop(crop_dim)
        else:
            crop_transform = transforms.RandomCrop(crop_dim)

    img_transforms = []
    if modality == "rgb":
        if not is_eval:
            img_transforms.append(transforms.RandomHorizontalFlip())

            if not defer_transform:
                img_transforms.extend(
                    [
                        # Jittering separately is faster (low variance)
                        transforms.RandomApply(
                            nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25
                        ),
                        transforms.RandomApply(
                            nn.ModuleList(
                                [transforms.ColorJitter(saturation=(0.7, 1.2))]
                            ),
                            p=0.25,
                        ),
                        transforms.RandomApply(
                            nn.ModuleList(
                                [transforms.ColorJitter(brightness=(0.7, 1.2))]
                            ),
                            p=0.25,
                        ),
                        transforms.RandomApply(
                            nn.ModuleList(
                                [transforms.ColorJitter(contrast=(0.7, 1.2))]
                            ),
                            p=0.25,
                        ),
                        # Jittering together is slower (high variance)
                        # transforms.RandomApply(
                        #     nn.ModuleList([
                        #         transforms.ColorJitter(
                        #             brightness=(0.7, 1.2), contrast=(0.7, 1.2),
                        #             saturation=(0.7, 1.2), hue=0.2)
                        #     ]), p=0.8),
                        transforms.RandomApply(
                            nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25
                        ),
                    ]
                )

        if not defer_transform:
            img_transforms.append(
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            )
    else:
        raise NotImplementedError(modality)

    img_transform = torch.jit.script(nn.Sequential(*img_transforms))
    return crop_transform, img_transform

def get_remaining(data_len, batch_size):
    """Return the padding that ensures that all batches have an equal number of items, which is required with the pipeline to make sur that all clips are processed.
    Args:
        data_len (int): The length of dataset.
        batch_size (int).

    Returns:
        (int): The number of elements to add.
    """
    return (math.ceil(data_len / batch_size) * batch_size) - data_len


import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

class RandomHorizontalFlipFLow(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1)[0] < self.p:
            shape = img.shape
            img.view((-1,) + shape[-3:])[:, 1, :, :] *= -1
            return img.flip(-1)
        return img


class RandomOffsetFlow(nn.Module):

    def __init__(self, p=0.5, x=0.1, y=0.05):
        super().__init__()
        self.p = p
        self.x = x
        self.y = y

    def forward(self, img):
        if torch.rand(1)[0] < self.p:
            shape = img.shape
            view = img.view((-1,) + shape[-3:])
            view[:, 1, :, :] += (torch.rand(1, device=img.device)[0] * 2 - 1) * self.x
            view[:, 0, :, :] += (torch.rand(1, device=img.device)[0] * 2 - 1) * self.y
        return img


class RandomGaussianNoise(nn.Module):

    def __init__(self, p=0.5, s=0.1):
        super().__init__()
        self.p = p
        self.std = s**0.5

    def forward(self, img):
        v = torch.rand(1)[0]
        if v < self.p:
            img += torch.randn(img.shape, device=img.device) * self.std
        return img


class SeedableRandomSquareCrop:

    def __init__(self, dim):
        self._dim = dim

    def __call__(self, img):
        c, h, w = img.shape[-3:]
        x, y = 0, 0
        if h > self._dim:
            y = random.randint(0, h - self._dim)
        if w > self._dim:
            x = random.randint(0, w - self._dim)
        return F.crop(img, y, x, self._dim, self._dim)


class ThreeCrop:

    def __init__(self, dim):
        self._dim = dim

    def __call__(self, img):
        c, h, w = img.shape[-3:]
        y = (h - self._dim) // 2
        ret = []
        dw = w - self._dim
        for x in (0, dw // 2, dw):
            ret.append(F.crop(img, y, x, self._dim, self._dim))
        return torch.stack(ret)


class VideoTransform:
    def __init__(self, config, mode="train"):
        self.mode = mode
        self.config = config

        self.frame_height, self.frame_width = config.DATA.frame_size
        self.augmentations = config.DATA.augmentations

    def __call__(self, frames: np.ndarray):
        """
        frames: np.ndarray (T, H, W, C)
        """

        if self.mode != "train":
            return frames

        is_float = frames.dtype in (np.float32, np.float64)

        T_, H, W, C = frames.shape
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)

        aug = self.augmentations

        # ---------------- Random crop ----------------
        if getattr(aug, "random_crop", False):
            scale = getattr(aug, "scale", (0.8, 1.0))
            ratio = getattr(aug, "ratio", (3/4, 4/3))

            i, j, h, w = T.RandomResizedCrop.get_params(
                frames_t[0], scale=scale, ratio=ratio
            )

            frames_t = torch.stack([
                F.resized_crop(
                    f, i, j, h, w,
                    size=(self.frame_height, self.frame_width),
                    interpolation=F.InterpolationMode.BILINEAR,
                )
                for f in frames_t
            ])

        # ---------------- Affine ----------------
        if getattr(aug, "random_affine", False):
            max_translate = getattr(aug, "translate", (0.1, 0.1))
            scale_range = getattr(aug, "affine_scale", (0.9, 1.0))

            tx = int(random.uniform(-max_translate[0], max_translate[0]) * W)
            ty = int(random.uniform(-max_translate[1], max_translate[1]) * H)
            scale_factor = random.uniform(scale_range[0], scale_range[1])

            frames_t = torch.stack([
                F.affine(
                    f,
                    angle=0.0,
                    translate=[tx, ty],
                    scale=scale_factor,
                    shear=[0.0, 0.0],
                    interpolation=F.InterpolationMode.BILINEAR,
                )
                for f in frames_t
            ])

        # ---------------- Perspective ----------------
        if getattr(aug, "random_perspective", False):
            distortion_scale = getattr(aug, "distortion_scale", 0.3)
            p = getattr(aug, "perspective_prob", 0.5)

            if random.random() < p:
                startpoints, endpoints = T.RandomPerspective.get_params(
                    width=W, height=H, distortion_scale=distortion_scale
                )

                frames_t = torch.stack([
                    F.perspective(
                        f,
                        startpoints=startpoints,
                        endpoints=endpoints,
                        interpolation=F.InterpolationMode.BILINEAR,
                    )
                    for f in frames_t
                ])

        # ---------------- Rotation ----------------
        if getattr(aug, "random_rotation", False):
            degrees = getattr(aug, "rotation_degrees", 5)
            angle = random.uniform(-degrees, degrees)

            frames_t = torch.stack([
                F.rotate(f, angle=angle,
                         interpolation=F.InterpolationMode.BILINEAR)
                for f in frames_t
            ])

        # ---------------- Color jitter ----------------
        if getattr(aug, "color_jitter", False):
            jitter_prob = getattr(aug, "jitter_prob", 1.0) # defaults to 1.0
            brightness, contrast, saturation, hue = getattr(
                aug, "jitter_params", (0.2, 0.2, 0.2, 0.05)
            )

            jitter = T.ColorJitter(brightness, contrast, saturation, hue)
            frames_t = torch.stack([jitter(f) for f in frames_t])

        # ---------------- Flip ----------------
        if getattr(aug, "random_horizontal_flip", False):
            if random.random() < getattr(aug, "flip_prob", 0.5):
                frames_t = torch.flip(frames_t, dims=[3])

        result = frames_t.permute(0, 2, 3, 1)
        return result.numpy().astype(np.float32 if is_float else np.uint8)


def build_transform(config, mode="train"):
    return VideoTransform(config, mode)

def get_transforms_model(pre_model):
    from torchvision.models.video import R3D_18_Weights, MC3_18_Weights
    from torchvision.models.video import R2Plus1D_18_Weights, S3D_Weights
    from torchvision.models.video import MViT_V2_S_Weights, MViT_V1_B_Weights
    from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights
    
    if pre_model == "r3d_18":
        transforms_model = R3D_18_Weights.KINETICS400_V1.transforms()        
    elif pre_model == "s3d":
        transforms_model = S3D_Weights.KINETICS400_V1.transforms()       
    elif pre_model == "mc3_18":
        transforms_model = MC3_18_Weights.KINETICS400_V1.transforms()       
    elif pre_model == "r2plus1d_18":
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "mvit_v2_s":
        transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    else:
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()

    return transforms_model

def feats2clip(
    feats, stride, clip_length, padding="replicate_last", off=0, modif_last_index=False
):
    """Converts a sequence of feature vectors into a sequence of overlapping clips.
    Args:
        feats: A tensor of shape (num_frames, feature_dim), representing the input feature vectors.
        stride: The step size between the starting points of consecutive clips.
        clip_length: The number of frames in each clip.
        padding: The padding strategy, either "zeropad" or "replicate_last".
        off: An offset to adjust the starting points of clips.
        modif_last_index: A flag indicating whether to modify the last index to ensure the last clip is aligned with the end of feats.
    """
    if padding == "zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0] / stride) * stride
        print("pad need to be", clip_length - pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length - pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)
    if modif_last_index:
        off = 0
    idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length - off):
        idxs.append(idx + i)
    idx = torch.stack(idxs, dim=1)

    if padding == "replicate_last":
        idx = idx.clamp(0, feats.shape[0] - 1)
    if modif_last_index:
        idx[-1] = torch.arange(clip_length) + feats.shape[0] - clip_length
        return feats[idx, :]
    # print(idx)
    return feats[idx, ...]

def getNegativeIndexes(labels, params, chunk_size):

    zero_one_labels = np.zeros(labels.shape)
    for i in np.arange(labels.shape[1]):
        zero_one_labels[:, i] = 1 - np.logical_or(
            np.where(labels[:, i] >= params[3, i], 1, 0),
            np.where(labels[:, i] <= params[0, i], 1, 0),
        )
    zero_one = np.where(np.sum(zero_one_labels, axis=1) > 0, 0, 1)

    zero_one_pad = np.append(
        np.append(
            [
                1 - zero_one[0],
            ],
            zero_one,
            axis=0,
        ),
        [1 - zero_one[-1]],
        axis=0,
    )
    zero_one_pad_shift = np.append(zero_one_pad[1:], zero_one_pad[-1])

    zero_one_sub = zero_one_pad - zero_one_pad_shift

    zero_to_one_index = np.where(zero_one_sub == -1)[0]
    one_to_zero_index = np.where(zero_one_sub == 1)[0]

    if zero_to_one_index[0] > one_to_zero_index[0]:
        one_to_zero_index = one_to_zero_index[1:]
    if zero_to_one_index.shape[0] > one_to_zero_index.shape[0]:
        zero_to_one_index = zero_to_one_index[:-1]

    list_indexes = list()

    for i, j in zip(zero_to_one_index, one_to_zero_index):
        if j - i >= chunk_size:
            list_indexes.append([i, j])

    return list_indexes

def getChunks_anchors(labels, game_index, params, chunk_size=240, receptive_field=80):

    # get indexes of labels
    indexes = list()
    for i in np.arange(labels.shape[1]):
        indexes.append(np.where(labels[:, i] == 0)[0].tolist())

    # Positive chunks
    anchors = list()

    class_counter = 0
    for event in indexes:
        for element in event:
            anchors.append([game_index, element, class_counter])
        class_counter += 1

    # Negative chunks

    negative_indexes = getNegativeIndexes(labels, params, chunk_size)

    for negative_index in negative_indexes:
        start = [negative_index[0], negative_index[1]]
        anchors.append([game_index, start, labels.shape[1]])

    return anchors

def getTimestampTargets(labels, num_detections):

    targets = np.zeros(
        (labels.shape[0], num_detections, 2 + labels.shape[-1]), dtype="float"
    )

    for i in np.arange(labels.shape[0]):

        time_indexes, class_values = np.where(labels[i] == 0)

        counter = 0

        for time_index, class_value in zip(time_indexes, class_values):

            # Confidence
            targets[i, counter, 0] = 1.0
            # frame index normalized
            targets[i, counter, 1] = time_index / (labels.shape[1])
            # The class one hot encoded
            targets[i, counter, 2 + class_value] = 1.0
            counter += 1

            if counter >= num_detections:
                print(
                    "More timestamp than what was fixed... A lot happened in that chunk"
                )
                break

    return targets


def rulesToCombineShifts(shift_from_last_event, shift_until_next_event, params):
    """Set the rule to combine shifts based on two rules and parameters.

    Args:
        shift_from_last_event: First rule.
        shift_until_next_event: Second rule.
        params: Parameters to choose the rule.
    Returns:
        The rule.
    """
    s1 = shift_from_last_event
    s2 = shift_until_next_event
    K = params

    if s1 < K[2]:
        value = s1
    elif s1 < K[3]:
        if s2 <= K[0]:
            value = s1
        else:
            if (s1 - K[2]) / (K[3] - K[2]) < (K[1] - s2) / (K[1] - K[0]):
                value = s1
            else:
                value = s2
    else:
        value = s2

    return value


def oneHotToShifts(onehot, params):
    """

    Args:
        onehot: onehot vector of the shape (number of frames, number of actions).
        params: used to construct the shift.
    """
    nb_frames = onehot.shape[0]
    nb_actions = onehot.shape[1]

    Shifts = np.empty(onehot.shape)

    for i in range(nb_actions):

        x = onehot[:, i]
        K = params[:, i]
        shifts = np.empty(nb_frames)

        loc_events = np.where(x == 1)[0]
        nb_events = len(loc_events)

        if nb_events == 0:
            shifts = np.full(nb_frames, K[0])
        elif nb_events == 1:
            shifts = np.arange(nb_frames) - loc_events
        else:
            loc_events = np.concatenate(([-K[3]], loc_events, [nb_frames - K[0]]))
            for j in range(nb_frames):
                shift_from_last_event = j - loc_events[np.where(j >= loc_events)[0][-1]]
                shift_until_next_event = j - loc_events[np.where(j < loc_events)[0][0]]
                shifts[j] = rulesToCombineShifts(
                    shift_from_last_event, shift_until_next_event, K
                )

        Shifts[:, i] = shifts

    return Shifts

def timestamps2long(output_spotting, video_size, chunk_size, receptive_field):
    """Method to transform the timestamps to vectors"""
    start = 0
    last = False
    receptive_field = receptive_field // 2

    timestamps_long = (
        torch.zeros(
            [video_size, output_spotting.size()[-1] - 2],
            dtype=torch.float,
            device=output_spotting.device,
        )
        - 1
    )

    for batch in np.arange(output_spotting.size()[0]):

        tmp_timestamps = (
            torch.zeros(
                [chunk_size, output_spotting.size()[-1] - 2],
                dtype=torch.float,
                device=output_spotting.device,
            )
            - 1
        )

        for i in np.arange(output_spotting.size()[1]):
            tmp_timestamps[
                torch.floor(output_spotting[batch, i, 1] * (chunk_size - 1)).type(
                    torch.int
                ),
                torch.argmax(output_spotting[batch, i, 2:]).type(torch.int),
            ] = output_spotting[batch, i, 0]

        # ------------------------------------------
        # Store the result of the chunk in the video
        # ------------------------------------------

        # For the first chunk
        if start == 0:
            timestamps_long[0 : chunk_size - receptive_field] = tmp_timestamps[
                0 : chunk_size - receptive_field
            ]

        # For the last chunk
        elif last:
            timestamps_long[start + receptive_field : start + chunk_size] = (
                tmp_timestamps[receptive_field:]
            )
            break

        # For every other chunk
        else:
            timestamps_long[
                start + receptive_field : start + chunk_size - receptive_field
            ] = tmp_timestamps[receptive_field : chunk_size - receptive_field]

        # ---------------
        # Loop Management
        # ---------------

        # Update the index
        start += chunk_size - 2 * receptive_field
        # Check if we are at the last index of the game
        if start + chunk_size >= video_size:
            start = video_size - chunk_size
            last = True
    return timestamps_long


def batch2long(output_segmentation, video_size, chunk_size, receptive_field):
    """Method to transform the batches to vectors."""
    start = 0
    last = False
    receptive_field = receptive_field // 2

    segmentation_long = torch.zeros(
        [video_size, output_segmentation.size()[-1]],
        dtype=torch.float,
        device=output_segmentation.device,
    )

    for batch in np.arange(output_segmentation.size()[0]):

        tmp_segmentation = torch.nn.functional.one_hot(
            torch.argmax(output_segmentation[batch], dim=-1),
            num_classes=output_segmentation.size()[-1],
        )

        # ------------------------------------------
        # Store the result of the chunk in the video
        # ------------------------------------------

        # For the first chunk
        if start == 0:
            segmentation_long[0 : chunk_size - receptive_field] = tmp_segmentation[
                0 : chunk_size - receptive_field
            ]

        # For the last chunk
        elif last:
            segmentation_long[start + receptive_field : start + chunk_size] = (
                tmp_segmentation[receptive_field:]
            )
            break

        # For every other chunk
        else:
            segmentation_long[
                start + receptive_field : start + chunk_size - receptive_field
            ] = tmp_segmentation[receptive_field : chunk_size - receptive_field]

        # ---------------
        # Loop Management
        # ---------------

        # Update the index
        start += chunk_size - 2 * receptive_field
        # Check if we are at the last index of the game
        if start + chunk_size >= video_size:
            start = video_size - chunk_size
            last = True
    return segmentation_long

# import torch
# import numpy as np
# import decord
# from decord import cpu
# import torchvision.transforms as T

# def read_video(video_path):
#     """Read video frames into tensor (T, C, H, W)"""
#     vr = decord.VideoReader(video_path, ctx=cpu(0))
#     frames = vr.get_batch(range(len(vr)))  # (T, H, W, C)
#     frames = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2)  # (T, C, H, W)
#     return frames

# def resample_video_idx(num_frames, original_fps, new_fps):
#     """Return frame indices to match new fps"""
#     step = float(original_fps) / new_fps
#     if step.is_integer():
#         step = int(step)
#         return slice(None, None, step)
#     idxs = torch.arange(num_frames, dtype=torch.float32) * step
#     idxs = idxs.floor().to(torch.int64)
#     return idxs

# def process_frames(video_tensor, target_num_frames, input_fps, target_fps=None):
#     """Uniform sampling, padding, FPS adjustment"""
#     target_fps = target_fps or input_fps
#     num_frames = video_tensor.size(0)
#     duration = num_frames / input_fps

#     # Too short → resample
#     if num_frames < target_num_frames:
#         new_fps = np.ceil(target_num_frames / duration)
#         idxs = resample_video_idx(target_num_frames, input_fps, new_fps)
#         idxs = np.clip(idxs, 0, num_frames - 1)
#         video_tensor = video_tensor[idxs]

#     # Too long → uniform sample
#     elif num_frames > target_num_frames:
#         idxs = np.linspace(0, num_frames - 1, target_num_frames)
#         video_tensor = video_tensor[idxs.astype(int)]

#     # Pad if still short
#     if video_tensor.size(0) < target_num_frames:
#         pad = target_num_frames - video_tensor.size(0)
#         last_frame = video_tensor[-1:].repeat(pad, 1, 1, 1)
#         video_tensor = torch.cat([video_tensor, last_frame], dim=0)

#     return video_tensor

# def build_transform(frame_size=(224, 224), mean=None, std=None):
#     """Return default frame transform"""
#     mean = mean or [0.485, 0.456, 0.406]
#     std = std or [0.229, 0.224, 0.225]
#     return T.Compose([
#         T.ConvertImageDtype(torch.float32),
#         T.Resize(frame_size),
#         T.Normalize(mean, std),
#     ])

