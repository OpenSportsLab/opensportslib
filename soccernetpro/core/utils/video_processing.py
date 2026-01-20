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

def get_remaining(data_len, batch_size):
    """Return the padding that ensures that all batches have an equal number of items, which is required with the pipeline to make sur that all clips are processed.
    Args:
        data_len (int): The length of dataset.
        batch_size (int).

    Returns:
        (int): The number of elements to add.
    """
    return (math.ceil(data_len / batch_size) * batch_size) - data_len


def build_transform(config, mode="train"):
    import random
    import numpy as np
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as F

    frame_height, frame_width = config.DATA.frame_size
    augmentations = config.DATA.augmentations

    def transform_fn(frames: np.ndarray):
        """
        frames: np.ndarray, (T, H, W, C), dtype=uint8
        returns: np.ndarray, (T, H, W, C), dtype=uint8
        """

        if mode != "train":
            return frames  # processor handles resize & norm

        T_, H, W, C = frames.shape
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)  # T,C,H,W

        # ---- Random resized crop (once per clip) ----
        if getattr(augmentations, "random_crop", False):
            scale = getattr(augmentations, "scale", (0.8, 1.0))
            ratio = getattr(augmentations, "ratio", (3/4, 4/3))

            i, j, h, w = torch.transforms.RandomResizedCrop.get_params(
                frames_t[0],
                scale=scale,
                ratio=ratio,
            )

            frames_t = torch.stack([
                F.resized_crop(
                    f,
                    i, j, h, w,
                    size=(frame_height, frame_width),
                    interpolation=F.InterpolationMode.BILINEAR,
                )
                for f in frames_t
            ])

        # ---- Random affine (translate + scale, ONCE per clip) ----
        if getattr(augmentations, "random_affine", False):
            max_translate = getattr(augmentations, "translate", (0.1, 0.1))
            scale_range = getattr(augmentations, "affine_scale", (0.9, 1.0))

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

        # ---- Random perspective (ONCE per clip) ----
        if getattr(augmentations, "random_perspective", False):
            distortion_scale = getattr(augmentations, "distortion_scale", 0.3)
            p = getattr(augmentations, "perspective_prob", 0.5)

            if random.random() < p:
                startpoints, endpoints = T.RandomPerspective.get_params(
                    width=W,
                    height=H,
                    distortion_scale=distortion_scale,
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

        # ---- Random rotation (ONCE per clip) ----
        if getattr(augmentations, "random_rotation", False):
            degrees = getattr(augmentations, "rotation_degrees", 5)
            angle = random.uniform(-degrees, degrees)

            frames_t = torch.stack([
                F.rotate(
                    f,
                    angle=angle,
                    interpolation=F.InterpolationMode.BILINEAR,
                )
                for f in frames_t
            ])


        # ---- Color jitter (ONCE per clip) ----
        if getattr(augmentations, "color_jitter", False):
            brightness, contrast, saturation, hue = getattr(
                augmentations,
                "jitter_params",
                (0.2, 0.2, 0.2, 0.05),
            )

            # Create jitter transform ONCE
            color_jitter = T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )

            # Apply same jitter to all frames
            frames_t = torch.stack([
                color_jitter(f) for f in frames_t
            ])
        
        # ---- Horizontal flip (once per clip) ----
        if getattr(augmentations, "random_horizontal_flip", False):
            if random.random() < getattr(augmentations, "flip_prob", 0.5):
                frames_t = torch.flip(frames_t, dims=[3])

        return frames_t.permute(0, 2, 3, 1).numpy().astype(np.uint8)

    return transform_fn

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

