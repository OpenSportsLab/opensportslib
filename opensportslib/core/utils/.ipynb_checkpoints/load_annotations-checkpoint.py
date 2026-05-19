import json
import os
import tqdm
import logging
import cv2
import math
import torch
from opensportslib.core.utils.video_processing import get_stride, read_fps, get_num_frames
from opensportslib.core.utils.config import load_json
from collections import defaultdict

def load_annotations(
    annotations_path,
    task_key="action",
    exclude_labels=None,
    multiview=False,
    input_type="video",
    allow_missing_labels=False,
    max_games=None
):

    with open(annotations_path, "r") as f:
        data = json.load(f)

    # this is used for data slicing experiments.
    # and doesn't affect the validation and test sets.
    # if you haven't added "data_scaling" to the config, this will be ignored.
    if max_games is not None:
        all_game_ids = sorted(set(
            item.get("metadata", {}).get("game_id", "")
            for item in data["data"]
        )) 

        # remove empty string if any items lack game_id
        all_game_ids = [g for g in all_game_ids if g]

        # warn if any items lack game_id
        items_without_game_id = sum(
            1 for item in data["data"]
            if not item.get("metadata", {}).get("game_id", "")
        )
        if items_without_game_id > 0:
            print(f"warning: {items_without_game_id}/{len(data['data'])} items have "
                  f"no game_id, these will not be affected by data slicing")

        if max_games < len(all_game_ids):
            keep_ids = set(all_game_ids[:max_games])
            original_count = len(data["data"])
            data["data"] = [
                item for item in data["data"]
                if item.get("metadata", {}).get("game_id", "") in keep_ids
            ]
            print(f"data slicing: {max_games}/{len(all_game_ids)} games, "
                  f"{len(data['data'])}/{original_count} samples retained")
    # ----- data slicing ends here -----

    exclude_labels = set(exclude_labels or [""])

    # Label list for the selected task
    label_list = [
        lbl for lbl in data["labels"][task_key]["labels"]
        if lbl not in exclude_labels
    ]
    label_map = {name: idx for idx, name in enumerate(label_list)}

    # Group by action id (without view suffix)
    grouped = defaultdict(lambda: {
        "video_paths": [],
        "label": None
    })

    for item in data["data"]:
        label_idx = None

        if "labels" in item and task_key in item["labels"]:
            action_label = item["labels"][task_key].get("label", None)
            if action_label in exclude_labels:
                continue

            if action_label in label_map:
                label_idx = label_map[action_label]

        elif not allow_missing_labels:
            # training mode requires labels
            continue

        # Extract group key
        item_id = item["id"]
        if multiview and "_view" in item_id:
            group_id = item_id.rsplit("_view", 1)[0]
        else:
            group_id = item_id

        # Collect clips
        clips = [
            inp["path"]
            for inp in item.get("inputs", [])
            if inp.get("type") == input_type and "path" in inp
        ]
        if not clips:
            continue

        grouped[group_id]["video_paths"].extend(clips)
        if label_idx is not None:
            grouped[group_id]["label"] = label_idx
        grouped[group_id]["id"] = group_id

    return list(grouped.values()), label_map


def load_annotations_(annotations_path, exclude_labels=None):
    with open(annotations_path, "r") as f:
        data = json.load(f)

    exclude_labels = exclude_labels or ["", "Challenge"]
    # Filter labels
    label_list = [
        name for name in data["labels"]["foul_type"]["labels"]
        if name not in exclude_labels
    ]
    
    label_map = {name: idx for idx, name in enumerate(label_list)}
    samples = []

    for item in data["data"]:
        foul_label = item["labels"]["foul_type"]["label"]
        
        # Skip unwanted labels
        if foul_label in exclude_labels:
            continue

        label_idx = label_map.get(foul_label, -1)
        if label_idx == -1:
            continue

        # collect *all* video clips
        all_clips = [
            c.get("path", "") for c in item.get("inputs", [])
            if c.get("type") == "video"
        ]
        all_clips = [p.replace("Dataset/Train", "train")
                     .replace("Dataset/Test", "test")
                     .replace("Dataset/Valid", "valid") + ".mp4"
                     for p in all_clips if p]

        if not all_clips:
            continue

        samples.append({
            "video_paths": all_clips,
            "label": label_idx
        })
    print(label_map)
    return samples, label_map


def has_localization_events(annotation_path):
    import json

    if annotation_path is None:
        return False

    with open(annotation_path) as f:
        data = json.load(f)

    for item in data.get("data", []):
        if item.get("events"):
            return True

    return False

def annotationstoe2eformat(
    label_files,
    video_dirs,
    input_fps,
    extract_fps,
    dali
):
    """
    Adapt SN Ball Action Spotting annotations to E2E format.

    Supports JSON with:
      - top-level "data"
      - video path in inputs[0]["path"]
      - events with "label" and "position_ms"

    Args:
        label_files (str | list[str]): Annotation JSON files
        video_dirs (str | list[str]): Root video directories
        input_fps (int): FPS expected by the model
        extract_fps (int): FPS for frame extraction
        dali (bool): Whether using DALI or OpenCV
    """

    if not isinstance(label_files, list):
        label_files = [label_files]
    if not isinstance(video_dirs, list):
        video_dirs = [video_dirs]

    assert len(label_files) == len(video_dirs)

    labels_e2e = []
    classes_by_label_dir = []
    task_name_list = []

    for label_path, video_dir in zip(label_files, video_dirs):
        logging.info(f"Processing {label_path} to e2e format")

        annotations = load_json(label_path)

        # ---- Extract class list (ball_action) ----
        for task_name, task_data in annotations["labels"].items():
            labels = task_data.get("labels", {})
            task_name_list.append(task_name)

        classes_by_label_dir.append(labels)

        # ---- Iterate videos ----
        videos = annotations["data"]

        for video in tqdm.tqdm(videos):
            # ---- Video path & metadata ----
            video_path = video["inputs"][0]["path"].replace(" ", "_")
            #game_dir  = os.path.dirname(video_path)
            #game_name = os.path.basename(video_path)
            full_video_path = os.path.join(video_dir, video_path)
            assert os.path.isfile(full_video_path), full_video_path
            vc = cv2.VideoCapture(full_video_path)
            width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vc.get(cv2.CAP_PROP_FPS)
            num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

            # ---- FPS handling ----
            target_fps = extract_fps if extract_fps < fps else fps
            sample_fps = read_fps(fps, target_fps)

            num_frames_after = get_num_frames(
                num_frames, fps, target_fps
            )

            if dali:
                if get_stride(fps, target_fps) != get_stride(input_fps, extract_fps):
                    sample_fps = fps / get_stride(input_fps, extract_fps)
                    num_frames_dali = math.ceil(
                        num_frames / get_stride(input_fps, extract_fps)
                    )
                else:
                    num_frames_dali = num_frames_after

            # ---- Events ----
            events = []
            for ann in video.get("events", []):
                position_ms = float(ann["position_ms"])

                if dali:
                    if get_stride(fps, target_fps) != get_stride(input_fps, extract_fps):
                        adj_frame = (
                            position_ms / 1000
                            * (fps / get_stride(input_fps, extract_fps))
                        )
                    else:
                        adj_frame = position_ms / 1000 * sample_fps
                else:
                    adj_frame = position_ms / 1000 * sample_fps

                if int(adj_frame) == 0:
                    adj_frame = 1

                events.append({
                    "frame": int(adj_frame),
                    "label": ann["label"],

                })

            events.sort(key=lambda x: x["frame"])

            labels_e2e.append({
                "events": events,
                "fps": sample_fps,
                "num_frames": num_frames_dali if dali else num_frames_after,
                "num_frames_base": num_frames,
                "num_events": len(events),
                "width": width,
                "height": height,
                "video": full_video_path,
                "path": video_path,
            })

    # ---- Sanity checks ----
    base_classes = classes_by_label_dir[0]
    for c in classes_by_label_dir:
        assert c == base_classes

    labels_e2e.sort(key=lambda x: x["video"])

    return labels_e2e, task_name_list[0]

# def annotationstoe2eformat(label_files, video_dirs, input_fps, extract_fps, dali):
#     """Adapt annotations jsons to e2e format.

#     Args:
#         label_files (string,list[string]): Json files of annotations.
#         label_dirs (string,list[string]): Data root folder of videos. Must match number of label files.
#         input_fps (int): Fps of input videos.
#         extract_fps (int): Fps at which we extract frames.
#         dali (bool): WHether processing with dali or opencv.
#     """

#     if not isinstance(label_files, list):
#         label_files = [label_files]
#     if not isinstance(video_dirs, list):
#         video_dirs = [video_dirs]
#     assert len(label_files) == len(video_dirs)

#     labels_e2e = list()
#     classes_by_label_dir = []
#     for label_dir, video_dir in zip(label_files, video_dirs):
#         logging.info("Processing " + label_dir + " to e2e format.")
#         videos = []
#         annotations = load_json(label_dir)
#         labels = annotations["labels"]
#         classes_by_label_dir.append(labels)
#         videos = annotations["videos"]
#         for video in tqdm.tqdm(videos):
#             if "annotations" in video.keys():
#                 video_annotations = video["annotations"]
#             else:
#                 video_annotations = []

#             num_events = 0

#             vc = cv2.VideoCapture(os.path.join(video_dir, video["path"]))
#             width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = vc.get(cv2.CAP_PROP_FPS)
#             num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

#             sample_fps = read_fps(fps, extract_fps if extract_fps < fps else fps)
#             num_frames_after = get_num_frames(
#                 num_frames, fps, extract_fps if extract_fps < fps else fps
#             )

#             if dali:
#                 if get_stride(
#                     fps, extract_fps if extract_fps < fps else fps
#                 ) != get_stride(input_fps, extract_fps):
#                     sample_fps = fps / get_stride(input_fps, extract_fps)
#                     num_frames_dali = math.ceil(
#                         num_frames / get_stride(input_fps, extract_fps)
#                     )
#                 else:
#                     num_frames_dali = num_frames_after

#             # video_id = os.path.splitext(video["path"])[0]
#             video_id = os.path.join(video_dir, video["path"])

#             events = []
#             for annotation in video_annotations:
#                 if dali:
#                     if get_stride(
#                         fps, extract_fps if extract_fps < fps else fps
#                     ) != get_stride(input_fps, extract_fps):
#                         adj_frame = (
#                             float(annotation["position"])
#                             / 1000
#                             * (fps / get_stride(input_fps, extract_fps))
#                         )
#                     else:
#                         adj_frame = float(annotation["position"]) / 1000 * sample_fps
#                     if int(adj_frame) == 0:
#                         adj_frame = 1
#                 else:
#                     adj_frame = float(annotation["position"]) / 1000 * sample_fps
#                 events.append(
#                     {
#                         "frame": int(adj_frame),
#                         "label": annotation["label"],
#                         "team": annotation["team"],
#                         "visibility": annotation["visibility"],
#                     }
#                 )

#             num_events += len(events)
#             events.sort(key=lambda x: x["frame"])

#             labels_e2e.append(
#                 {
#                     "events": events,
#                     "fps": sample_fps,
#                     "num_frames": num_frames_dali if dali else num_frames_after,
#                     "num_frames_base": num_frames,
#                     "num_events": len(events),
#                     "width": width,
#                     "height": height,
#                     "video": video_id,
#                     "path": video["path"],
#                 }
#             )
#         assert len(video_annotations) == num_events
#     classes = classes_by_label_dir[0]
#     for classes_tmp in classes_by_label_dir:
#         assert classes == classes_tmp
#     labels_e2e.sort(key=lambda x: x["video"])
#     return labels_e2e

def construct_labels(path, extract_fps):
    """This method is used when the input of the dataset is a video file instead of a json file.
    It creates a pseudo json by processing the video to get metadatas.

    Args:
        path (string): The path of the video file.
        extract_fps (int): The fps at which we want to extract frames.

    Returns:
        List(dict): The pseudo json object.
        (int): stride at which we will process the video.
    """
    wanted_sample_fps = extract_fps
    vc = cv2.VideoCapture(path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_fps = read_fps(fps, wanted_sample_fps if wanted_sample_fps < fps else fps)
    num_frames_after = get_num_frames(
        num_frames, fps, wanted_sample_fps if wanted_sample_fps < fps else fps
    )

    return [
        {
            "video": path,
            "path": path,
            "num_frames": num_frames_after,
            "num_frames_base": num_frames,
            "num_events": 0,
            "events": [],
            "fps": sample_fps,
            "width": 398,
            "height": 224,
        }
    ], get_stride(fps, wanted_sample_fps if wanted_sample_fps < fps else fps)


# def get_repartition_gpu():
#     """Returns the distribution of gpus that will be used by pipelines for dali."""
#     x = torch.cuda.device_count() - 1
#     print("Number of gpus:", x)
#     if x == 1:
#         return [0], [0]
#     if x == 2:
#         return [0, 1], [0, 1]
#     elif x == 3:
#         return [0, 1], [1, 2]
#     elif x > 3:
#         return [0, 1, 2, 3], [0, 2, 1, 3]

def get_repartition_gpu(max_train_gpus=2):
    n = torch.cuda.device_count()

    if n == 0:
        return [], []

    train_gpus = list(range(min(n, max_train_gpus)))
    dali_gpus = train_gpus.copy()

    return train_gpus, dali_gpus
    

def check_config(cfg, split="train"):
    """Check for incoherences, missing elements in dict config.
    The checks are different regarding the methods.

    Args:
        cfg (dict): Config dictionnary.

    """
    from opensportslib.core.utils.config import load_json, load_classes
    from omegaconf import ListConfig
    if cfg.MODEL.runner.type == "runner_e2e":
        if cfg.dali == True:
            cfg.TRAIN.repartitions = get_repartition_gpu(cfg.SYSTEM.GPU)
        assert cfg.DATA.modality in ["rgb"]
        assert cfg.MODEL.backbone.type in [
            # From torchvision
            "rn18",
            "rn18_tsm",
            "rn18_gsm",
            "rn50",
            "rn50_tsm",
            "rn50_gsm",
            # From timm (following its naming conventions)
            "rny002",
            "rny002_tsm",
            "rny002_gsm",
            "rny008",
            "rny008_tsm",
            "rny008_gsm",
            # From timm
            "convnextt",
            "convnextt_tsm",
            "convnextt_gsm",
        ]
        assert cfg.MODEL.head.type in ["", "gru", "deeper_gru", "mstcn", "asformer"]
        # assert cfg.dataset.batch_size % cfg.training.acc_grad_iter == 0
        assert cfg.DATA.train.dataloader.batch_size % cfg.TRAIN.acc_grad_iter == 0
        assert cfg.TRAIN.criterion_valid in ["map", "loss"]
        assert cfg.TRAIN.num_epochs == cfg.TRAIN.scheduler.num_epochs
        assert cfg.TRAIN.acc_grad_iter == cfg.TRAIN.scheduler.acc_grad_iter

        if split=="train":
            data_path = cfg.DATA.train.path
        elif split=="valid":
            data_path = cfg.DATA.valid.path
        elif split=="test":
            data_path = cfg.DATA.test.path
        else:
            raise ValueError(f"Unknown split {split}")
        
        if cfg.TRAIN.start_valid_epoch is None:
            cfg.TRAIN.start_valid_epoch = (
                cfg.TRAIN.num_epochs - cfg.TRAIN.base_num_valid_epochs
            )
        if cfg.DATA.crop_dim is None or cfg.DATA.crop_dim <= 0:
            cfg.DATA.crop_dim = None
        if (
            data_path != None
            and os.path.isfile(data_path)
            and data_path.endswith(".json")
            and "labels" in load_json(data_path).keys()
        ):
            for task_name, task_data in load_json(data_path)["labels"].items():
                classes = task_data.get("labels", {})
            #classes = load_json(cfg.DATA.test.path)["labels"]["action"]["labels"]
        else:
            assert isinstance(cfg.DATA.classes, (list, ListConfig))
            classes = cfg.DATA.classes
        
        #print(classes)
        cfg.DATA.classes = load_classes(classes)


def whether_infer_split(cfg):
    """Given a config dict, check whether we want to infer a split or a single element (can be a game, video or feature file)/

    Args:
        cfg (dict): Config dict.

    Returns:
        bool : True if we infer split, false otherwise. Raises an error if the input is not expected.
    """
    if cfg.type == "SoccerNetGames" or cfg.type == "SoccerNetClipsTestingCALF":
        if cfg.split == None:
            return False
        else:
            return True
    elif (
        cfg.type == "FeatureVideosfromJSON" or cfg.type == "FeatureVideosChunksfromJson"
    ):
        if cfg.path.endswith(".json"):
            return True
        else:
            return False
    elif cfg.type == "VideoGameWithOpencvVideo" or cfg.type == "VideoGameWithDaliVideo":
        if cfg.path.endswith(".json"):
            return True
        else:
            return False
    else:
        raise ValueError(f"Unknown dataset type {cfg.type}")