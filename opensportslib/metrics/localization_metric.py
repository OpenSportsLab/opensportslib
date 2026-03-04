"""
Copyright 2022 James Hong, Haotian Zhang, Matthew Fisher, Michael Gharbi,
Kayvon Fatahalian

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from collections import defaultdict
import os
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from opensportslib.core.utils.config import load_json, store_gz_json, store_json
from opensportslib.core.utils.wandb import log_table_wandb
# from oslactionspotting.core.utils.score import compute_mAPs_E2E
import time
import json
import os
import sys
from collections import defaultdict
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import logging
import copy
from datetime import datetime


def parse_ground_truth(truth):
    """Parse ground truth dict to create dict with the following structure:
    {label : {game : list of frames index}}

    Args:
        truth : object containing list of informations about videos.

    Returns:
        label_dict : {label : {game : list of frames index}}.
    """
    label_dict = defaultdict(lambda: defaultdict(list))
    for x in truth:
        for e in x["events"]:
            label_dict[e["label"]][x["path"]].append(e["frame"])
    return label_dict


def get_predictions(pred, label=None):
    """Get a list of all predictions for a particular label.

    Args:
        pred : Object containing predictions data.

    Returns:
        flat_pred (List): List of tuples containing the name of the video, the frame index and
        the score confidence for each occurence of the particular label.
    """
    flat_pred = []
    for x in pred:
        for e in x["events"]:
            if label is None or e["label"] == label:
                flat_pred.append((x["video"], e["frame"], e["confidence"]))
    flat_pred.sort(key=lambda x: x[-1], reverse=True)
    return flat_pred


def compute_average_precision(
    pred,
    truth,
    tolerance=0,
    min_precision=0,
    plot_ax=None,
    plot_label=None,
    plot_raw_pr=True,
):
    """Compute average precision of predictions regarding truth data with a certain tolerance.

    Args:
        pred: prediction data.
        truth: groundtruth data.
        tolerance (int).
            Default: 0.
        min_precision (int).
            Default: 0.
        plot_ax (list): list of indexes for the axes.
            Default: None.
        plot_label (string): label of the plot.
            Default: None.
        plot_raw_pr (bool): whether to plot raw predictions.
            Default: False.
    Returns:
        average precision.
    """
    total = sum([len(x) for x in truth.values()])
    recalled = set()

    # The full precision curve has TOTAL number of bins, when recall increases
    # by in increments of one
    pc = []
    _prev_score = 1
    for i, (video, frame, score) in enumerate(pred, 1):
        assert score <= _prev_score
        _prev_score = score

        # Find the ground truth frame that is closest to the prediction
        gt_closest = None
        for gt_frame in truth.get(video, []):
            if (video, gt_frame) in recalled:
                continue
            if gt_closest is None or (abs(frame - gt_closest) > abs(frame - gt_frame)):
                gt_closest = gt_frame

        # Record precision each time a true positive is encountered
        if gt_closest is not None and abs(frame - gt_closest) <= tolerance:
            recalled.add((video, gt_closest))
            p = len(recalled) / i
            pc.append(p)

            # Stop evaluation early if the precision is too low.
            # Not used, however when nin_precision is 0.
            if p < min_precision:
                break

    interp_pc = []
    max_p = 0
    for p in pc[::-1]:
        max_p = max(p, max_p)
        interp_pc.append(max_p)
    interp_pc.reverse()  # Not actually necessary for integration

    if plot_ax is not None:
        rc = np.arange(1, len(pc) + 1) / total
        if plot_raw_pr:
            plot_ax.plot(rc, pc, label=plot_label, alpha=0.8)
        plot_ax.plot(rc, interp_pc, label=plot_label, alpha=0.8)

    # Compute AUC by integrating up to TOTAL bins
    return sum(interp_pc) / total


def compute_mAPs_E2E(truth, pred, tolerances=[0, 1, 2, 3, 4], plot_pr=False):
    """Compute mAPs metric for the training module for the E2E method.

    Args:
        truth : Object containing the ground truth data.
        pred : Object containing the predictions data.
        tolerances (List[int]): List of tolerances values.
            Default: [0, 1, 2, 4].
        plot_pr (bool): Whether to plot or not the precision recall curve.
            Default: False.
    """

    assert {v["path"] for v in truth} == {
        v["video"] for v in pred
    }, "Video set mismatch!"

    truth_by_label = parse_ground_truth(truth)

    fig, axes = None, None
    if plot_pr:
        fig, axes = plt.subplots(
            len(truth_by_label),
            len(tolerances),
            sharex=True,
            sharey=True,
            figsize=(16, 16),
        )

    class_aps_for_tol = []
    mAPs = []
    for i, tol in enumerate(tolerances):
        class_aps = []
        for j, (label, truth_for_label) in enumerate(sorted(truth_by_label.items())):
            ap = compute_average_precision(
                get_predictions(pred, label=label),
                truth_for_label,
                tolerance=tol,
                plot_ax=axes[j, i] if axes is not None else None,
            )
            class_aps.append((label, ap))
        mAP = np.mean([x[1] for x in class_aps])
        mAPs.append(mAP)
        class_aps.append(("mAP", mAP))
        class_aps_for_tol.append(class_aps)
    header = ["Class"] + [f"AP@{t}" for t in tolerances]
    rows = []
    for c, _ in class_aps_for_tol[0]:
        row = [str(c)]
        for class_aps in class_aps_for_tol:
            for c2, val in class_aps:
                if c2 == c:
                    row.append(float(val) * 100)
        rows.append(row)

    log_table_wandb(name="AP@tol", rows=rows, headers=header)
    logging.info(tabulate(rows, headers=header, floatfmt="0.2f"))
    logging.info("Avg mAP (across tolerances): {:0.2f}".format(np.mean(mAPs) * 100))

    if plot_pr:
        for i, tol in enumerate(tolerances):
            for j, label in enumerate(sorted(truth_by_label.keys())):
                ax = axes[j, i]
                ax.set_xlabel("Recall")
                ax.set_xlim(0, 1)
                ax.set_ylabel("Precision")
                ax.set_ylim(0, 1.01)
                ax.set_title("{} @ tol={}".format(label, tol))
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    sys.stdout.flush()
    return mAPs, tolerances


class ErrorStat:
    """Class to have error statistics"""

    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        self._err += np.sum(true != pred)
        self._total += true.shape[0]

    def get(self):
        return self._err / self._total

    def get_acc(self):
        return 1.0 - self._get()


class ForegroundF1:
    """Class to have f1 scores"""

    def __init__(self):
        self._tp = defaultdict(int)
        self._fp = defaultdict(int)
        self._fn = defaultdict(int)

    def update(self, true, pred):
        if pred != 0:
            if true != 0:
                self._tp[None] += 1
            else:
                self._fp[None] += 1

            if pred == true:
                self._tp[pred] += 1
            else:
                self._fp[pred] += 1
                if true != 0:
                    self._fn[true] += 1
        elif true != 0:
            self._fn[None] += 1
            self._fn[true] += 1

    def get(self, k):
        return self._f1(k)

    def tp_fp_fn(self, k):
        return self._tp[k], self._fp[k], self._fn[k]

    def _f1(self, k):
        denom = self._tp[k] + 0.5 * self._fp[k] + 0.5 * self._fn[k]
        if denom == 0:
            assert self._tp[k] == 0
            denom = 1
        return self._tp[k] / denom


def build_snpro_prediction_json(pred_events, head_name="action", split=None, created_by="model"):
    """
    Build new v2 prediction JSON.
    Only 'data' is required.
    """

    data = []

    for item in pred_events:
        video = item["video"]
        fps = item["fps"]

        events = []
        for ev in item["events"]:
            events.append({
                "head": head_name,
                "label": ev["label"],
                "gameTime": ev["gameTime"],
                "frame": ev["frame"],
                "position_ms": ev["position"],
                "confidence": ev["confidence"],
            })

        data.append({
            "inputs": [{
                "type": "video",
                "path": video,
                "fps": fps,
            }],
            "events": events
        })

    return {
        "version": "2.0",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "task": "localization",
        "metadata": {
            "type": "predictions",
            "created_by": created_by,
            "split": split,
        },
        "data": data
    }

def process_frame_predictions(
    dali, dataset, classes, pred_dict, high_recall_score_threshold=0.01
):
    """Process predictions by computing statistics, creating dictionnaries
    with predictions and their associated informations

    Args:
        dali (bool): If data processing with dali or opencv.
        dataset (Dataset or DaliGenericIterator).
        classes (Dict): Classes associated with indexes.
        pred_dict (Dict): Mapping between clip and a tuple of scores and support.
        high_recall_score_threshold (float):
            Default: 0.01.

    Returns:
        err (ErrorStat).
        f1 (ForegroundF1).
        pred_events (List[dict]): List of dictionnaries with video, events, fps. Only one class maximum per frame.
        pred_events_high_recall (List[dict]): List of dictionnaries with video, events, fps. Several possible classes per frame.
        pred_scores (dict): Mapping between videos and associated scores.
    """
    classes_inv = {v: k for k, v in classes.items()}

    fps_dict = {}

    for video, _, fps in dataset.videos:
        fps_dict[video] = fps

    err = ErrorStat()
    f1 = ForegroundF1()

    pred_events = []
    pred_events_high_recall = []
    pred_scores = {}
    for video, (scores, support) in sorted(pred_dict.items()):
        label = dataset.get_labels(video)

        # support[support == 0] = 1   # get rid of divide by zero
        if dali:
            assert np.min(support[1:]) > 0, (video, support[1:].tolist())
            scores[1:] /= support[1:, None]
            pred = np.argmax(scores[1:], axis=1)
            err.update(label[1:], pred)
        else:
            assert np.min(support) > 0, (video, support.tolist())
            scores /= support[:, None]
            pred = np.argmax(scores, axis=1)
            err.update(label, pred)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []
        # for i in range(1,pred.shape[0]):
        for i in range(pred.shape[0]):

            if dali:
                f1.update(label[i + 1], pred[i])
            else:
                f1.update(label[i], pred[i])
            if pred[i] != 0:
                if dali:
                    tmp = i + 1
                else:
                    tmp = i
                seconds = int((tmp // fps_dict[video]) % 60)
                minutes = int((tmp // fps_dict[video]) // 60)
                events.append(
                    {
                        "label": classes_inv[pred[i]],
                        "position": int((tmp * 1000) / fps_dict[video]),
                        "gameTime": f"{minutes:02.0f}:{seconds:02.0f}",
                        # 'frame': i,
                        "frame": tmp,
                        "confidence": scores[tmp, pred[i]].item(),
                        # 'score': scores[i, pred[i]].item()
                    }
                )

            for j in classes_inv:
                if dali:
                    tmp = i + 1
                else:
                    tmp = i
                if scores[tmp, j] >= high_recall_score_threshold:
                    # if scores[i, j] >= high_recall_score_threshold:
                    seconds = int((tmp // fps_dict[video]) % 60)
                    minutes = int((tmp // fps_dict[video]) // 60)
                    events_high_recall.append(
                        {
                            "label": classes_inv[j],
                            "position": int((tmp * 1000) / fps_dict[video]),
                            "gameTime": f"{minutes:02.0f}:{seconds:02.0f}",
                            "frame": tmp,
                            # 'frame': i,
                            "confidence": scores[tmp, j].item(),
                            # 'score': scores[i, j].item()
                        }
                    )
        pred_events.append({"video": video, "events": events, "fps": fps_dict[video]})
        pred_events_high_recall.append(
            {"video": video, "events": events_high_recall, "fps": fps_dict[video]}
        )

    return err, f1, pred_events, pred_events_high_recall, pred_scores


def infer_and_process_predictions_e2e(
    model,
    dali,
    dataset,
    split,
    classes,
    save_pred,
    calc_stats=True,
    dataloader_params=None,
    return_pred=False,
):
    """Infer prediction of actions from clips, process these predictions.

    Args:
        model .
        dali (bool): Whether dali has been used or opencv to process videos.
        dataset (Dataset or DaliGenericIterator).
        split (string): Split of the data.
        classes (dict) : Classes associated with indexes.
        save_pred (bool) : Save predictions or not.
        calc_stats (bool) : display stats or not.
            Default: True.
        dataloader_params (dict): Parameters for the dataloader.
            Default: None.
        return_pred (bool): Return dict of predictions or not.
            Default: False

    Returns:
        pred_events_high_recall (List[dict]): List of dictionnaries with video, events, fps. Several possible classes per frame.
        avg_mAP (float): Average mean AP computed for the predictions.
    """
    # print(dataset.)
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32),
        )

    batch_size = dataloader_params.batch_size

    for clip in tqdm(
        dataset
        if dali
        else DataLoader(
            dataset,
            num_workers=dataloader_params.num_workers,
            pin_memory=dataloader_params.pin_memory,
            batch_size=batch_size,
        )
    ):
        if batch_size > 1:
            # Batched by dataloader
            _, batch_pred_scores = model.predict(clip["frame"])
            for i in range(clip["frame"].shape[0]):
                video = clip["video"][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip["start"][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[: end - start, :]
                scores[start:end, :] += pred_scores
                support[start:end] += 1

        else:
            # Batched by dataset
            scores, support = pred_dict[clip["video"][0]]

            start = clip["start"][0].item()
            # start=start-1
            _, pred_scores = model.predict(clip["frame"][0])
            if start < 0:
                pred_scores = pred_scores[:, -start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:, : end - start, :]

            scores[start:end, :] += np.sum(pred_scores, axis=0)
            support[start:end] += pred_scores.shape[0]

    err, f1, pred_events, pred_events_high_recall, pred_scores = (
        process_frame_predictions(dali, dataset, classes, pred_dict)
    )

    avg_mAP = None
    if calc_stats:
        logging.info(f"=== Results on {split} (w/o NMS) ===")
        logging.info(f"Error (frame-level): {err.get() * 100:.2f}")

        def get_f1_tab_row(str_k):
            k = classes[str_k] if str_k != "any" else None
            return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]

        rows = [get_f1_tab_row("any")]
        for c in sorted(classes):
            rows.append(get_f1_tab_row(c))
        header = ["Exact frame", "F1", "TP", "FP", "FN"]
        logging.info(
            tabulate(
                rows, headers=header, floatfmt="0.2f"
            )
        )
        log_table_wandb(name="Confusion matrix table", rows=rows, headers=header)

        mAPs, _ = compute_mAPs_E2E(dataset.labels, pred_events_high_recall)
        avg_mAP = np.mean(mAPs[1:])

    pred_events = build_snpro_prediction_json(pred_events, head_name=dataset.task_name, split=split, created_by="model")
    pred_events_high_recall = build_snpro_prediction_json(pred_events_high_recall, head_name=dataset.task_name, split=split, created_by="model")
    if save_pred is not None:
        store_json(save_pred + ".json", pred_events, pretty=True)
        store_json(save_pred + ".high_recall.json", pred_events_high_recall, pretty=True)
        store_gz_json(save_pred + ".recall.json.gz", pred_events_high_recall)
        # if save_scores:
        #     store_gz_json(save_pred + '.score.json.gz', pred_scores)
    if return_pred:
        return pred_events_high_recall

    logging.info(f"avg_mAP: {avg_mAP}")
    return avg_mAP


def search_best_epoch(work_dir):
    """
    Args:
        work_dir (string): Path in which there is the json file that contains losses for each epoch.

    Returns:
        epoch/epoch_mAP (int): The best epoch.
    """
    loss = load_json(os.path.join(work_dir, "loss.json"))
    valid_mAP = 0
    valid = float("inf")
    epoch = -1
    epoch_mAP = -1
    for epoch_loss in loss:
        if epoch_loss["valid_mAP"] > valid_mAP:
            valid_mAP = epoch_loss["valid_mAP"]
            epoch_mAP = epoch_loss["epoch"]
        if epoch_loss["valid"] < valid:
            valid = epoch_loss["valid"]
            epoch = epoch_loss["epoch"]
    if epoch_mAP != -1:
        return epoch_mAP
    else:
        return epoch


def non_maximum_supression(pred, window):
    """Non maximum suppression for predictions for a window size

    Args:
        pred (List[dict]): List of dictionnaries that contain predictions per video
        window (int): The window size between frames.

    Returns:
        new_pred (List[dict]) : The predictions after the nms.
    """
    new_pred = []
    for video_pred in pred:
        events_by_label = defaultdict(list)
        for e in video_pred["events"]:
            events_by_label[e["label"]].append(e)

        events = []
        for v in events_by_label.values():
            for e1 in v:
                for e2 in v:
                    if (
                        e1["frame"] != e2["frame"]
                        and abs(e1["frame"] - e2["frame"]) <= window
                        and e1["confidence"] < e2["confidence"]
                    ):
                        # Found another prediction in the window that has a
                        # higher score
                        break
                else:
                    events.append(e1)
        events.sort(key=lambda x: x["frame"])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred["events"] = events
        new_video_pred["num_events"] = len(events)
        new_pred.append(new_video_pred)
    return new_pred


# def store_eval_files_json(raw_pred, eval_dir):
#     """
#     Store predictions.

#     If eval_dir ends with .json → store NEW v2 unified file  
#     Else → store OLD per-video SN files (original behavior)
#     """

#     # ==========================================================
#     # 🟢 NEW V2 FORMAT (single json file)
#     # ==========================================================
#     if eval_dir.endswith(".json"):
#         from datetime import datetime

#         data = []

#         for obj in raw_pred:
#             video = obj["video"]
#             fps = obj["fps"]

#             events = []
#             for ev in obj["events"]:
#                 events.append({
#                     "head": "action",  # generic
#                     "label": ev["label"],
#                     "position_ms": ev["position"],
#                     "confidence": ev.get("confidence", 1.0),
#                 })

#             data.append({
#                 "inputs": [{
#                     "type": "video",
#                     "path": video,
#                     "fps": fps,
#                 }],
#                 "events": events,
#             })

#         out = {
#             "version": "2.0",
#             "date": datetime.now().strftime("%Y-%m-%d"),
#             "task": "action_spotting",
#             "metadata": {
#                 "type": "predictions",
#                 "notes": "Generated by model"
#             },
#             "data": data,
#         }

#         os.makedirs(os.path.dirname(eval_dir), exist_ok=True)
#         with open(eval_dir, "w") as f:
#             json.dump(out, f, indent=2)

#         logging.info(f"Stored V2 predictions → {eval_dir}")
#         return True

#     # ==========================================================
#     # 🔵 OLD FORMAT (original code untouched)
#     # ==========================================================
#     only_one_file = False
#     video_pred = defaultdict(list)
#     video_fps = defaultdict(list)

#     for obj in raw_pred:
#         video = obj["video"]
#         video_fps[video] = obj["fps"]

#         for event in obj["events"]:
#             video_pred[video].append(
#                 {
#                     "frame": event["frame"],
#                     "label": event["label"],
#                     "confidence": event["confidence"],
#                     "position": event["position"],
#                     "gameTime": event["gameTime"],
#                 }
#             )

#     for video, pred in video_pred.items():
#         if len(raw_pred) == 1:
#             video_out_dir = eval_dir
#             only_one_file = True
#         else:
#             video_out_dir = os.path.join(eval_dir, os.path.splitext(video)[0])

#         os.makedirs(video_out_dir, exist_ok=True)

#         store_json(
#             os.path.join(video_out_dir, "results_spotting.json"),
#             {"Url": video, "predictions": pred, "fps": video_fps[video]},
#             pretty=True,
#         )

#     logging.info(f"Stored OLD format → {eval_dir}")
#     return only_one_file

def store_eval_files_json(raw_pred, eval_dir, save_v2=True):
    """
    Store predictions.

    Supports:
      - old internal format (list)
      - new v2 format (dict with 'data')

    Store predictions.

    save_v2=True  → store NEW unified v2 json
    save_v2=False → original SN behavior.
    """

    
    # ==========================================================
    # 🟢 NEW V2 FORMAT
    # ==========================================================
    only_one_file = False

    if save_v2:
        from datetime import datetime

        for obj in raw_pred:
            video = obj["video"]
            fps = obj["fps"]

            # directory logic SAME as old
            if len(raw_pred) == 1:
                video_out_dir = eval_dir
                only_one_file = True
            else:
                video_out_dir = os.path.join(eval_dir, os.path.splitext(video)[0])

            os.makedirs(video_out_dir, exist_ok=True)

            events = []
            for ev in obj["events"]:
                events.append({
                    "head": "action",
                    "label": ev.get("label"),
                    "frame": ev.get("frame"),
                    "position_ms": int(ev.get("position")),
                    "confidence": float(ev.get("confidence")),
                    "gameTime": ev.get("gameTime")
                })

            out = {
                "version": "2.0",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "task": "action_spotting",
                "metadata": {"type": "predictions"},
                "data": [{
                    "inputs": [{
                        "type": "video",
                        "path": video,
                        "fps": fps,
                    }],
                    "events": events,
                }],
            }

            out_path = os.path.join(video_out_dir, "results_spotting.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

        logging.info(f"Stored V2 predictions → {eval_dir}")
        return only_one_file


    # ==========================================================
    # 🔵 OLD FORMAT (UNCHANGED)
    # ==========================================================
    only_one_file = False
    video_pred = defaultdict(list)
    video_fps = defaultdict(list)

    for obj in raw_pred:
        video = obj["video"]
        video_fps[video] = obj["fps"]

        for event in obj["events"]:
            video_pred[video].append(
                {
                    "frame": event["frame"],
                    "label": event["label"],
                    "confidence": event["confidence"],
                    "position": event["position"],
                    "gameTime": event["gameTime"],
                }
            )

    for video, pred in video_pred.items():
        if len(raw_pred) == 1:
            video_out_dir = eval_dir
            only_one_file = True
        else:
            video_out_dir = os.path.join(eval_dir, os.path.splitext(video)[0])

        os.makedirs(video_out_dir, exist_ok=True)

        store_json(
            os.path.join(video_out_dir, "results_spotting.json"),
            {"Url": video, "predictions": pred, "fps": video_fps[video]},
            pretty=True,
        )

    logging.info(f"Stored OLD format → {eval_dir}")
    return only_one_file




def label2vector(
    labels, num_classes=17, framerate=2, EVENT_DICTIONARY={}, vector_size=None
):
    """Transform list of dict containing ground truth labels into a 2D array.
    Args:
        labels (List[dict]): List of groundtruth labels for a video.
        num_classes (int): Number of classes.
            Default: 17.
        framerate (int): Rate at which the frames have been processed in a video.
            Default: 2.
        EVENT_DICTIONARY (dict): Mapping between classes and indexes.
            Default: {}.
        vector_size (int): Size of the returned vector.
            Default: None.

    Returns:
        dense_labels (Numpy array): Array of shape (vector_size, num_classes) containing a value.
            The first index is the frame number and the second one is the class. The value can be -1 is the event
            is unshown, 1 is the event is visible and 0 otherwise, meaning that the event do not occur for this frame.

    """
    vector_size = int(90 * 60 * framerate if vector_size is None else vector_size)

    dense_labels = np.zeros((vector_size, num_classes))

    for annotation in labels:

        #print(annotation)
        event = annotation["label"]
        if "frame" in annotation:
            frame = int(annotation["frame"])

        else:
            time = annotation["gameTime"]

            minutes = int(time[-5:-3])
            seconds = int(time[-2::])
            # annotation at millisecond precision
            if "position" in annotation:
                frame = int(framerate * (int(annotation["position"]) / 1000))
            # annotation at second precision
            else:
                frame = framerate * (seconds + 60 * minutes)

        label = EVENT_DICTIONARY[event]

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        frame = min(frame, vector_size - 1)
        dense_labels[frame][label] = value

    return dense_labels


def predictions2vector(
    predictions, num_classes=17, framerate=2, EVENT_DICTIONARY={}, vector_size=None
):
    """Transform list of dict containing predictions into a 2D array.
    Args:
        predictions (List[dict]): List of predictions for a video.
        num_classes (int): Number of classes.
            Default: 17.
        framerate (int): Rate at which the frames have been processed in a video.
            Default: 2.
        EVENT_DICTIONARY (dict): Mapping between classes and indexes.
            Default: {}.
        vector_size (int): Size of the returned vector.
            Default: None.

    Returns:
        dense_predictions (Numpy array): Array of shape (vector_size, num_classes) containing a value.
            The first index is the frame number and the second one is the class. The value is the score of the class.

    """
    vector_size = int(90 * 60 * framerate if vector_size is None else vector_size)

    dense_predictions = np.zeros((vector_size, num_classes)) - 1

    for annotation in predictions:

        event = annotation["label"]

        if "frame" in annotation:
            frame = int(annotation["frame"])
        else:
            time = int(annotation["position"])

            # half = int(annotation["half"])

            frame = int(framerate * (time / 1000))

        label = EVENT_DICTIONARY[event]

        frame = min(frame, vector_size - 1)
        dense_predictions[frame][label] = annotation["confidence"]

    return dense_predictions


np.seterr(divide="ignore", invalid="ignore")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_class_scores(target, closest, detection, delta):
    """Compute the scores for a single class.

    Args:
        target (np.array(vector_size): List of ground truth targets of shape (number of frames).
        closest (np.array(vector_size - 1): List of closest action index of shape (number of frames - 1).
        detection (np.array(vector_size): List of predictions of shape (number of frames).
        delta (int): Tolerance.

    Returns:
        game_detections (np.array(number of predictions,3): Array to save the scores with the first column being
        the prediction score, the second one being 1 if an event have been found with requirements, the third one being
        the score of the closest action.
        len(gt_indexes_visible) (int): number of visible events.
        len(gt_indexes_unshown) (int): number of unshown events.

    """

    # Retrieving the important variables
    gt_indexes = np.where(target != 0)[0]
    gt_indexes_visible = np.where(target > 0)[0]
    gt_indexes_unshown = np.where(target < 0)[0]
    pred_indexes = np.where(detection >= 0)[0]
    pred_scores = detection[pred_indexes]

    # Array to save the results, each is [pred_scor,{1 or 0},closest[pred_indexes]]
    game_detections = np.zeros((len(pred_indexes), 3))
    game_detections[:, 0] = np.copy(pred_scores)
    game_detections[:, 2] = np.copy(closest[pred_indexes])

    remove_indexes = list()

    for gt_index in gt_indexes:

        max_score = -1
        max_index = None
        game_index = 0
        selected_game_index = 0

        for pred_index, pred_score in zip(pred_indexes, pred_scores):

            if pred_index < gt_index - delta:
                game_index += 1
                continue
            if pred_index > gt_index + delta:
                break

            if (
                abs(pred_index - gt_index) <= delta / 2
                and pred_score > max_score
                and pred_index not in remove_indexes
            ):
                max_score = pred_score
                max_index = pred_index
                selected_game_index = game_index
            game_index += 1

        if max_index is not None:
            game_detections[selected_game_index, 1] = 1
            remove_indexes.append(max_index)

    return game_detections, len(gt_indexes_visible), len(gt_indexes_unshown)


def compute_precision_recall_curve(targets, closests, detections, delta):
    """Compute precision recall curve.

    Args:
        target (List(np.array(vector_size,num_classes)): List of ground truth targets of shape (number of videos, number of frames,number of classes).
        closest (List(np.array(vector_size - 1,num_classes)): List of closest action index of shape (number of videos, number of frames - 1,number of classes).
        detection (List(np.array(vector_size,num_classes)): List of predictions of shape (number of videos, number of frames,number of classes).
        delta (int): Tolerance.

    Returns:
        precision: List of precision points.
        recall: List of recall points.
        precision_visible: List of precision points for only the visible events.
        recall_visible: List of recall points for only the visible events.
        precision_unshown: List of precision points for only the unshown events.
        recall_unshown: List of recall points for only the unshown events.

    """

    # Store the number of classes
    num_classes = targets[0].shape[-1]

    # 200 confidence thresholds between [0,1]
    thresholds = np.linspace(0, 1, 200)

    # Store the precision and recall points
    precision = list()
    recall = list()
    precision_visible = list()
    recall_visible = list()
    precision_unshown = list()
    recall_unshown = list()

    # Apply Non-Maxima Suppression if required
    start = time.time()

    # Precompute the predictions scores and their correspondence {TP, FP} for each class
    for c in np.arange(num_classes):
        total_detections = np.zeros((1, 3))
        total_detections[0, 0] = -1
        n_gt_labels_visible = 0
        n_gt_labels_unshown = 0

        # Get the confidence scores and their corresponding TP or FP characteristics for each game
        for target, closest, detection in zip(targets, closests, detections):
            tmp_detections, tmp_n_gt_labels_visible, tmp_n_gt_labels_unshown = (
                compute_class_scores(
                    target[:, c], closest[:, c], detection[:, c], delta
                )
            )
            total_detections = np.append(total_detections, tmp_detections, axis=0)
            n_gt_labels_visible = n_gt_labels_visible + tmp_n_gt_labels_visible
            n_gt_labels_unshown = n_gt_labels_unshown + tmp_n_gt_labels_unshown

        precision.append(list())
        recall.append(list())
        precision_visible.append(list())
        recall_visible.append(list())
        precision_unshown.append(list())
        recall_unshown.append(list())

        # Get only the visible or unshown actions
        total_detections_visible = np.copy(total_detections)
        total_detections_unshown = np.copy(total_detections)
        total_detections_visible[
            np.where(total_detections_visible[:, 2] <= 0.5)[0], 0
        ] = -1
        total_detections_unshown[
            np.where(total_detections_unshown[:, 2] >= -0.5)[0], 0
        ] = -1

        # Get the precision and recall for each confidence threshold
        for threshold in thresholds:
            pred_indexes = np.where(total_detections[:, 0] >= threshold)[0]
            pred_indexes_visible = np.where(
                total_detections_visible[:, 0] >= threshold
            )[0]
            pred_indexes_unshown = np.where(
                total_detections_unshown[:, 0] >= threshold
            )[0]
            TP = np.sum(total_detections[pred_indexes, 1])
            TP_visible = np.sum(total_detections[pred_indexes_visible, 1])
            TP_unshown = np.sum(total_detections[pred_indexes_unshown, 1])
            p = np.nan_to_num(TP / len(pred_indexes))
            r = np.nan_to_num(TP / (n_gt_labels_visible + n_gt_labels_unshown))
            p_visible = np.nan_to_num(TP_visible / len(pred_indexes_visible))
            r_visible = np.nan_to_num(TP_visible / n_gt_labels_visible)
            p_unshown = np.nan_to_num(TP_unshown / len(pred_indexes_unshown))
            r_unshown = np.nan_to_num(TP_unshown / n_gt_labels_unshown)
            precision[-1].append(p)
            recall[-1].append(r)
            precision_visible[-1].append(p_visible)
            recall_visible[-1].append(r_visible)
            precision_unshown[-1].append(p_unshown)
            recall_unshown[-1].append(r_unshown)

    precision = np.array(precision).transpose()
    recall = np.array(recall).transpose()
    precision_visible = np.array(precision_visible).transpose()
    recall_visible = np.array(recall_visible).transpose()
    precision_unshown = np.array(precision_unshown).transpose()
    recall_unshown = np.array(recall_unshown).transpose()

    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall[:, i])
        precision[:, i] = precision[index_sort, i]
        recall[:, i] = recall[index_sort, i]

    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall_visible[:, i])
        precision_visible[:, i] = precision_visible[index_sort, i]
        recall_visible[:, i] = recall_visible[index_sort, i]

    # Sort the points based on the recall, class per class
    for i in np.arange(num_classes):
        index_sort = np.argsort(recall_unshown[:, i])
        precision_unshown[:, i] = precision_unshown[index_sort, i]
        recall_unshown[:, i] = recall_unshown[index_sort, i]

    return (
        precision,
        recall,
        precision_visible,
        recall_visible,
        precision_unshown,
        recall_unshown,
    )


def compute_mAP(precision, recall):
    """Compute mean AP per class from precision and recall points

    Args:
        precision: List of precision points.
        recall: List of recall points.

    Returns:
        np.mean(mAP_per_class): mean of the mAP over all classes.
        mAP_per_class: List of mAP for each class.

    """
    # Array for storing the AP per class
    AP = np.array([0.0] * precision.shape[-1])

    # Loop for all classes
    for i in np.arange(precision.shape[-1]):

        # 11 point interpolation
        for j in np.arange(11) / 10:

            index_recall = np.where(recall[:, i] >= j)[0]

            possible_value_precision = precision[index_recall, i]
            max_value_precision = 0

            if possible_value_precision.shape[0] != 0:
                max_value_precision = np.max(possible_value_precision)

            AP[i] += max_value_precision

    mAP_per_class = AP / 11

    return np.mean(mAP_per_class), mAP_per_class


# Tight: (SNv3): np.arange(5)*1 + 1
# Loose: (SNv1/v2): np.arange(12)*5 + 5


def delta_curve(targets, closests, detections, framerate, deltas=np.arange(5) * 1 + 1):
    """Compute lists of mAP for each tolerance.

    Args:
        targets (List(np.array(vector_size,num_classes)): List of ground truth targets of shape (number of videos, number of frames,number of classes).
        closests (List(np.array(vector_size - 1,num_classes)): List of closest action index of shape (number of videos, number of frames - 1,number of classes).
        detections (List(np.array(vector_size,num_classes)): List of predictions of shape (number of videos, number of frames,number of classes).
        delta (np.array): Tolerances.

    Returns:
        mAP: List of mean mAP over all the classes for each tolerance.
        mAP_per_class: List of list of mAP for all classes for each tolerance.
        mAP_visible: List of mean mAP over all the classes for each tolerance only for the visible events.
        mAP_per_class_visible: List of list of mAP for all classes for each tolerance only for the visible events.
        mAP_unshown: List of mean mAP over all the classes for each tolerance only for the unshown events.
        mAP_per_class_unshown: List of list of mAP for all classes for each tolerance only for the unshown events.

    """
    mAP = list()
    mAP_per_class = list()
    mAP_visible = list()
    mAP_per_class_visible = list()
    mAP_unshown = list()
    mAP_per_class_unshown = list()

    for delta in tqdm(deltas * framerate):

        (
            precision,
            recall,
            precision_visible,
            recall_visible,
            precision_unshown,
            recall_unshown,
        ) = compute_precision_recall_curve(targets, closests, detections, delta)

        tmp_mAP, tmp_mAP_per_class = compute_mAP(precision, recall)
        mAP.append(tmp_mAP)
        mAP_per_class.append(tmp_mAP_per_class)
        # TODO: compute visible/undshown from another JSON file containing only the visible/unshown annotations
        tmp_mAP_visible, tmp_mAP_per_class_visible = compute_mAP(
            precision_visible, recall_visible
        )
        mAP_visible.append(tmp_mAP_visible)
        mAP_per_class_visible.append(tmp_mAP_per_class_visible)
        tmp_mAP_unshown, tmp_mAP_per_class_unshown = compute_mAP(
            precision_unshown, recall_unshown
        )
        mAP_unshown.append(tmp_mAP_unshown)
        mAP_per_class_unshown.append(tmp_mAP_per_class_unshown)

    return (
        mAP,
        mAP_per_class,
        mAP_visible,
        mAP_per_class_visible,
        mAP_unshown,
        mAP_per_class_unshown,
    )


def average_mAP(
    targets, detections, closests, framerate=2, deltas=np.arange(5) * 1 + 1
):
    """Compute average mAP.

    Args:
        targets (List(np.array(vector_size,num_classes)): List of ground truth targets of shape (number of videos, number of frames,number of classes).
        closests (List(np.array(vector_size - 1,num_classes)): List of closest action index of shape (number of videos, number of frames - 1,number of classes).
        detections (List(np.array(vector_size,num_classes)): List of predictions of shape (number of videos, number of frames,number of classes).
        framerate (int).
            Default: 2.
        delta (np.array): Tolerances.

    Returns:
        a_mAP: average mAP.
        a_mAP_per_class: List of average mAP for all classes.
        a_mAP_visible: average mAP only for the visible events.
        a_mAP_per_class_visible: List of average mAP only for the visible events.
        a_mAP_unshown: average mAP only for the unshown events.
        a_mAP_per_class_unshown: List of average mAP only for the unshown events.

    """
    (
        mAP,
        mAP_per_class,
        mAP_visible,
        mAP_per_class_visible,
        mAP_unshown,
        mAP_per_class_unshown,
    ) = delta_curve(targets, closests, detections, framerate, deltas)

    if len(mAP) == 1:
        return (
            mAP[0],
            mAP_per_class[0],
            mAP_visible[0],
            mAP_per_class_visible[0],
            mAP_unshown[0],
            mAP_per_class_unshown[0],
        )

    # Compute the average mAP
    integral = 0.0
    for i in np.arange(len(mAP) - 1):
        integral += (mAP[i] + mAP[i + 1]) / 2
    a_mAP = integral / ((len(mAP) - 1))

    integral_visible = 0.0
    for i in np.arange(len(mAP_visible) - 1):
        integral_visible += (mAP_visible[i] + mAP_visible[i + 1]) / 2
    a_mAP_visible = integral_visible / ((len(mAP_visible) - 1))

    integral_unshown = 0.0
    for i in np.arange(len(mAP_unshown) - 1):
        integral_unshown += (mAP_unshown[i] + mAP_unshown[i + 1]) / 2
    a_mAP_unshown = integral_unshown / ((len(mAP_unshown) - 1))
    a_mAP_unshown = a_mAP_unshown * 17 / 13

    a_mAP_per_class = list()
    for c in np.arange(len(mAP_per_class[0])):
        integral_per_class = 0.0
        for i in np.arange(len(mAP_per_class) - 1):
            integral_per_class += (mAP_per_class[i][c] + mAP_per_class[i + 1][c]) / 2
        a_mAP_per_class.append(integral_per_class / ((len(mAP_per_class) - 1)))

    a_mAP_per_class_visible = list()
    for c in np.arange(len(mAP_per_class_visible[0])):
        integral_per_class_visible = 0.0
        for i in np.arange(len(mAP_per_class_visible) - 1):
            integral_per_class_visible += (
                mAP_per_class_visible[i][c] + mAP_per_class_visible[i + 1][c]
            ) / 2
        a_mAP_per_class_visible.append(
            integral_per_class_visible / ((len(mAP_per_class_visible) - 1))
        )

    a_mAP_per_class_unshown = list()
    for c in np.arange(len(mAP_per_class_unshown[0])):
        integral_per_class_unshown = 0.0
        for i in np.arange(len(mAP_per_class_unshown) - 1):
            integral_per_class_unshown += (
                mAP_per_class_unshown[i][c] + mAP_per_class_unshown[i + 1][c]
            ) / 2
        a_mAP_per_class_unshown.append(
            integral_per_class_unshown / ((len(mAP_per_class_unshown) - 1))
        )

    return (
        a_mAP,
        a_mAP_per_class,
        a_mAP_visible,
        a_mAP_per_class_visible,
        a_mAP_unshown,
        a_mAP_per_class_unshown,
    )


def LoadJsonFromZip(zippedFile, JsonPath):
    with zipfile.ZipFile(zippedFile, "r") as z:
        # print(filename)
        with z.open(JsonPath) as f:
            data = f.read()
            d = json.loads(data.decode("utf-8"))

    return d


def get_closest_action_index(dense_labels, closest_numpy):
    """Assign the action to a list of indexes based on the previous and the next indexes.
    Example:
        If action occurs at index 10, the previous index at which the actions occured is 6 and the next one is 12,
        the action will then be assigned from index 8 to 11.

    Args:
        dense_labels: Vector of groundtruth labels.
        closest_numpy: Empty vector that stores the labels at the closest indexes.

    Returns:
        closest_numpy: Vector that stores the labels at the closest indexes.
    """
    for c in np.arange(dense_labels.shape[-1]):
        indexes = np.where(dense_labels[:, c] != 0)[0].tolist()
        if len(indexes) == 0:
            continue
        indexes.insert(0, -indexes[0])
        indexes.append(2 * closest_numpy.shape[0])
        for i in np.arange(len(indexes) - 2) + 1:
            start = max(0, (indexes[i - 1] + indexes[i]) // 2)
            stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
            closest_numpy[start:stop, c] = dense_labels[indexes[i], c]
    return closest_numpy


def compute_performances_mAP(
    metric, targets_numpy, detections_numpy, closests_numpy, INVERSE_EVENT_DICTIONARY
):
    """Compute the different mAP and display them.

    Args:
        metric (string): The metric that will be used to compute the mAP.
        targets_numpy (List(np.array(vector_size,num_classes)): List of ground truth targets of shape (number of videos, number of frames,number of classes).
        detections_numpy (List(np.array(vector_size - 1,num_classes)): List of closest action index of shape (number of videos, number of frames - 1,number of classes).
        closests_numpy (List(np.array(vector_size,num_classes)): List of predictions of shape (number of videos, number of frames,number of classes).
        INVERSE_EVENT_DICTIONARY (dict): mapping between indexes and class name.

    Returns:
        results (dict): Dictionnary containing the different mAP computed.
    """
    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])

    # Compute the performances
    (
        a_mAP,
        a_mAP_per_class,
        a_mAP_visible,
        a_mAP_per_class_visible,
        a_mAP_unshown,
        a_mAP_per_class_unshown,
    ) = average_mAP(
        targets_numpy, detections_numpy, closests_numpy, framerate=2, deltas=deltas
    )

    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible,
        "a_mAP_per_class_visible": a_mAP_per_class_visible,
        "a_mAP_unshown": a_mAP_unshown,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown,
    }

    rows = []
    for i in range(len(results["a_mAP_per_class"])):
        label = INVERSE_EVENT_DICTIONARY[i]
        rows.append(
            (
                label,
                "{:0.2f}".format(results["a_mAP_per_class"][i] * 100),
                "{:0.2f}".format(results["a_mAP_per_class_visible"][i] * 100),
                "{:0.2f}".format(results["a_mAP_per_class_unshown"][i] * 100),
            )
        )
    rows.append(
        (
            "Average mAP",
            "{:0.2f}".format(results["a_mAP"] * 100),
            "{:0.2f}".format(results["a_mAP_visible"] * 100),
            "{:0.2f}".format(results["a_mAP_unshown"] * 100),
        )
    )

    header = ["", "Any", "Visible", "Unseen"]
    result = tabulate(rows, headers=header)
    log_table_wandb(name=f"Final Scores (Metric : {metric})", rows=rows, headers=header)
    logging.info("Best Performance at end of training ")
    logging.info("Metric: " + metric)
    logging.info("\n" + result)
    # print(tabulate(rows, headers=['', 'Any', 'Visible', 'Unseen']))
    return result

np.seterr(divide="ignore", invalid="ignore")
