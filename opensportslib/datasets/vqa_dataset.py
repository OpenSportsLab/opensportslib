"""VQA dataset adapter for canonical OpenSportsLib JSON annotations."""

from __future__ import annotations

import json
import os
import pickle
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from opensportslib.core.config.accessors import (
    get_split_annotation_path,
    get_split_source_path,
    get_vqa_feature_source,
    get_vqa_xvars_feature_mode,
    get_xvars_train_video_token_len,
)
from opensportslib.models.utils.xvars_clip_index import (
    build_xvars_prior_from_prediction,
    load_feature_index,
    load_prediction_index,
)
from opensportslib.models.utils.vqa_xvars_features import validate_xvars_feature_tensor


class VQADataset(Dataset):
    """Flatten VQA annotations into single question-answer training samples."""

    def __init__(self, config, annotation_file: str | None = None, split: str = "train"):
        self.config = config
        self.split = split
        self._train_execution = self._as_dict(getattr(getattr(config, "TRAIN", None), "execution", None))
        self._view_policy = str(self._train_execution.get("view_sampling_policy", "random_train_deterministic_eval")).lower()
        self._rng = random.Random(42)
        self.annotation_path = annotation_file or get_split_annotation_path(config, split)
        if not self.annotation_path:
            raise ValueError(
                f"Missing annotation path for split '{split}'. "
                f"Expected DATA.common.splits.{split}.annotation_path."
            )
        self.annotation_path = os.path.abspath(os.path.expanduser(self.annotation_path))
        with open(self.annotation_path, encoding="utf-8") as f:
            payload = json.load(f)

        data = payload.get("data", [])
        source_root = get_split_source_path(config, split) or ""
        source_root = os.path.abspath(os.path.expanduser(source_root)) if source_root else ""
        common = self._as_dict(self._as_dict(getattr(config, "DATA", None)).get("common"))
        feature_index_path = str(common.get("feature_index") or "").strip()
        prediction_index_path = str(common.get("prediction_index") or "").strip()
        feature_backend = str(self._train_execution.get("feature_backend", "xvars_clip")).lower()
        feature_source = get_vqa_feature_source(config, default="indexed")
        if feature_backend != "xvars_clip":
            raise ValueError(f"Unsupported VQA feature backend '{feature_backend}'. Expected 'xvars_clip'.")
        require_feature_index = feature_source in {"indexed", ""}
        if require_feature_index and not feature_index_path:
            raise ValueError("Missing required config key DATA.common.feature_index for VQA xvars_clip mode.")
        self.feature_source = feature_source
        self.feature_mode = get_vqa_xvars_feature_mode(config, default="strict_xvars")
        self.expected_feature_tokens = get_xvars_train_video_token_len(config)
        self.feature_index = (
            load_feature_index(os.path.abspath(os.path.expanduser(feature_index_path)), split=split)
            if feature_index_path
            else {}
        )
        self.prediction_index = (
            load_prediction_index(os.path.abspath(os.path.expanduser(prediction_index_path)), split=split)
            if prediction_index_path
            else {}
        )

        self.samples: list[dict[str, Any]] = []
        for item in data:
            item_id = item.get("id")
            item_id_str = str(item_id)
            labels = item.get("labels", {})
            metadata = item.get("metadata", {})
            inputs = item.get("inputs", [])
            video_path = None
            for inp in inputs:
                if str(inp.get("type", "")).lower() == "video":
                    rel = inp.get("path")
                    if rel:
                        video_path = os.path.join(source_root, rel) if source_root and not os.path.isabs(rel) else rel
                        break

            feature_candidates = self.feature_index.get(item_id_str, [])
            if require_feature_index and not feature_candidates:
                raise ValueError(
                    f"Missing feature index entry for sample id '{item_id_str}'. "
                    "Provide DATA.common.feature_index mapping with feature_paths or feature_dir/path."
                )
            pred_row = self.prediction_index.get(item_id_str, {})
            for qa in item.get("answers", []):
                question = qa.get("question")
                refs = qa.get("answers", []) or []
                if not question:
                    continue
                prior_prediction_text = build_xvars_prior_from_prediction(pred_row)
                self.samples.append(
                    {
                        "id": item_id,
                        "question": question,
                        "references": refs,
                        "video_path": video_path,
                        "feature_candidates": list(feature_candidates),
                        "feature_source": self.feature_source,
                        "labels": labels,
                        "metadata": metadata,
                        "prediction": pred_row,
                        "prior_prediction_text": prior_prediction_text,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.samples[idx])
        feature_path = self._choose_feature_path(sample.get("feature_candidates") or [])
        if not feature_path or not os.path.exists(feature_path):
            if self.feature_source in {"raw_video", "auto"}:
                sample["selected_feature_path"] = None
                sample["video_spatio_temporal_features"] = None
                return sample
            raise FileNotFoundError(
                f"Missing CLIP feature file for sample '{sample.get('id')}'. "
                f"Selected path: {feature_path!r}"
            )
        with open(feature_path, "rb") as f:
            raw = pickle.load(f)
        features = validate_xvars_feature_tensor(
            torch.as_tensor(raw, dtype=torch.float32),
            expected_tokens=self.expected_feature_tokens,
            context=f"X-VARS features for sample '{sample.get('id')}'",
        )
        sample["selected_feature_path"] = feature_path
        sample["video_spatio_temporal_features"] = features
        return sample

    def _choose_feature_path(self, candidates: list[str]) -> str:
        existing = [p for p in candidates if p and os.path.exists(p)]
        if not existing:
            return ""
        if self.split == "train" and self._view_policy == "random_train_deterministic_eval":
            return existing[self._rng.randint(0, len(existing) - 1)]
        return existing[0]

    @staticmethod
    def _as_dict(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return vars(obj)
        return {}
