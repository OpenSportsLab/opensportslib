"""VQA dataset adapter for canonical OpenSportsLib JSON annotations."""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from opensportslib.core.config.accessors import (
    get_vqa_backend,
    get_split_annotation_path,
    get_split_source_path,
    get_vqa_feature_source,
    get_vqa_xvars_feature_mode,
    get_xvars_train_video_token_len,
)
from opensportslib.models.utils.vqa_prediction_priors import build_prediction_prior_text
from opensportslib.models.utils.xvars_clip_index import load_feature_index, load_prediction_index, validate_xvars_feature_tensor

logger = logging.getLogger(__name__)


class VQADataset(Dataset):
    """Flatten VQA annotations into single question-answer training samples."""

    def __init__(self, config, annotation_file: str | None = None, split: str = "train"):
        self.config = config
        self.split = split
        self._train_execution = self._as_dict(getattr(getattr(config, "TRAIN", None), "execution", None))
        self._prompt_cfg = self._as_dict(self._train_execution.get("prompt"))
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
        label_space = []
        payload_labels = payload.get("labels", {})
        if isinstance(payload_labels, dict):
            action_spec = payload_labels.get("action", {})
            if isinstance(action_spec, dict):
                label_space = [str(label).strip() for label in action_spec.get("labels", []) if str(label).strip()]
        source_root = get_split_source_path(config, split) or ""
        source_root = (
            os.path.abspath(os.path.expanduser(source_root))
            if source_root
            else os.path.dirname(self.annotation_path)
        )
        common = self._as_dict(self._as_dict(getattr(config, "DATA", None)).get("common"))
        feature_index_path = str(common.get("feature_index") or "").strip()
        prediction_index_path = str(common.get("prediction_index") or "").strip()
        backend = get_vqa_backend(config)
        self.backend = backend
        self.native_vl = backend == "qwen_vl_native_infer"
        feature_backend = str(self._train_execution.get("feature_backend", "xvars_clip")).lower()
        feature_source = get_vqa_feature_source(config, default="indexed")
        if not self.native_vl and feature_backend != "xvars_clip":
            raise ValueError(f"Unsupported VQA feature backend '{feature_backend}'. Expected 'xvars_clip'.")
        strict_feature_index = (not self.native_vl) and feature_source in {"indexed", ""}
        fallback_feature_index = (not self.native_vl) and feature_source in {"indexed_or_raw", "indexed_or_raw_clip"}
        if strict_feature_index and not feature_index_path:
            raise ValueError("Missing required config key DATA.common.feature_index for VQA xvars_clip mode.")
        self.feature_source = feature_source
        self.feature_mode = get_vqa_xvars_feature_mode(config, default="strict_xvars")
        self.expected_feature_tokens = get_xvars_train_video_token_len(config)
        self.feature_index = self._load_feature_index(
            feature_index_path,
            split=split,
            strict=strict_feature_index,
            allow_missing=fallback_feature_index,
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
            allowed_labels = [str(label).strip() for label in item.get("allowed_labels", []) if str(label).strip()] or label_space
            video_path = self._resolve_video_path(item, source_root)
            frame_paths = self._resolve_frame_paths(item, source_root)

            feature_candidates = self.feature_index.get(item_id_str, [])
            if strict_feature_index and not feature_candidates:
                raise ValueError(
                    f"Missing feature index entry for sample id '{item_id_str}'. "
                    "Provide DATA.common.feature_index mapping with feature_paths or feature_dir/path."
                )
            pred_row = self.prediction_index.get(item_id_str, {})
            qa_rows = self._iter_qa_rows(item)
            for qa in qa_rows:
                question = qa["question"]
                refs = qa["references"]
                prior_prediction_text = build_prediction_prior_text(
                    pred_row,
                    adapter=self._prompt_cfg.get("prediction_prior_adapter"),
                    fields=self._prompt_cfg.get("prediction_prior_fields"),
                )
                self.samples.append(
                    {
                        "id": qa.get("sample_id", item_id),
                        "question": question,
                        "references": refs,
                        "video_path": video_path,
                        "frame_paths": frame_paths,
                        "video_frames": list(item.get("video_frames") or []),
                        "feature_candidates": list(feature_candidates),
                        "feature_source": self.feature_source,
                        "labels": labels,
                        "metadata": metadata,
                        "prediction": pred_row,
                        "prior_prediction_text": prior_prediction_text,
                        "ground_truth_label": qa["ground_truth_label"],
                        "allowed_labels": list(allowed_labels),
                    }
                )
        if self.native_vl:
            for sample in self.samples:
                if sample.get("video_path") or sample.get("frame_paths") or sample.get("video_frames"):
                    continue
                raise ValueError(
                    f"Native Qwen VL sample '{sample.get('id')}' is missing visual input. "
                    "Expected video_path, frame_paths, or video_frames."
                )

    def _load_feature_index(
        self,
        feature_index_path: str,
        *,
        split: str,
        strict: bool,
        allow_missing: bool,
    ) -> dict[str, list[str]]:
        if not feature_index_path:
            return {}

        resolved_path = os.path.abspath(os.path.expanduser(feature_index_path))
        if allow_missing and not os.path.exists(resolved_path):
            logger.warning(
                "VQA feature index unavailable; falling back to raw-video extraction | "
                "feature_source=%s | split=%s | feature_index=%s",
                self.feature_source,
                split,
                resolved_path,
            )
            return {}

        try:
            return load_feature_index(resolved_path, split=split)
        except Exception:
            if strict or not allow_missing:
                raise
            logger.warning(
                "VQA feature index unreadable; falling back to raw-video extraction | "
                "feature_source=%s | split=%s | feature_index=%s",
                self.feature_source,
                split,
                resolved_path,
                exc_info=True,
            )
            return {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.samples[idx])
        if self.native_vl:
            sample["selected_feature_path"] = None
            sample["video_spatio_temporal_features"] = None
            return sample
        if self.feature_source == "raw_video":
            sample["selected_feature_path"] = None
            sample["video_spatio_temporal_features"] = None
            return sample
        feature_path = self._choose_feature_path(sample.get("feature_candidates") or [])
        if not feature_path or not os.path.exists(feature_path):
            if self.feature_source in {"raw_video", "auto", "indexed_or_raw", "indexed_or_raw_clip"}:
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
    def _resolve_video_path(item: dict[str, Any], source_root: str) -> str | None:
        direct_path = item.get("video_path")
        if direct_path:
            path = str(direct_path)
            return os.path.join(source_root, path) if source_root and not os.path.isabs(path) else path

        inputs = item.get("inputs", [])
        for inp in inputs:
            input_type = str(inp.get("type", "")).lower()
            if input_type not in {"video", "frames_npy", "frames"}:
                continue
            rel = inp.get("path")
            if rel:
                rel = str(rel)
                return os.path.join(source_root, rel) if source_root and not os.path.isabs(rel) else rel
        return None

    @staticmethod
    def _resolve_frame_paths(item: dict[str, Any], source_root: str) -> list[str]:
        frame_paths = item.get("frame_paths") or []
        if frame_paths:
            out = []
            for path in frame_paths:
                path = str(path)
                out.append(os.path.join(source_root, path) if source_root and not os.path.isabs(path) else path)
            return out

        inputs = item.get("inputs", [])
        for inp in inputs:
            input_type = str(inp.get("type", "")).lower()
            if input_type not in {"image", "images", "frame", "frames"}:
                continue
            path = inp.get("path")
            if path:
                path = str(path)
                return [os.path.join(source_root, path) if source_root and not os.path.isabs(path) else path]
            paths = inp.get("paths") or []
            out = []
            for subpath in paths:
                subpath = str(subpath)
                out.append(os.path.join(source_root, subpath) if source_root and not os.path.isabs(subpath) else subpath)
            if out:
                return out
        return []

    @staticmethod
    def _iter_qa_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        direct_question = str(item.get("question", "")).strip()
        if direct_question:
            refs = item.get("references", []) or []
            if isinstance(refs, str):
                refs = [refs]
            ground_truth_label = str(item.get("ground_truth_label", "")).strip()
            if not ground_truth_label and refs:
                ground_truth_label = str(refs[0]).strip()
            rows.append(
                {
                    "sample_id": item.get("id"),
                    "question": direct_question,
                    "references": [str(ref).strip() for ref in refs if str(ref).strip()],
                    "ground_truth_label": ground_truth_label,
                }
            )
            return rows

        direct_questions = item.get("questions", []) or []
        if isinstance(direct_questions, list) and direct_questions:
            for idx, question in enumerate(direct_questions):
                question = str(question).strip()
                if not question:
                    continue
                rows.append(
                    {
                        "sample_id": f"{item.get('id')}:{idx}",
                        "question": question,
                        "references": [],
                        "ground_truth_label": str(item.get("ground_truth_label", "")).strip(),
                    }
                )
            if rows:
                return rows

        for qa in item.get("answers", []):
            question = str(qa.get("question", "")).strip()
            refs = qa.get("answers", []) or []
            if isinstance(refs, str):
                refs = [refs]
            if not question:
                continue
            ground_truth_label = str(item.get("ground_truth_label", "")).strip()
            if not ground_truth_label and refs:
                ground_truth_label = str(refs[0]).strip()
            rows.append(
                {
                    "sample_id": item.get("id"),
                    "question": question,
                    "references": [str(ref).strip() for ref in refs if str(ref).strip()],
                    "ground_truth_label": ground_truth_label,
                }
            )
        return rows

    @staticmethod
    def _as_dict(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return vars(obj)
        return {}
