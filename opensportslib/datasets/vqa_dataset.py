"""VQA dataset adapter for canonical OpenSportsLib JSON annotations."""

from __future__ import annotations

import json
import os
from typing import Any

from torch.utils.data import Dataset

from opensportslib.core.config.accessors import get_split_annotation_path, get_split_source_path


class VQADataset(Dataset):
    """Flatten VQA annotations into single question-answer training samples."""

    def __init__(self, config, annotation_file: str | None = None, split: str = "train"):
        self.config = config
        self.split = split
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

        self.samples: list[dict[str, Any]] = []
        for item in data:
            item_id = item.get("id")
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

            for qa in item.get("answers", []):
                question = qa.get("question")
                refs = qa.get("answers", []) or []
                if not question:
                    continue
                # X-VARS-style short prior text used by prompt and SFT builders.
                pred_action = ((labels.get("action") or {}).get("label") or "").strip()
                pred_offence = ((labels.get("offence") or {}).get("label") or "").strip()
                prior_prediction_text = " ".join([x for x in (pred_action, pred_offence) if x]).strip()
                self.samples.append(
                    {
                        "id": item_id,
                        "question": question,
                        "references": refs,
                        "video_path": video_path,
                        "labels": labels,
                        "metadata": metadata,
                        "prior_prediction_text": prior_prediction_text,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]
