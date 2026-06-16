#!/usr/bin/env python3
"""Build X-VARS feature/prediction index JSON files from OSL VQA annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON payload: {path}")
    return payload


def _norm_offence(raw: str) -> str:
    x = raw.strip().lower()
    if "no offence" in x or "no offense" in x:
        return "No offence"
    if "offence" in x or "offense" in x or "foul" in x:
        return "Offence"
    return ""


def _norm_severity(raw: str) -> str:
    x = raw.strip().lower()
    if "red" in x:
        return "5.0"
    if "yellow" in x:
        return "3.0"
    if "no card" in x:
        return "1.0"
    return ""


def _sample_feature_dirs(
    item: dict[str, Any],
    data_root: Path,
    features_root: Path,
    split_name: str,
) -> list[Path]:
    out: list[Path] = []
    sid = str(item.get("id", "")).strip()
    if sid:
        out.append(features_root / split_name / sid)
    inputs = item.get("inputs") or []
    for inp in inputs:
        if not isinstance(inp, dict):
            continue
        if str(inp.get("type", "")).lower() != "video":
            continue
        rel = str(inp.get("path", "")).strip()
        if not rel:
            continue
        rel_path = Path(rel)
        if rel_path.is_absolute():
            abs_vid = rel_path
            try:
                rel_to_data = abs_vid.parent.relative_to(data_root)
            except ValueError:
                out.append(abs_vid.parent)
            else:
                out.append(features_root / rel_to_data)
        else:
            out.append(features_root / rel_path.parent)
    # dedup, keep order
    uniq: list[Path] = []
    seen = set()
    for p in out:
        k = str(p.resolve()) if p.exists() else str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def _feature_candidates(dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        for i in (1, 2, 3):
            out.append(d / f"PRE_CLIP_feature_clip_{i}.pkl")
    return out


def _build_prediction_row(item: dict[str, Any]) -> dict[str, Any]:
    labels = item.get("labels") or {}
    action = str(((labels.get("action") or {}).get("label")) or "").strip()
    offence = str(((labels.get("offence") or {}).get("label")) or "").strip()
    card = str(((labels.get("card") or {}).get("label")) or "").strip()
    row = {
        "id": str(item.get("id", "")).strip(),
        "Action class": action,
        "Offence": _norm_offence(offence),
        "Severity": _norm_severity(card),
        "source": "osl_labels",
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Build feature_index.json and prediction_index.json for xvars_clip mode.")
    ap.add_argument("--dataset-root", required=True, help="OSL-XFoul data root containing train/valid/test folders and split JSON files.")
    ap.add_argument(
        "--features-root",
        default=None,
        help="Optional root where CLIP feature files are stored. Defaults to --dataset-root.",
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write feature_index.json and prediction_index.json.")
    ap.add_argument(
        "--emit-expected-paths",
        action="store_true",
        help="Emit expected PRE_CLIP paths even when files are missing (bootstrap mode).",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    features_root = Path(args.features_root).expanduser().resolve() if args.features_root else dataset_root
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    missing_features = 0
    total = 0

    for split in ("train", "valid", "test"):
        ann = dataset_root / f"{split}/{split}.json"
        payload = _load_json(ann)
        items = payload.get("data") or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id", "")).strip()
            if not sid:
                continue
            total += 1
            dirs = _sample_feature_dirs(
                item,
                data_root=dataset_root,
                features_root=features_root,
                split_name=split,
            )
            candidates = _feature_candidates(dirs)
            existing = [str(p) for p in candidates if p.exists()]
            if existing:
                feature_rows.append({"id": sid, "feature_paths": existing, "split": split})
            elif args.emit_expected_paths and candidates:
                feature_rows.append({"id": sid, "feature_paths": [str(p) for p in candidates], "split": split, "missing": True})
                missing_features += 1
            else:
                missing_features += 1
            pred_rows.append(_build_prediction_row(item))

    feat_path = output_dir / "feature_index.json"
    pred_path = output_dir / "prediction_index.json"
    feat_path.write_text(json.dumps(feature_rows, indent=2), encoding="utf-8")
    pred_path.write_text(json.dumps(pred_rows, indent=2), encoding="utf-8")

    print(f"wrote: {feat_path}")
    print(f"wrote: {pred_path}")
    print(f"samples_total={total} feature_rows={len(feature_rows)} missing_features={missing_features}")
    if missing_features > 0 and not args.emit_expected_paths:
        raise SystemExit(
            "Some samples have no CLIP feature pickles. Re-run with --emit-expected-paths to bootstrap index files."
        )


if __name__ == "__main__":
    main()
