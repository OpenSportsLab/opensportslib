"""
Convert a dataset stored as:
  - metadata.parquet
  - WebDataset TAR shards (created by osl_json_to_parquet.py)

back into an OpenSportsLib-style JSON file.

Notes
-----
- Reconstruction relies on the per-sample sidecar JSON stored inside each TAR shard.
  The sidecar is the canonical full-fidelity annotation source.
- metadata.parquet is used only for routing (sample_index, shard_name) and lightweight
  filtering; it does not store a full copy of each sample.
- By default, reconstructed ``inputs[].path`` values remain the original relative paths.
  Pass ``extract_media=True`` to also extract the input files from the shards.

Public entry point:
    convert_parquet_to_json(...)
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ._common import maybe_json_loads


def _read_sidecar_json_from_tar(tar_path: Path, sample_index: int) -> Optional[Dict[str, Any]]:
    key = f"{sample_index:09d}.json"
    with tarfile.open(tar_path, "r") as tar:
        try:
            member = tar.getmember(key)
        except KeyError:
            return None
        f = tar.extractfile(member)
        if f is None:
            return None
        payload = json.loads(f.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid sidecar JSON format for sample_index={sample_index} in {tar_path}.")
        return payload


def _extract_sample_media_from_tar(
    tar_path: Path,
    sample_index: int,
    output_media_root: Path,
    original_paths: List[str],
    overwrite: bool = False,
) -> int:
    """
    Extract all input files for *sample_index* from the shard.

    Files are written to ``output_media_root / original_path``, preserving the
    original relative path structure so that ``inputs[].path`` values stay valid.

    Returns the number of files extracted.
    """
    key_prefix = f"{sample_index:09d}."
    extracted = 0

    with tarfile.open(tar_path, "r") as tar:
        members = [
            m
            for m in tar.getmembers()
            if m.isfile() and m.name.startswith(key_prefix) and not m.name.endswith(".json")
        ]

        def _input_idx(m: tarfile.TarInfo) -> int:
            part = m.name[len(key_prefix) :].split(".", 1)[0]
            try:
                return int(part)
            except ValueError:
                return 0

        members.sort(key=_input_idx)

        for member in members:
            input_idx = _input_idx(member)
            if input_idx >= len(original_paths):
                continue
            out_path = output_media_root / original_paths[input_idx]
            if out_path.exists() and not overwrite:
                extracted += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            f = tar.extractfile(member)
            if f is None:
                continue
            with open(out_path, "wb") as out_f:
                shutil.copyfileobj(f, out_f)
            extracted += 1

    return extracted


def _reconstruct_sample_from_row(
    row: pd.Series,
    sidecar: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a sample dict from a Parquet row, enriched by sidecar when available.

    Supported schema:
    - ``sample_payload`` (optional fallback)
    - sidecar JSON from shard (preferred)

    Legacy column layouts are intentionally not supported.
    """
    sample_payload = maybe_json_loads(row.get("sample_payload"), None)
    if sample_payload is not None and not isinstance(sample_payload, dict):
        raise ValueError(
            f"Invalid sample_payload type for sample_id={row.get('sample_id')!r}: expected JSON object."
        )

    if sidecar is None and sample_payload is None:
        raise ValueError(
            f"Cannot reconstruct sample_id={row.get('sample_id')!r}: missing both sidecar JSON and sample_payload. "
            "Legacy schemas are no longer supported."
        )

    sample: Dict[str, Any] = {}
    if isinstance(sample_payload, dict):
        sample.update(sample_payload)
    if isinstance(sidecar, dict):
        sample.update(sidecar)

    sample.setdefault("id", row["sample_id"])
    if "inputs" not in sample:
        sample["inputs"] = []
    elif not isinstance(sample.get("inputs"), list):
        raise ValueError(f"Invalid inputs for sample_id={sample.get('id')!r}: expected a list.")

    return sample


def convert_parquet_to_json(
    dataset_dir: str | Path,
    output_json_path: str | Path,
    *,
    extract_media: bool = False,
    output_media_root: Optional[str | Path] = None,
    overwrite_media: bool = False,
    json_indent: int = 2,
) -> Dict[str, Any]:
    """
    Convert a Parquet + WebDataset directory back to an OSL-style JSON file.

    The input directory must contain ``metadata.parquet`` and a ``shards/``
    sub-directory produced by :func:`opensportslib.tools.convert_json_to_parquet`.

    Parameters
    ----------
    dataset_dir:
        Directory produced by the forward converter.
    output_json_path:
        Destination JSON file path.
    extract_media:
        If ``True``, extract media files from TAR shards into ``output_media_root``.
        ``inputs[].path`` values are preserved as-is (original relative paths).
    output_media_root:
        Root directory for extracted media. Required when ``extract_media=True``.
    overwrite_media:
        Whether to overwrite already-extracted media files.
    json_indent:
        Indentation level for the output JSON.

    Returns
    -------
    dict
        Summary with sample count and extraction statistics.
    """
    dataset_dir = Path(dataset_dir)
    output_json_path = Path(output_json_path)
    metadata_path = dataset_dir / "metadata.parquet"
    shards_dir = dataset_dir / "shards"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not shards_dir.exists():
        raise FileNotFoundError(f"Missing shards directory: {shards_dir}")
    if extract_media and output_media_root is None:
        raise ValueError("output_media_root must be provided when extract_media=True")

    output_media_root_path = Path(output_media_root) if output_media_root is not None else None
    if output_media_root_path is not None:
        output_media_root_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(metadata_path).sort_values("sample_index").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("metadata.parquet is empty.")

    required_cols = {"sample_id", "sample_index", "shard_name", "header"}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        if "header" in missing:
            raise ValueError(
                "metadata.parquet is missing required 'header' column. "
                "Legacy schemas are no longer supported."
            )
        raise ValueError(f"metadata.parquet is missing required columns: {', '.join(missing)}")

    top_doc = maybe_json_loads(df.iloc[0].get("header"), None)
    if not isinstance(top_doc, dict):
        raise ValueError("metadata.parquet has invalid 'header' payload; expected JSON object.")
    top_doc = dict(top_doc)
    top_doc.pop("data", None)

    data: List[Dict[str, Any]] = []
    extracted_media_count = 0

    for _, row in df.iterrows():
        sample_index = int(row["sample_index"])
        shard_name = str(row["shard_name"]) if row.get("shard_name") is not None else ""
        if not shard_name:
            raise ValueError(f"Invalid shard_name for sample_index={sample_index}.")

        tar_path = shards_dir / shard_name
        if not tar_path.exists():
            raise FileNotFoundError(f"Missing shard file: {tar_path}")

        sidecar = _read_sidecar_json_from_tar(tar_path, sample_index)
        sample = _reconstruct_sample_from_row(row, sidecar=sidecar)

        if extract_media:
            inputs = sample.get("inputs", []) if isinstance(sample, dict) else []
            original_input_paths = [
                str(inp["path"])
                for inp in inputs
                if isinstance(inp, dict) and inp.get("path")
            ]
            extracted_media_count += _extract_sample_media_from_tar(
                tar_path=tar_path,
                sample_index=sample_index,
                output_media_root=output_media_root_path,
                original_paths=original_input_paths,
                overwrite=overwrite_media,
            )

        data.append(sample)

    top_doc["data"] = data

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(top_doc, f, ensure_ascii=False, indent=json_indent)

    return {
        "dataset_dir": str(dataset_dir),
        "output_json_path": str(output_json_path),
        "num_samples": len(data),
        "extract_media": extract_media,
        "output_media_root": str(output_media_root_path) if output_media_root_path else None,
        "extracted_input_files": extracted_media_count,
        "extracted_media_files": extracted_media_count,
    }


__all__ = ["convert_parquet_to_json"]
