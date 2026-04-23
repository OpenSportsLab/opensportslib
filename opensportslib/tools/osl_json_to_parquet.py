"""
Convert an OpenSportsLib-style JSON annotation file into:
  1) a Parquet table with flattened metadata
  2) WebDataset TAR shards containing the referenced input files + sample metadata

Each sample typically contains:
  - id
  - inputs: [{"type": "...", "path": "...", ...}, ...]
  - optional task-specific fields: events / captions / dense_captions / labels / metadata

Public entry point:
    convert_json_to_parquet(...)
"""

from __future__ import annotations

import io
import json
import math
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ._common import json_dumps_compact, maybe_json_loads


def _load_json(json_path: str | Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_inputs_with_path(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    inputs = sample.get("inputs", [])
    if not isinstance(inputs, list):
        return []
    return [inp for inp in inputs if isinstance(inp, dict) and inp.get("path")]


def _flatten_sample_for_parquet(
    sample: Dict[str, Any],
    sample_index: int,
    shard_pattern: str = "shard-%06d.tar",
) -> Dict[str, Any]:
    sample_id = sample.get("id", f"sample_{sample_index:06d}")
    num_inputs = len(_extract_inputs_with_path(sample))

    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "num_inputs": num_inputs,
        "sample_payload": json_dumps_compact(sample),
        "suggested_shard_pattern": shard_pattern,
    }

    return row


def _build_sidecar_metadata(sample: Dict[str, Any]) -> bytes:
    """Canonical per-sample annotation payload stored as JSON inside each TAR shard."""
    return json.dumps(sample, ensure_ascii=False, indent=2).encode("utf-8")


def _resolve_media_path(
    media_root: str | Path,
    relative_path: str,
    missing_policy: str = "raise",
) -> Optional[Path]:
    path = Path(media_root) / relative_path
    if path.exists():
        return path
    if missing_policy == "skip":
        return None
    raise FileNotFoundError(f"Missing media file: {path}")


def _add_bytes_to_tar(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _add_file_to_tar(tar: tarfile.TarFile, src_path: Path, arcname: str) -> None:
    tar.add(str(src_path), arcname=arcname, recursive=False)


def convert_json_to_parquet(
    json_path: str | Path,
    media_root: str | Path,
    output_dir: str | Path,
    *,
    samples_per_shard: int = 100,
    compression: Optional[str] = "zstd",
    shard_prefix: str = "shard",
    missing_policy: str = "raise",
    keep_relative_paths_in_parquet: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Convert an OSL-style JSON file to:
      - metadata.parquet  (flattened, queryable per-sample table)
      - shards/*.tar      (WebDataset TAR shards with input files + sidecar JSON)
      - shard_manifest.parquet

    Parameters
    ----------
    json_path:
        Path to the JSON annotation file.
    media_root:
        Root directory where files referenced in ``inputs[].path`` live.
    output_dir:
        Destination directory.
    samples_per_shard:
        Number of samples grouped in one TAR shard.
    compression:
        Parquet compression codec (``"zstd"``, ``"snappy"``, ``"gzip"``, or ``None``).
    shard_prefix:
        Prefix for TAR shard file names.
    missing_policy:
        ``"raise"`` — abort if a referenced input file is missing.
        ``"skip"``  — keep sample in Parquet but omit missing file from shard.
    keep_relative_paths_in_parquet:
        If ``True`` (default), Parquet stores original relative paths from JSON.
        If ``False``, Parquet stores resolved absolute paths.
    overwrite:
        Whether to remove and recreate ``output_dir`` if it already exists.

    Returns
    -------
    dict
        Summary with counts of samples, shards, and input files processed.
    """
    json_path = Path(json_path)
    media_root = Path(media_root)
    output_dir = Path(output_dir)
    shards_dir = output_dir / "shards"
    parquet_path = output_dir / "metadata.parquet"

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use overwrite=True.")
        shutil.rmtree(output_dir)

    shards_dir.mkdir(parents=True, exist_ok=True)

    doc = _load_json(json_path)
    samples = doc.get("data", [])
    if not isinstance(samples, list):
        raise ValueError("JSON format error: top-level 'data' must be a list.")

    parquet_rows: List[Dict[str, Any]] = []
    shard_manifest: List[Dict[str, Any]] = []
    total_input_files_added = 0
    total_missing_input_files = 0
    header = {k: v for k, v in doc.items() if k != "data"}

    num_shards = max(1, math.ceil(len(samples) / max(1, samples_per_shard)))

    for shard_idx in range(num_shards):
        start = shard_idx * samples_per_shard
        end = min(len(samples), (shard_idx + 1) * samples_per_shard)
        shard_samples = samples[start:end]

        shard_name = f"{shard_prefix}-{shard_idx:06d}.tar"
        shard_path = shards_dir / shard_name

        with tarfile.open(shard_path, mode="w") as tar:
            for local_idx, sample in enumerate(shard_samples):
                global_idx = start + local_idx
                sample_id = sample.get("id", f"sample_{global_idx:06d}")
                key = f"{global_idx:09d}"

                row = _flatten_sample_for_parquet(sample, global_idx, shard_pattern=f"{shard_prefix}-%06d.tar")
                row["shard_name"] = shard_name
                row["sample_index"] = global_idx
                row["header"] = json_dumps_compact(header)
                parquet_rows.append(row)

                _add_bytes_to_tar(tar, f"{key}.json", _build_sidecar_metadata(sample))

                for input_idx, input_item in enumerate(_extract_inputs_with_path(sample)):
                    rel_path = str(input_item["path"])
                    resolved = _resolve_media_path(media_root, rel_path, missing_policy=missing_policy)

                    if resolved is None:
                        total_missing_input_files += 1
                        shard_manifest.append({
                            "sample_id": sample_id,
                            "shard_name": shard_name,
                            "input_index": input_idx,
                            "input_type": str(input_item.get("type", "")).strip(),
                            "relative_path": rel_path,
                            "resolved_path": None,
                            "status": "missing",
                        })
                        continue

                    ext = resolved.suffix.lstrip(".").lower() or "bin"
                    arcname = f"{key}.{input_idx}.{ext}"
                    _add_file_to_tar(tar, resolved, arcname)
                    shard_manifest.append({
                        "sample_id": sample_id,
                        "shard_name": shard_name,
                        "input_index": input_idx,
                        "input_type": str(input_item.get("type", "")).strip(),
                        "relative_path": rel_path,
                        "resolved_path": str(resolved if not keep_relative_paths_in_parquet else rel_path),
                        "status": "ok",
                        "wds_member": arcname,
                    })
                    total_input_files_added += 1

    df = pd.DataFrame(parquet_rows)

    if not keep_relative_paths_in_parquet:
        manifest_df = pd.DataFrame(shard_manifest)
        if not manifest_df.empty:
            ok_manifest = manifest_df[manifest_df["status"] == "ok"].copy()
            if not ok_manifest.empty:
                ok_manifest["input_index"] = ok_manifest["input_index"].astype(int)
                by_sample_input_paths: Dict[str, Dict[int, str]] = {}
                for sample_id, sample_manifest in ok_manifest.groupby("sample_id", sort=False):
                    sample_manifest = sample_manifest.sort_values("input_index")
                    by_sample_input_paths[str(sample_id)] = {
                        int(rec["input_index"]): str(rec["resolved_path"])
                        for _, rec in sample_manifest.iterrows()
                    }

                def _resolved_payload_for_row(row: pd.Series) -> str:
                    payload = maybe_json_loads(row.get("sample_payload"), {})
                    if not isinstance(payload, dict):
                        return json_dumps_compact(payload)

                    payload_copy = dict(payload)
                    inputs_value = payload_copy.get("inputs", [])
                    if not isinstance(inputs_value, list):
                        return json_dumps_compact(payload_copy)

                    resolved_by_index = by_sample_input_paths.get(str(row.get("sample_id")), {})
                    with_path_idx = 0
                    rewritten: List[Any] = []
                    for inp in inputs_value:
                        if not isinstance(inp, dict):
                            rewritten.append(inp)
                            continue
                        inp_copy = dict(inp)
                        if inp_copy.get("path"):
                            resolved = resolved_by_index.get(with_path_idx)
                            if resolved is not None:
                                inp_copy["path"] = resolved
                            with_path_idx += 1
                        rewritten.append(inp_copy)
                    payload_copy["inputs"] = rewritten
                    return json_dumps_compact(payload_copy)

                df["sample_payload"] = df.apply(_resolved_payload_for_row, axis=1)

    df.to_parquet(parquet_path, index=False, compression=compression)

    manifest_path = output_dir / "shard_manifest.parquet"
    pd.DataFrame(shard_manifest).to_parquet(manifest_path, index=False, compression=compression)

    return {
        "json_path": str(json_path),
        "media_root": str(media_root),
        "output_dir": str(output_dir),
        "parquet_path": str(parquet_path),
        "manifest_path": str(manifest_path),
        "shards_dir": str(shards_dir),
        "num_samples": len(samples),
        "num_shards": num_shards,
        "input_files_added": total_input_files_added,
        "missing_input_files": total_missing_input_files,
    }


__all__ = ["convert_json_to_parquet"]
