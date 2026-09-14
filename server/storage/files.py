from __future__ import annotations

import shutil
import urllib.parse
import urllib.request
import json
import zipfile
from pathlib import Path
from typing import Any, BinaryIO


def prepare_video_source(source_path: str | None, media_url: str | None, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if source_path:
        path = Path(source_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        return path
    if not media_url:
        raise ValueError("Either `source_path` or `media_url` must be provided.")

    parsed = urllib.parse.urlparse(media_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https media URLs are supported.")

    filename = Path(parsed.path).name or "downloaded_video"
    destination = destination_dir / filename
    with urllib.request.urlopen(media_url) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination.resolve()


def save_uploaded_file(filename: str | None, source_file: BinaryIO, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename or "uploaded_video").name or "uploaded_video"
    destination = destination_dir / safe_name
    source_file.seek(0)
    with open(destination, "wb") as handle:
        shutil.copyfileobj(source_file, handle)
    return destination.resolve()


def prepare_uploaded_test_set(
    manifest_path: str | Path,
    archive_path: str | Path,
    destination_dir: Path,
    *,
    max_archive_bytes: int,
    max_extracted_bytes: int,
    max_file_count: int,
    allowed_extensions: tuple[str, ...],
) -> Path:
    """Extract an uploaded archive and rewrite manifest media paths to local absolute paths."""

    manifest_path = Path(manifest_path).resolve()
    archive_path = Path(archive_path).resolve()
    if archive_path.stat().st_size > max_archive_bytes:
        raise ValueError("Uploaded media archive exceeds the configured size limit.")
    extract_dir = destination_dir / "media"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) > max_file_count:
            raise ValueError("Uploaded media archive exceeds the configured file-count limit.")
        if len({info.filename for info in members}) != len(members):
            raise ValueError("Uploaded media archive contains duplicate file names.")
        total_size = sum(info.file_size for info in members)
        if total_size > max_extracted_bytes:
            raise ValueError("Uploaded media archive exceeds the configured extracted-size limit.")
        for info in members:
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("Uploaded media archive contains an unsafe file path.")
            if allowed_extensions and relative.suffix.lower() not in allowed_extensions:
                raise ValueError(f"Archive file type is not allowed: {relative.name}")
            destination = (extract_dir / relative).resolve()
            if extract_dir.resolve() not in destination.parents:
                raise ValueError("Uploaded media archive contains an unsafe file path.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Uploaded test manifest must be an OSL JSON object with a `data` list.")
    for value, setter in _manifest_media_references(payload):
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Manifest contains an unsafe media path: {value}")
        resolved = (extract_dir / relative).resolve()
        if extract_dir.resolve() not in resolved.parents or not resolved.is_file():
            raise ValueError(f"Manifest media file is missing from uploaded archive: {value}")
        setter(str(resolved))

    rewritten_path = destination_dir / "uploaded_test_set.json"
    rewritten_path.write_text(json.dumps(payload), encoding="utf-8")
    return rewritten_path.resolve()


def _manifest_media_references(payload: dict[str, Any]):
    for sample in payload.get("data", []):
        if not isinstance(sample, dict):
            continue
        if isinstance(sample.get("video_path"), str):
            yield sample["video_path"], lambda value, sample=sample: sample.__setitem__("video_path", value)
        for key in ("frame_paths", "video_frames"):
            values = sample.get(key)
            if isinstance(values, list):
                for index, value in enumerate(values):
                    if isinstance(value, str):
                        yield value, lambda replacement, values=values, index=index: values.__setitem__(index, replacement)
        for input_obj in sample.get("inputs") or []:
            if not isinstance(input_obj, dict):
                continue
            for key in ("path", "ball_path"):
                if isinstance(input_obj.get(key), str):
                    yield input_obj[key], lambda value, input_obj=input_obj, key=key: input_obj.__setitem__(key, value)
            values = input_obj.get("paths")
            if isinstance(values, list):
                for index, value in enumerate(values):
                    if isinstance(value, str):
                        yield value, lambda replacement, values=values, index=index: values.__setitem__(index, replacement)
