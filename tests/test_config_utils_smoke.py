from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from opensportslib.core.utils.config import (
    dict_to_namespace,
    download_hf_dataset,
    expand,
    load_classes,
    load_gz_json,
    load_json,
    resolve_hf_dataset,
    store_gz_json,
    store_json,
)


def test_json_store_and_load_roundtrip(tmp_path):
    payload = {"a": 1, "b": ["x", "y"]}

    json_path = tmp_path / "payload.json"
    gz_path = tmp_path / "payload.json.gz"

    store_json(str(json_path), payload, pretty=True)
    store_gz_json(str(gz_path), payload)

    assert load_json(str(json_path)) == payload
    assert load_gz_json(str(gz_path)) == payload


def test_expand_returns_absolute_path(tmp_path):
    rel_path = Path("relative") / "path"
    expanded = expand(str(rel_path))

    assert Path(expanded).is_absolute()


def test_load_classes_supports_list():
    classes = ["PASS", "SHOT", "GOAL"]
    mapping = load_classes(classes)

    assert mapping == {"PASS": 1, "SHOT": 2, "GOAL": 3}


def test_dict_to_namespace_preserves_classes_shape():
    data = {"DATA": {"classes": ["A", "B"], "num_classes": 2}}
    ns = dict_to_namespace(data)

    assert ns.DATA.classes == ["A", "B"]
    assert ns.DATA.num_classes == 2


# def test_download_hf_dataset_passes_revision(monkeypatch, tmp_path):
#     calls = {}
#     cached_path = tmp_path / "hf-cache"

#     def fake_snapshot_download(**kwargs):
#         calls.update(kwargs)
#         return str(cached_path)

#     monkeypatch.setitem(
#         sys.modules,
#         "huggingface_hub",
#         SimpleNamespace(snapshot_download=fake_snapshot_download),
#     )

#     result = download_hf_dataset(
#         "OpenSportsLab/example-dataset",
#         revision="mvfouls",
#         hf_token="secret",
#     )

#     assert result == str(cached_path)
#     assert calls == {
#         "repo_id": "OpenSportsLab/example-dataset",
#         "repo_type": "dataset",
#         "revision": "mvfouls",
#         "token": "secret",
#     }


# def test_resolve_hf_dataset_updates_data_dir_and_resolves_paths(monkeypatch, tmp_path):
#     cached_path = tmp_path / "hf-cache"
#     cached_path.mkdir()

#     monkeypatch.setattr(
#         "opensportslib.core.utils.config.download_hf_dataset",
#         lambda repo_id, revision=None, hf_token=None: str(cached_path),
#     )

#     cfg = dict_to_namespace(
#         {
#             "DATA": {
#                 "data_dir": "/old/data",
#                 "train": {
#                     "video_path": "${DATA.data_dir}/train",
#                     "path": "${DATA.train.video_path}/annotations-train.json",
#                 },
#             }
#         }
#     )

#     resolved = resolve_hf_dataset(
#         cfg,
#         dataset_repo_id="OpenSportsLab/example-dataset",
#         dataset_revision="v1",
#     )

#     assert resolved.DATA.data_dir == str(cached_path)
#     assert resolved.DATA.train.video_path == str(cached_path / "train")
#     assert resolved.DATA.train.path == str(
#         cached_path / "train" / "annotations-train.json"
#     )


# def test_resolve_hf_dataset_rewrites_existing_data_dir_paths(monkeypatch, tmp_path):
#     cached_path = tmp_path / "hf-cache"
#     cached_path.mkdir()
#     old_data_dir = tmp_path / "old-data"

#     monkeypatch.setattr(
#         "opensportslib.core.utils.config.download_hf_dataset",
#         lambda repo_id, revision=None, hf_token=None: str(cached_path),
#     )

#     cfg = dict_to_namespace(
#         {
#             "DATA": {
#                 "data_dir": str(old_data_dir),
#                 "annotations": {
#                     "train": str(old_data_dir / "train" / "annotations.json"),
#                 },
#             }
#         }
#     )

#     resolved = resolve_hf_dataset(
#         cfg,
#         dataset_repo_id="OpenSportsLab/example-dataset",
#         dataset_revision="v1",
#     )

#     assert resolved.DATA.annotations.train == str(
#         cached_path / "train" / "annotations.json"
#     )


# def test_resolve_hf_dataset_skips_when_data_dir_overridden(monkeypatch):
#     called = False

#     def fake_download(*args, **kwargs):
#         nonlocal called
#         called = True

#     monkeypatch.setattr(
#         "opensportslib.core.utils.config.download_hf_dataset",
#         fake_download,
#     )

#     cfg = dict_to_namespace({"DATA": {"data_dir": "/local/data"}})
#     resolved = resolve_hf_dataset(
#         cfg,
#         dataset_repo_id="OpenSportsLab/example-dataset",
#         skip_download=True,
#     )

#     assert resolved is cfg
#     assert cfg.DATA.data_dir == "/local/data"
#     assert called is False


# def test_download_hf_dataset_invalid_repo_raises_clear_error(monkeypatch):
#     def fake_snapshot_download(**kwargs):
#         raise ValueError("missing revision")

#     monkeypatch.setitem(
#         sys.modules,
#         "huggingface_hub",
#         SimpleNamespace(snapshot_download=fake_snapshot_download),
#     )

#     with pytest.raises(RuntimeError, match="Could not download Hugging Face dataset repo"):
#         download_hf_dataset("OpenSportsLab/missing", revision="bad-revision")
