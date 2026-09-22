"""Hub JSON staging tests with a generated local video in place of the Hub."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from opensportslib.core.config.editable import Config
from opensportslib.core.utils.config import dict_to_namespace
from opensportslib.core.utils.load_annotations import annotationstoe2eformat
from opensportslib.datasets.hf_json import prepare_hf_json_split


def _video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 24))
    assert writer.isOpened()
    for index in range(60):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()


@pytest.fixture
def hub(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    for split in ("train", "valid", "test"):
        filename = f"{split}/video/1.mp4"
        _video(remote / filename)
        payload = {
            "version": "2.0", "task": "action_spotting",
            "labels": {"action": {"labels": ["PASS"]}},
            "data": [{
                "game_id": "1",
                "inputs": [
                    {"type": "tracking_parquet", "path": f"{split}/tracking/1.parquet"},
                    {"type": "video", "path": filename, "fps": 30.0},
                ],
                "events": [{"label": "PASS", "position_ms": 500}],
            }],
        }
        (remote / f"{split}.json").write_text(json.dumps(payload))

    calls = []

    class FakeApi:
        def __init__(self, token):
            assert token is True

        def repo_info(self, **kwargs):
            assert kwargs["revision"] == "multimodal"
            return SimpleNamespace(sha="abc123")

    def download(*, filename, **kwargs):
        calls.append(filename)
        path = remote / filename
        if not path.is_file():
            raise FileNotFoundError(filename)
        return str(path)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return remote, calls


def _config(tmp_path):
    payload = Config.from_file("sngar_spotting_video_hf.yaml").get_config()
    payload["DATA"]["inputs"]["video"]["source"]["cache_dir"] = str(tmp_path / "cache")
    return dict_to_namespace(payload)


def test_hf_json_stages_each_split_and_reuses_cache(tmp_path, hub):
    _, calls = hub
    config = _config(tmp_path)
    for split in ("train", "valid", "test"):
        prepared = prepare_hf_json_split(config, split)
        payload = json.loads(prepared.annotation_path.read_text())
        sample = payload["data"][0]
        assert sample["inputs"] == [{
            "type": "video", "path": f"{split}/video/1.mp4", "fps": 30.0
        }]
        assert (prepared.source_path / sample["inputs"][0]["path"]).is_file()
        labels, task = annotationstoe2eformat(
            str(prepared.annotation_path), str(prepared.source_path), 30, 5, False
        )
        assert task == "action"
        assert labels[0]["events"] == [{"frame": 2, "label": "PASS"}]
        assert labels[0]["num_frames"] > 0
        before = len(calls)
        assert prepare_hf_json_split(config, split) == prepared
        assert len(calls) == before
    assert config.DATA.inputs.video.source.resolved_revision == "abc123"
    assert sorted(calls) == sorted(
        filename
        for split in ("train", "valid", "test")
        for filename in (f"{split}.json", f"{split}/video/1.mp4")
    )


def test_hf_json_rejects_missing_video_input(tmp_path, hub):
    remote, _ = hub
    manifest = remote / "train.json"
    payload = json.loads(manifest.read_text())
    payload["data"][0]["inputs"] = payload["data"][0]["inputs"][:1]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="Expected one 'video' input"):
        prepare_hf_json_split(_config(tmp_path), "train")


def test_hf_json_reports_hub_access_failure(tmp_path, monkeypatch):
    class DeniedApi:
        def __init__(self, token):
            pass

        def repo_info(self, **kwargs):
            raise PermissionError("access denied")

    monkeypatch.setattr("huggingface_hub.HfApi", DeniedApi)
    with pytest.raises(RuntimeError, match="Request access and run `hf auth login`"):
        prepare_hf_json_split(_config(tmp_path), "train")


def test_hf_json_resolve_updates_valid_frame_split(tmp_path, hub):
    from opensportslib.apis.localization import LocalizationModel

    model = LocalizationModel.__new__(LocalizationModel)
    model.config = _config(tmp_path)
    resolved = model._resolve_split_path("valid")
    assert Path(resolved).is_file()
    assert model.config.DATA.common.splits.valid.source_path == str(Path(resolved).parent)
    assert model.config.DATA.common.splits.valid_data_frames.source_path == str(Path(resolved).parent)


def test_hf_json_builds_video_spotting_clip(tmp_path, hub):
    from opensportslib.apis.localization import LocalizationModel
    from opensportslib.core.config.accessors import set_split_annotation_path
    from opensportslib.datasets.builder import build_dataset

    model = LocalizationModel.__new__(LocalizationModel)
    model.config = _config(tmp_path)
    model.config.DATA.inputs.video.sampling.clip_len = 10
    model.config.DATA.inputs.video.sampling.epoch_num_frames = 100
    model.config.DATA.inputs.video.transform.resize.height = 24
    model.config.DATA.inputs.video.transform.resize.width = 32
    set_split_annotation_path(model.config, "train", model._resolve_split_path("train"))
    data = build_dataset(model.config, split="train")
    dataset = data.building_dataset(data.cfg, gpu=0, default_args=data.default_args)
    clip = dataset[0]
    assert clip["frame"].shape == (10, 3, 24, 32)
    assert clip["label"].shape == (10,)


def test_opencv_accumulates_across_batches_without_batch_divisibility():
    from opensportslib.core.utils.load_annotations import check_config

    config = dict_to_namespace(Config.from_file("sngar_spotting_video_hf.yaml").get_config())
    assert config.DATA.common.splits.train.dataloader.batch_size == 1
    assert config.TRAIN.execution.acc_grad_iter == 4
    check_config(config, split="train")

    config.DATA.common.runtime.loader_backend = "dali"
    with pytest.raises(ValueError, match="DALI train batch_size must be divisible"):
        check_config(config, split="train")
