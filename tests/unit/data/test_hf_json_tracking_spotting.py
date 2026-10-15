"""A generated tracking Parquet exercises Hub staging and graph loading."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from opensportslib.apis.localization import LocalizationModel
from opensportslib.core.config.editable import Config
from opensportslib.core.config.accessors import set_split_annotation_path
from opensportslib.core.utils.config import dict_to_namespace
from opensportslib.datasets.builder import build_dataset


@pytest.fixture(params=["tracking_parquet", "tracking_tracab"])
def tracking_hub(tmp_path, monkeypatch, request):
    remote = tmp_path / "remote"
    for split in ("train", "valid", "test"):
        relative = f"{split}/tracking_parquet/1.parquet"
        parquet = remote / relative
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "videoTimeMs": [1000 * frame / 30 for frame in range(120)],
            "balls": ['[{"x": 0, "y": 0, "z": 0}]'] * 120,
            "homePlayers": ["[]"] * 120,
            "awayPlayers": ["[]"] * 120,
        }).to_parquet(parquet)
        source = {"type": "tracking_parquet", "path": relative, "fps": 30.0}
        if request.param == "tracking_tracab":
            from tests.helpers.tracab import write_tracab
            parquet.unlink()
            relative = f"{split}/tracking_tracab/1.dat"
            source = write_tracab(remote / relative, sidecar=True)
            source["path"] = relative
            for key in ("metadata_path", "tf05_path", "tf08_path"):
                source[key] = str(Path(source[key]).relative_to(remote))
        manifest = {
            "version": "2.0", "task": "action_spotting",
            "labels": {"action": {"labels": ["PASS"]}},
            "data": [{
                "game_id": "1", "split": split,
                "inputs": [
                    {"type": "video", "path": f"{split}/video/1.mp4"},
                    source,
                ],
                "events": [{"head": "action", "label": "PASS", "position_ms": 500}],
            }],
        }
        (remote / f"{split}.json").write_text(json.dumps(manifest), encoding="utf-8")

    calls = []

    class FakeApi:
        def __init__(self, token):
            assert token is True

        def repo_info(self, **kwargs):
            assert kwargs["revision"] == "multimodal"
            return SimpleNamespace(sha="test-commit")

    def download(*, filename, **kwargs):
        calls.append(filename)
        result = remote / filename
        if not result.is_file():
            raise FileNotFoundError(filename)
        return str(result)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return calls, request.param


def test_tracking_hf_json_builds_graph_clips_for_all_splits(tmp_path, tracking_hub):
    calls, modality = tracking_hub
    payload = Config.from_file("examples/sngar/sngar_spotting_tracking_hf.yaml").get_config()
    payload["DATA"]["inputs"]["video"]["source"]["cache_dir"] = str(tmp_path / "cache")
    payload["DATA"]["inputs"]["video"]["sampling"]["clip_len"] = 10
    payload["DATA"]["inputs"]["video"]["sampling"]["epoch_num_frames"] = 100
    payload["DATA"]["common"]["splits"]["test"]["overlap_len"] = 0
    payload["DATA"]["inputs"]["video"]["modality"] = modality
    payload["DATA"]["inputs"]["video"]["source"]["input_type"] = modality
    config = dict_to_namespace(payload)
    model = LocalizationModel.__new__(LocalizationModel)
    model.config = config

    for split in ("train", "valid", "test"):
        annotation = Path(model._resolve_split_path(split))
        staged = json.loads(annotation.read_text(encoding="utf-8"))
        source = staged["data"][0]["inputs"][0]
        assert source["type"] == modality
        assert (annotation.parent / source["path"]).is_file()
        if modality == "tracking_tracab":
            assert source["metadata"]["periods"][1]["period"] == 2
        assert f"{split}/video/1.mp4" not in calls
        set_split_annotation_path(config, split, str(annotation))

    set_split_annotation_path(config, "valid_data_frames", config.DATA.common.splits.valid.annotation_path)

    train = build_dataset(config, split="train")
    dataset = train.building_dataset(train.cfg, gpu=0, default_args=train.default_args)
    sample = dataset[0]
    assert len(sample["graphs"]) == 10
    assert sample["graphs"][0].x.shape == (23, 8)
    assert sample["label"].shape == (10,)
    assert dataset._labels[0]["events"] == [
        {"frame": 2, "eval_frame": 2, "label": "PASS"}
    ]
    for split in ("valid_data_frames", "test"):
        data = build_dataset(config, split=split)
        evaluation = data.building_dataset(data.cfg, gpu=0, default_args=data.default_args)
        assert len(evaluation) > 0
        assert len(evaluation[0]["graphs"]) == 10
    assert config.DATA.inputs.video.source.resolved_revision == "test-commit"
    before = len(calls)
    assert Path(model._resolve_split_path("train")).is_file()
    assert len(calls) == before
