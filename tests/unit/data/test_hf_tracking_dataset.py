"""Exercise indexed Hub tracking without a network connection."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

datasets = pytest.importorskip("datasets")
pytest.importorskip("torch_geometric")

from opensportslib.apis import Config
from opensportslib.apis.classification import ClassificationModel
from opensportslib.core.utils.config import dict_to_namespace
from opensportslib.core.utils.data import tracking_collate_fn
from opensportslib.datasets.classification_dataset import TrackingDataset, build
from opensportslib.datasets.hf_tracking import (
    HFTarTrackingReader,
    PreparedTrackingSplit,
    prepare_hf_tracking_split,
)
from opensportslib.tools import convert_json_to_parquet


@pytest.fixture
def hub_fixture(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    remote.mkdir()
    local.mkdir()
    payloads = {}

    for split in ("train", "valid", "test"):
        samples = []
        for index, label in enumerate(("A", "B")):
            rel = f"{split}/clip_{index}.parquet"
            path = local / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "balls": [json.dumps([{"x": index + t, "y": 2, "z": 0}]) for t in range(2)],
                    "homePlayers": [json.dumps([{"jerseyNum": 1, "x": 10, "y": 5}])] * 2,
                    "awayPlayers": [json.dumps([{"jerseyNum": 2, "x": -10, "y": -5}])] * 2,
                }
            ).to_parquet(path)
            samples.append(
                {
                    "id": f"{split}_{index}",
                    "inputs": [{"type": "tracking_parquet", "path": rel}],
                    "labels": {"action": {"label": label}},
                }
            )
        payload = {
            "version": "2.0",
            "task": "action_classification",
            "labels": {"action": {"type": "single_label", "labels": ["A", "B"]}},
            "data": samples,
        }
        annotations = local / split / "annotations.json"
        annotations.write_text(json.dumps(payload), encoding="utf-8")
        convert_json_to_parquet(
            annotations,
            local,
            remote / split,
            shard_mode="samples",
            samples_per_shard=1,
        )
        if split == "train":
            manifest_path = remote / split / "shard_manifest.parquet"
            manifest = pd.read_parquet(manifest_path)
            manifest["file_role"] = None  # Current SoccerNet-GAR upload omits this role.
            manifest.to_parquet(manifest_path)
        payloads[split] = annotations

    def fake_repo_info(self, *, repo_id, revision, repo_type):
        assert repo_type == "dataset"
        return SimpleNamespace(sha="fixturecommit")

    monkeypatch.setattr("huggingface_hub.HfApi.repo_info", fake_repo_info)
    real_load_dataset = datasets.load_dataset

    def fake_load_dataset(format_name, *, data_files, split, streaming, token, cache_dir):
        assert format_name == "parquet" and streaming and token
        filename = Path(data_files[split]).name
        return real_load_dataset(
            "parquet",
            data_files={split: str(remote / split / filename)},
            split=split,
            streaming=True,
            cache_dir=cache_dir,
        )

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    calls = []

    def fake_download(*, filename, **kwargs):
        calls.append(filename)
        return str(remote / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    config_dict = Config.from_file("sngar_tracking_hf.yaml").get_config()
    config_dict["DATA"]["inputs"]["tracking"]["source"]["cache_dir"] = str(tmp_path / "cache")
    config_dict["DATA"]["inputs"]["tracking"]["augmentations"] = {}
    return dict_to_namespace(config_dict), payloads, local, remote, calls


def test_hf_tracking_matches_local_graphs_and_caches_shard(hub_fixture):
    config, payloads, local, remote, calls = hub_fixture
    hf_data = build(config, None, split="train")
    assert sorted(calls) == [
        "train/shards/shard-000000.tar",
        "train/shards/shard-000001.tar",
    ]
    local_config = copy.deepcopy(config)
    local_config.DATA.inputs.tracking.source.format = "parquet"
    local_config.DATA.common.splits.train.source_path = str(local)
    local_data = TrackingDataset(local_config, str(payloads["train"]), split="train")

    assert len(hf_data) == len(local_data) == 2
    for index in range(2):
        remote_sample = hf_data[index]
        local_sample = local_data[index]
        assert remote_sample["id"] == local_sample["id"]
        assert remote_sample["label"] == local_sample["label"]
        for left, right in zip(remote_sample["graphs"], local_sample["graphs"]):
            torch.testing.assert_close(left.x, right.x)
            torch.testing.assert_close(left.edge_index, right.edge_index)

    hf_data[0]
    assert calls.count("train/shards/shard-000000.tar") == 1
    assert calls.count("train/shards/shard-000001.tar") == 1
    assert not list((Path(config.DATA.inputs.tracking.source.cache_dir)).rglob("clip_*.parquet"))
    assert hf_data.get_sample_weights().tolist() == local_data.get_sample_weights().tolist()


def test_hf_tracking_all_splits_and_multiworker_sampler(hub_fixture):
    config, payloads, local, remote, calls = hub_fixture
    model = ClassificationModel.__new__(ClassificationModel)
    model.config = config
    model.remote = None
    assert Path(model._resolve_split_path("train")).is_file()
    assert config.DATA.inputs.tracking.source.resolved_revision == "fixturecommit"
    for split in ("train", "valid", "test"):
        prepared = prepare_hf_tracking_split(config, split)
        data = build(config, None, split=split)
        assert prepared.annotations_path.is_file()
        assert prepared.index_path.is_file()
        assert [data[i]["id"] for i in range(len(data))] == [f"{split}_0", f"{split}_1"]

    train = build(config, None, split="train")
    sampler = WeightedRandomSampler(
        train.get_sample_weights(), num_samples=8, replacement=True,
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(
        train, sampler=sampler, batch_size=2, num_workers=2,
        collate_fn=tracking_collate_fn,
    )
    batches = list(loader)
    assert len(batches) == 4
    assert sum(len(batch["id"]) for batch in batches) == 8


def test_hf_tracking_missing_tar_member_fails_clearly(hub_fixture):
    config, payloads, local, remote, calls = hub_fixture
    data = build(config, None, split="valid")
    data._hf_reader.members["valid/clip_0.parquet"] = (
        "shard-000000.tar", "absent.parquet"
    )
    with pytest.raises(FileNotFoundError, match="absent.parquet"):
        data[0]


def test_hf_tracking_hub_access_error_names_login(hub_fixture, monkeypatch):
    config, payloads, local, remote, calls = hub_fixture

    def deny_access(self, **kwargs):
        raise PermissionError("gated")

    monkeypatch.setattr("huggingface_hub.HfApi.repo_info", deny_access)
    with pytest.raises(RuntimeError, match="hf auth login"):
        prepare_hf_tracking_split(config, "train")


def test_hf_tracking_fails_before_training_if_a_shard_is_unavailable(
    hub_fixture, monkeypatch
):
    config, _, _, _, _ = hub_fixture

    def unavailable(*, filename, **kwargs):
        if filename.endswith("shard-000001.tar"):
            raise FileNotFoundError(filename)
        return str(hub_fixture[3] / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", unavailable)
    with pytest.raises(RuntimeError, match="train/shards/shard-000001.tar"):
        prepare_hf_tracking_split(config, "train")


def test_hf_tracking_reuses_all_staged_shards(hub_fixture, monkeypatch):
    config, _, _, _, calls = hub_fixture
    prepared = prepare_hf_tracking_split(config, "train")
    assert len(calls) == 2
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: pytest.fail("cached shards must not be downloaded again"),
    )
    assert prepare_hf_tracking_split(config, "train") == prepared
    data = build(config, None, split="train")
    assert data[0]["id"] == "train_0"


def test_hf_tracking_keeps_all_sngar_train_shards_open(tmp_path, monkeypatch):
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps({"members": {
            f"clip_{i}": [f"shard-{i:06d}.tar", f"{i}.parquet"]
            for i in range(13)
        }}),
        encoding="utf-8",
    )
    prepared = PreparedTrackingSplit(
        tmp_path / "annotations.json", index_path, "org/repo", "commit", "train", tmp_path,
    )
    reader = HFTarTrackingReader(prepared)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: kwargs["filename"])
    opened = []

    class FakeTar:
        def close(self):
            pass

    def fake_open(path, mode):
        opened.append(path)
        return FakeTar()

    monkeypatch.setattr("tarfile.open", fake_open)
    for i in range(13):
        reader._tar(f"shard-{i:06d}.tar")
    reader._tar("shard-000000.tar")
    assert len(opened) == 13  # The first shard was reused, not reopened.
