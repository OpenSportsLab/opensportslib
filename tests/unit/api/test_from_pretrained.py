"""Revision-aware task loading must keep config and checkpoint together."""

from pathlib import Path
import shutil

import pytest

from opensportslib.apis import LocalizationModel


def test_from_pretrained_loads_requested_snapshot_and_device(tmp_path, monkeypatch):
    import huggingface_hub

    source = Path(__file__).parents[3] / "opensportslib/configs/localization/tdeed/FineDiving_small.yaml"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    shutil.copy2(source, snapshot / "config.yaml")
    (snapshot / "model.pth").write_bytes(b"checkpoint fixture")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    loaded = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(LocalizationModel, "load_weights", lambda self, weights: loaded.append(weights))
    model = LocalizationModel.from_pretrained(
        "org/model", revision="v1-small-seed2", device="cpu",
        cache_dir=str(tmp_path / "cache"), token=False, local_files_only=True,
    )
    assert calls == [{
        "repo_id": "org/model", "repo_type": "model", "revision": "v1-small-seed2",
        "allow_patterns": ["config.yaml", "model.pth"],
        "cache_dir": str(tmp_path / "cache"), "token": False, "local_files_only": True,
    }]
    assert loaded == [str(snapshot / "model.pth")]
    assert model.config.SYSTEM.device == "cpu"
    assert model.hf_revision == "v1-small-seed2"


@pytest.mark.parametrize("revision", [None, "", "  "])
def test_from_pretrained_requires_revision(revision, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: pytest.fail("download attempted"))
    with pytest.raises(ValueError, match="revision"):
        LocalizationModel.from_pretrained("org/model", revision=revision)


@pytest.mark.parametrize("filename", ["../model.pth", "/tmp/model.pth", "config.yaml", ""])
def test_from_pretrained_rejects_invalid_checkpoint_path(filename):
    with pytest.raises(ValueError, match="weights_filename"):
        LocalizationModel.from_pretrained("org/model", revision="v1", weights_filename=filename)


def test_from_pretrained_missing_checkpoint_does_not_load_main(tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="org/model@main"):
        LocalizationModel.from_pretrained("org/model", revision="main")
