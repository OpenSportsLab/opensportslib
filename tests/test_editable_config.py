from copy import deepcopy
from pathlib import Path

import pytest

from opensportslib.apis import Config
from opensportslib.apis.base_task_model import BaseTaskModel


CONFIGS = Path(__file__).parents[1] / "opensportslib" / "configs"


def test_hierarchy_updates_references_and_reports_sources():
    config = Config.from_file(CONFIGS / "localization" / "video_ocv.yaml")

    config.update(
        data={"data_root": "/new/data"},
        training={"epochs": 30, "batch_size": 3},
        inference={"batch_size": 2},
    )
    effective = config.get_config()

    assert effective["TRAIN"]["scheduler"]["num_epochs"] == 30
    assert effective["DATA"]["common"]["splits"]["test"]["annotation_path"].startswith("/new/data")
    assert effective["DATA"]["common"]["splits"]["train"]["dataloader"]["batch_size"] == 3
    assert effective["DATA"]["common"]["splits"]["valid_data_frames"]["dataloader"]["batch_size"] == 2
    assert config.options()["training.epochs"]["sources"]["TRAIN.epochs"] == "user"


def test_advanced_update_is_atomic_and_detached():
    config = Config.from_file(CONFIGS / "classification" / "video.yaml")
    detached = config.get_config()
    detached["TRAIN"]["epochs"] = 999
    assert config.get_config()["TRAIN"]["epochs"] != 999

    before = config.get_config()
    with pytest.raises(ValueError, match="Unknown config path"):
        config.update(overrides={"TRAIN.optimizer.typo": 1})
    assert config.get_config() == before

    config.update(overrides={"TRAIN.scheduler.step_size": 7})
    assert config.get_config()["TRAIN"]["scheduler"]["step_size"] == 7


def test_collisions_and_range_errors_do_not_commit():
    config = Config.from_file(CONFIGS / "classification" / "video.yaml")
    before = config.get_config()
    with pytest.raises(ValueError, match="Conflicting"):
        config.update(
            training={"epochs": 12},
            overrides={"TRAIN.epochs": 13},
        )
    assert config.get_config() == before
    with pytest.raises(ValueError, match="outside"):
        config.update(training={"batch_size": 0})
    assert config.get_config() == before


def test_rule_variant_hides_and_rejects_ineffective_settings():
    config = Config.from_file(CONFIGS / "localization" / "h5_header_skeleton.yaml")
    config.update(overrides={
        "MODEL.components.rule.source.name": "h5_header_skeleton_recall",
    })
    assert "rule.angle_change_min_deg" not in config.options()
    with pytest.raises(ValueError, match="controlled by the selected rule variant"):
        config.update(overrides={
            "MODEL.components.rule.params.angle_change_min_deg": 20.0,
        })


def test_data_root_replaces_published_machine_source_paths():
    config = Config.from_file(CONFIGS / "localization" / "h5_header_distance.yaml")
    config.update(data={"data_root": "/datasets/tracking"})
    split = config.get_config()["DATA"]["common"]["splits"]["test"]
    assert split["source_path"] == "/datasets/tracking"
    # User manifests are independent inputs and are not rewritten when absolute.
    config.update(data={"test_set": "/manifests/test.json", "data_root": "/other"})
    assert config.get_config()["DATA"]["common"]["splits"]["test"]["annotation_path"] == "/manifests/test.json"


def test_vqa_training_batch_maps_to_sft_backend():
    config = Config.from_file(CONFIGS / "vqa" / "qwen3_vl_native.yaml")
    config.update(training={"batch_size": 4, "num_workers": 2})
    effective = config.get_config()
    assert effective["TRAIN"]["execution"]["sft"]["per_device_train_batch_size"] == 4
    assert effective["TRAIN"]["execution"]["sft"]["dataloader_num_workers"] == 2
    assert "inference.max_new_tokens" in config.options()
    assert "inference.batch_size" not in config.options()
    assert "inference.do_sample" in config.options()
    config.update(inference={"do_sample": True, "temperature": 0.7, "top_p": 0.9})
    generation = config.get_config()["TRAIN"]["execution"]["generation"]
    assert generation["do_sample"] is True
    assert generation["top_p"] == 0.9


def test_checkpoint_merge_reapplies_explicit_values():
    config = Config.from_file(CONFIGS / "classification" / "video.yaml")
    config.update(training={"epochs": 44}, data={"data_root": "/mine"})
    merged = config.get_config()
    merged["TRAIN"]["epochs"] = 2
    merged["DATA"]["common"]["data_root"] = "/checkpoint"

    effective = config.apply_to(merged)

    assert effective["TRAIN"]["epochs"] == 44
    assert effective["DATA"]["common"]["data_root"] == "/mine"
    assert effective["DATA"]["common"]["splits"]["test"]["annotation_path"].startswith("/mine")


def test_config_object_keeps_remembered_weights_and_is_copied(monkeypatch, tmp_path):
    import opensportslib.apis.base_task_model as base_api

    config = Config.from_file(CONFIGS / "classification" / "video.yaml")
    config.weights = "org/model"
    clone = deepcopy(config)
    clone.update(training={"epochs": 77})
    assert config.get_config()["TRAIN"]["epochs"] != 77
    assert clone.weights == "org/model"

    clone.update(runtime={"output_dir": str(tmp_path)})
    monkeypatch.setattr(base_api, "fetch_and_merge_config_from_HF", lambda cfg, *args, **kwargs: cfg)
    monkeypatch.setattr(base_api, "resolve_inference_class_metadata", lambda cfg: cfg)
    model = DummyModel(config=clone)
    assert model.loaded_weights == "org/model"
    explicit = DummyModel(config=clone, weights="other/model")
    assert explicit.loaded_weights == "other/model"


def test_all_concrete_direct_presets_expose_options():
    files = [
        path for task in ("classification", "localization", "vqa")
        for path in (CONFIGS / task).glob("*.yaml")
        if path.name != "default.yaml"
    ]
    assert files
    for path in files:
        assert Config.from_file(path).options(), path


class DummyModel(BaseTaskModel):
    def load_weights(self, weights=None, **kwargs):
        self.loaded_weights = weights

    def train(self, **kwargs):
        return None

    def infer(self, **kwargs):
        return {}

    def evaluate(self, **kwargs):
        return {}


def test_model_copies_prepared_config_and_supports_safe_updates(tmp_path):
    prepared = Config.from_file(CONFIGS / "classification" / "video.yaml")
    prepared.update(
        runtime={"output_dir": str(tmp_path)},
        training={"epochs": 21},
    )
    model = DummyModel(config=prepared)

    assert model.config.TRAIN.epochs == 21
    prepared.update(training={"epochs": 22})
    assert model.config.TRAIN.epochs == 21
    assert model.update_config(training={"epochs": 23}) is model
    assert model.config.TRAIN.epochs == 23
    with pytest.raises(ValueError, match="requires a new model"):
        model.update_config(runtime={"device": "cpu"})


def test_remote_config_serializes_only_after_capability_check(tmp_path):
    prepared = Config.from_file(CONFIGS / "classification" / "video.yaml")
    prepared.update(
        inference={"batch_size": 4},
    )
    model = DummyModel(config=prepared, remote="http://server", remote_model_id="model")
    model._open_request = lambda request: {
        "version": 1,
        "options": {"batch_size": {"minimum": 1, "maximum": 64}},
    }

    fields = model._config_request_fields({
        "task_type": "classification",
        "model_id": "model",
        "task_options": "{}",
    })

    import json
    envelope = json.loads(fields["task_options"])["config_overrides"]
    assert envelope == {"version": 1, "inference": {"batch_size": 4}}
