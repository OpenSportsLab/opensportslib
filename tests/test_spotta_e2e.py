from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest
from torch import nn

from opensportslib.core.config.accessors import get_loader_backend
from opensportslib.models.base.e2e import E2EModel
from opensportslib.adaptation.spotta import (
    FramewiseStrongAugmentation,
    RobustBatchNorm,
    SpoTTA,
    SpoTTAConfig,
    replace_batch_norm,
)
from opensportslib.core.config import load_config


class _TinyE2ESpot(nn.Module):
    def __init__(self, num_classes=2, predicted_class=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.classifier = nn.Linear(3, num_classes)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.bias.fill_(-2.0)
            self.classifier.bias[predicted_class] = 2.0

    def forward(self, clips):
        batch, time, channels, height, width = clips.shape
        frames = self.bn(clips.reshape(batch * time, channels, height, width))
        features = frames.mean(dim=(-1, -2))
        return self.classifier(features).reshape(batch, time, -1)


def _spotta_config(**overrides):
    config = {
        "enabled": True,
        "name": "spotta",
        "prediction_timing": "adapt_then_predict",
        "robust_bn": {"alpha": 0.05, "tether": {"cap": 0.5}},
        "confidence_gate": {
            "action_class_index": 1,
            "min_action_frames": 1,
            "threshold": 0.3,
        },
        "memory": {
            "capacity": 8,
            "update_frequency": 2,
            "lambda_t": 1.0,
            "lambda_u": 1.0,
        },
        "optimizer": {"learning_rate": 0.001, "beta": 0.9},
        "teacher": {
            "base_nu": 0.001,
            "max_nu": 0.02,
            "drift_scale": 10.0,
            "drift_threshold": 1.0,
            "drift_gamma": 0.2,
        },
        "augmentation": {"enabled": False},
    }
    config.update(overrides)
    return config


def test_robust_batch_norm_preserves_eval_output_and_frozen_anchor():
    torch.manual_seed(7)
    source = nn.Sequential(nn.BatchNorm2d(3)).eval()
    with torch.no_grad():
        source[0].running_mean.copy_(torch.tensor([0.2, -0.1, 0.4]))
        source[0].running_var.copy_(torch.tensor([0.7, 1.3, 2.0]))
        source[0].weight.copy_(torch.tensor([0.8, 1.1, 0.9]))
        source[0].bias.copy_(torch.tensor([-0.2, 0.3, 0.1]))
    inputs = torch.randn(4, 3, 5, 5)
    expected = source(inputs)

    layers = replace_batch_norm(source, alpha=0.05)
    source.eval()
    actual = source(inputs)

    assert len(layers) == 1
    assert isinstance(layers[0], RobustBatchNorm)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    anchor_mean = layers[0].anchor_mean.clone()
    source.train()
    source(inputs + 2)
    assert torch.equal(layers[0].anchor_mean, anchor_mean)
    assert not torch.equal(layers[0].source_mean, anchor_mean)


def test_spotta_gates_actions_and_updates_every_second_gated_clip():
    spotta = SpoTTA(_TinyE2ESpot(), _spotta_config())
    clips = torch.randn(2, 4, 3, 5, 5)

    predicted, probabilities = spotta.predict(clips, use_amp=False)

    assert predicted.shape == (2, 4)
    assert probabilities.shape == (2, 4, 2)
    assert spotta.stats["clips_seen"] == 2
    assert spotta.stats["clips_gated"] == 2
    assert spotta.stats["memory_insertions"] == 2
    assert spotta.stats["memory_occupancy"] == 2
    assert spotta.stats["update_attempts"] == 1
    assert spotta.stats["updates_completed"] == 1
    assert {layer.tether_rho for layer in spotta.tether_layers} == {0.25}


def test_framewise_strong_augmentation_preserves_clip_frame_shape_and_range():
    torch.manual_seed(11)
    frames = torch.randn(3, 3, 16, 20)

    augmented = FramewiseStrongAugmentation(image_size=16)(frames)

    assert augmented.shape == frames.shape
    assert float(augmented.min()) >= 0.0
    assert float(augmented.max()) <= 1.0


def test_e2e_wrapper_starts_fresh_spotta_without_mutating_source_model():
    wrapper = E2EModel.__new__(E2EModel)
    wrapper._model = _TinyE2ESpot()
    wrapper._num_classes = 2
    wrapper._multi_gpu = False
    wrapper._test_time_adapter = None
    source_state = {
        name: tensor.clone() for name, tensor in wrapper._model.state_dict().items()
    }

    wrapper.configure_test_time_adaptation(_spotta_config())
    wrapper.predict(torch.randn(2, 4, 3, 5, 5), use_amp=False)
    first_adapter = wrapper._test_time_adapter
    wrapper.configure_test_time_adaptation(_spotta_config())

    assert wrapper._test_time_adapter is not first_adapter
    for name, tensor in wrapper._model.state_dict().items():
        assert torch.equal(tensor, source_state[name])


def test_e2e_wrapper_keeps_one_spotta_instance_across_session_batches():
    wrapper = E2EModel.__new__(E2EModel)
    wrapper._model = _TinyE2ESpot()
    wrapper._num_classes = 2
    wrapper._multi_gpu = False
    wrapper._test_time_adapter = None
    wrapper.configure_test_time_adaptation(_spotta_config())
    session_adapter = wrapper._test_time_adapter

    wrapper.predict(torch.randn(1, 4, 3, 5, 5), use_amp=False)
    wrapper.predict(torch.randn(1, 4, 3, 5, 5), use_amp=False)

    assert wrapper._test_time_adapter is session_adapter
    assert wrapper.test_time_adaptation_stats["clips_seen"] == 2
    assert wrapper.test_time_adaptation_stats["memory_occupancy"] == 2
    assert wrapper.test_time_adaptation_stats["updates_completed"] == 1


def test_disabled_spotta_uses_ordinary_e2e_prediction_path():
    wrapper = E2EModel.__new__(E2EModel)
    wrapper._model = _TinyE2ESpot()
    wrapper._num_classes = 2
    wrapper._multi_gpu = False
    wrapper._test_time_adapter = None
    wrapper.device = torch.device("cpu")
    clips = torch.randn(2, 4, 3, 5, 5)
    wrapper._model.eval()
    with torch.no_grad():
        expected_probabilities = wrapper._model(clips).softmax(dim=2)
        expected_classes = expected_probabilities.argmax(dim=2)

    wrapper.configure_test_time_adaptation({"enabled": False, "name": "spotta"})
    predicted_classes, probabilities = wrapper.predict(clips, use_amp=False)

    assert wrapper._test_time_adapter is None
    assert torch.equal(torch.from_numpy(predicted_classes), expected_classes)
    assert torch.allclose(
        torch.from_numpy(probabilities), expected_probabilities, atol=1e-7
    )


def test_spotta_supports_configured_action_class_in_multiclass_output():
    config = _spotta_config()
    config["confidence_gate"]["action_class_index"] = 2
    spotta = SpoTTA(
        _TinyE2ESpot(num_classes=4, predicted_class=2), config
    )

    predicted, probabilities = spotta.predict(
        torch.randn(2, 4, 3, 5, 5), use_amp=False
    )

    assert predicted.shape == (2, 4)
    assert probabilities.shape == (2, 4, 4)
    assert torch.from_numpy(predicted).eq(2).all()
    assert spotta.stats["clips_gated"] == 2
    assert spotta.stats["updates_completed"] == 1


def test_spotta_config_rejects_negative_action_class_index():
    config = _spotta_config()
    config["confidence_gate"]["action_class_index"] = -1

    with pytest.raises(ValueError, match="must be non-negative"):
        SpoTTAConfig.from_mapping(config)


def test_spotta_rejects_action_class_outside_model_outputs():
    config = _spotta_config()
    config["confidence_gate"]["action_class_index"] = 4
    spotta = SpoTTA(_TinyE2ESpot(num_classes=4), config)

    with pytest.raises(ValueError, match="outside the model's 4 output classes"):
        spotta.predict(torch.randn(1, 4, 3, 5, 5), use_amp=False)


def test_spotta_config_rejects_unsupported_semantic_changes():
    config = _spotta_config()
    config["confidence_gate"]["aggregation"] = "mean"

    try:
        SpoTTAConfig.from_mapping(config)
    except ValueError as exc:
        assert "min_over_predicted_action_frames" in str(exc)
    else:
        raise AssertionError("Expected a changed confidence aggregation to fail.")


def test_spotta_config_contains_only_supported_options():
    with patch(
        "opensportslib.core.config.loader._dali_available", return_value=False
    ):
        config = load_config(
            "opensportslib/configs/localization/e2e_spotta.yaml",
            as_namespace=False,
        )
    spotta_config = config["MODEL"]["policies"]["test_time_adaptation"]

    assert spotta_config["enabled"] is True
    assert spotta_config["prediction_timing"] == "adapt_then_predict"
    assert spotta_config["confidence_gate"]["threshold"] == 0.3
    assert spotta_config["memory"]["capacity"] == 8
    assert spotta_config["memory"]["update_frequency"] == 2
    assert spotta_config["robust_bn"]["tether"]["mode"] == "bayesian"
    assert "class_policy" not in spotta_config["memory"]
    assert "frame_filter" not in spotta_config
    assert "steps" not in spotta_config
    assert "reset_frequency" not in spotta_config
    assert "action_frame_weight" not in spotta_config


def test_localization_starts_fresh_spotta_session_and_forces_opencv_runtime():
    from opensportslib.apis.localization import LocalizationModel

    configured = []

    class _Model:
        def configure_test_time_adaptation(self, adaptation):
            configured.append(adaptation)

    adaptation = SimpleNamespace(enabled=True, name="spotta")
    api = LocalizationModel.__new__(LocalizationModel)
    api.model = _Model()
    api.config = SimpleNamespace(
        MODEL=SimpleNamespace(
            metadata=SimpleNamespace(family="E2E"),
            policies=SimpleNamespace(test_time_adaptation=adaptation),
        ),
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                runtime=SimpleNamespace(loader_backend="dali"),
                splits=SimpleNamespace(
                    test=SimpleNamespace(type="VideoGameWithOpencvVideo")
                ),
            )
        ),
    )

    api._configure_test_time_adaptation()

    assert configured == [adaptation]
    assert get_loader_backend(api.config) == "opencv"
