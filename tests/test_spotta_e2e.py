from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from opensportslib.core.config.accessors import get_loader_backend
from opensportslib.models.base.e2e import E2EModel
from opensportslib.models.policies.spotta import (
    FramewiseStrongAugmentation,
    RobustBatchNorm,
    SpoTTA,
    SpoTTAConfig,
    replace_batch_norm,
)
from opensportslib.core.config import load_config


class _TinyE2ESpot(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.classifier = nn.Linear(3, 2)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.bias.copy_(torch.tensor([-2.0, 2.0]))

    def forward(self, clips):
        batch, time, channels, height, width = clips.shape
        frames = self.bn(clips.reshape(batch * time, channels, height, width))
        features = frames.mean(dim=(-1, -2))
        return self.classifier(features).reshape(batch, time, 2)


def _policy_config(**overrides):
    config = {
        "enabled": True,
        "name": "spotta",
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


def test_effective_recipe_gates_headers_and_updates_every_second_gated_clip():
    policy = SpoTTA(_TinyE2ESpot(), _policy_config())
    clips = torch.randn(2, 4, 3, 5, 5)

    predicted, probabilities = policy.predict(clips, use_amp=False)

    assert predicted.shape == (2, 4)
    assert probabilities.shape == (2, 4, 2)
    assert policy.stats["clips_seen"] == 2
    assert policy.stats["clips_gated"] == 2
    assert policy.stats["memory_insertions"] == 2
    assert policy.stats["memory_occupancy"] == 2
    assert policy.stats["update_attempts"] == 1
    assert policy.stats["updates_completed"] == 1
    assert {layer.tether_rho for layer in policy.tether_layers} == {0.25}


def test_framewise_strong_augmentation_preserves_clip_frame_shape_and_range():
    torch.manual_seed(11)
    frames = torch.randn(3, 3, 16, 20)

    augmented = FramewiseStrongAugmentation(image_size=16)(frames)

    assert augmented.shape == frames.shape
    assert float(augmented.min()) >= 0.0
    assert float(augmented.max()) <= 1.0


def test_e2e_wrapper_starts_fresh_policy_without_mutating_source_model():
    wrapper = E2EModel.__new__(E2EModel)
    wrapper._model = _TinyE2ESpot()
    wrapper._num_classes = 2
    wrapper._multi_gpu = False
    wrapper._test_time_policy = None
    source_state = {
        name: tensor.clone() for name, tensor in wrapper._model.state_dict().items()
    }

    wrapper.configure_test_time_adaptation(_policy_config())
    wrapper.predict(torch.randn(2, 4, 3, 5, 5), use_amp=False)
    first_policy = wrapper._test_time_policy
    wrapper.configure_test_time_adaptation(_policy_config())

    assert wrapper._test_time_policy is not first_policy
    for name, tensor in wrapper._model.state_dict().items():
        assert torch.equal(tensor, source_state[name])


def test_spotta_config_rejects_non_header_action_index():
    config = _policy_config()
    config["confidence_gate"]["action_class_index"] = 2

    try:
        SpoTTAConfig.from_mapping(config)
    except ValueError as exc:
        assert "class index 1" in str(exc)
    else:
        raise AssertionError("Expected invalid Header class index to fail.")


def test_spotta_config_rejects_recipe_semantic_changes():
    config = _policy_config()
    config["confidence_gate"]["aggregation"] = "mean"

    try:
        SpoTTAConfig.from_mapping(config)
    except ValueError as exc:
        assert "min_over_predicted_action_frames" in str(exc)
    else:
        raise AssertionError("Expected a changed confidence aggregation to fail.")


def test_spotta_header_config_contains_only_effective_recipe_options():
    with patch(
        "opensportslib.core.config.loader._dali_available", return_value=False
    ):
        config = load_config(
            "opensportslib/configs/localization/e2e_spotta_header.yaml",
            as_namespace=False,
        )
    policy = config["MODEL"]["policies"]["test_time_adaptation"]

    assert policy["enabled"] is True
    assert policy["confidence_gate"]["threshold"] == 0.3
    assert policy["memory"]["capacity"] == 8
    assert policy["memory"]["update_frequency"] == 2
    assert policy["robust_bn"]["tether"]["mode"] == "bayesian"
    assert "frame_filter" not in policy
    assert "steps" not in policy
    assert "reset_frequency" not in policy
    assert "action_frame_weight" not in policy


def test_localization_starts_fresh_spotta_stream_and_forces_opencv_runtime():
    from opensportslib.apis.localization import LocalizationModel

    configured = []

    class _Model:
        def configure_test_time_adaptation(self, policy):
            configured.append(policy)

    policy = SimpleNamespace(enabled=True, name="spotta")
    api = LocalizationModel.__new__(LocalizationModel)
    api.model = _Model()
    api.config = SimpleNamespace(
        MODEL=SimpleNamespace(
            metadata=SimpleNamespace(family="E2E"),
            policies=SimpleNamespace(test_time_adaptation=policy),
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

    assert configured == [policy]
    assert get_loader_backend(api.config) == "opencv"
