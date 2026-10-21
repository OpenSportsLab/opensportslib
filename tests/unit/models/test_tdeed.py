from pathlib import Path
import json

import numpy as np
import pytest
import torch

from opensportslib.datasets.tdeed import displacement_targets
from opensportslib.metrics.tdeed import frame_map, suppress
from opensportslib.models.base.tdeed import relocate_scores


def test_relocation_rounding_clipping_collisions_and_empty_destinations():
    logits = torch.tensor([[[0.0, 2.0], [3.0, 0.0], [0.0, 4.0], [1.0, 0.0]]])
    offset = torch.tensor([[100.0, 1.0, 0.0, -100.0]])
    got = relocate_scores(logits, offset)
    probabilities = logits.softmax(-1)
    torch.testing.assert_close(got[0, 0], probabilities[0, :2].amax(0))
    assert not got[0, 1].any()
    torch.testing.assert_close(got[0, 2:], probabilities[0, 2:])
    assert got[0, 0].sum() > 1  # Classwise max votes must not be normalized.
    with pytest.raises(ValueError, match="finite"):
        relocate_scores(logits, torch.full((1, 4), float("nan")))


def test_displacement_targets_use_floor_stride_last_annotation_and_signed_offsets():
    events = [{"label": "a", "frame": 3}, {"label": "b", "frame": 5}]
    labels, offset = displacement_targets(
        events, {"a": 1, "b": 2}, -2, 6, stride=2, radius=1
    )
    np.testing.assert_array_equal(labels, [0, 1, 2, 2, 2, 0])
    np.testing.assert_array_equal(offset, [0, -1, -1, 0, 1, 0])
    labels, offset = displacement_targets(
        events, {"a": 1, "b": 2}, 0, 7, radius=0, dilation=1
    )
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2, 2])
    assert not offset.any()


def test_pes_ap_and_greedy_suppression_keep_protocol_separate_from_soccernet():
    truth = [
        {
            "video": "v",
            "events": [{"frame": 4, "label": "a"}, {"frame": 9, "label": "a"}],
        }
    ]
    events = [
        {"frame": 0, "label": "a", "score": 0.99},
        {"frame": 5, "label": "a", "score": 0.9},
        {"frame": 6, "label": "a", "score": 0.8},
        {"frame": 9, "label": "a", "score": 0.7},
    ]
    result = frame_map(truth, [{"video": "v", "events": events}], [0, 1])
    assert result["tolerances"]["0"]["mAP"] == 0.125
    assert result["tolerances"]["1"]["mAP"] == 0.5
    hard = suppress(events, 1)
    assert [e["frame"] for e in hard] == [0, 5, 9]
    soft = suppress(events, 2, soft=True)
    assert next(e["score"] for e in soft if e["frame"] == 6) == pytest.approx(0.2)
    assert events[2]["score"] == 0.8
    with pytest.raises(ValueError, match="sets differ"):
        frame_map(truth, [])


def test_released_configs_match_pinned_hyperparameters():
    from opensportslib.core.config.editable import Config

    root = Path(__file__).parents[3]
    for ref in (root / "reproduction/tdeed/reference_configs").glob("*.json"):
        expected = json.loads(ref.read_text())
        cfg = Config.from_file(
            root / "opensportslib/configs/localization/tdeed" / f"{ref.stem}.yaml"
        ).get_config()
        head = cfg["MODEL"]["components"]["task_head"]["params"]
        for key in [
            "radi_displacement",
            "n_layers",
            "sgp_ks",
            "sgp_r",
            "temporal_arch",
        ]:
            assert head[key] == expected[key]
        assert cfg["TRAIN"]["epochs"] == expected["num_epochs"]
        assert cfg["TRAIN"]["scheduler"]["warm_up_epochs"] == expected["warm_up_epochs"]
        assert (
            cfg["DATA"]["inputs"]["video"]["sampling"]["clip_len"]
            == expected["clip_len"]
        )


def test_joint_head_loss_uses_only_selected_dataset_head():
    from opensportslib.models.base.tdeed import TDEEDModel
    from types import SimpleNamespace

    model = TDEEDModel.__new__(TDEEDModel)
    model.device = torch.device("cpu")
    model._num_classes = 2
    model.args = SimpleNamespace(auxiliary_num_classes=2)
    logits = torch.tensor(
        [[[0.2, 0.8, 1.0, 2.0, 3.0]], [[0.9, 0.1, 3.0, 2.0, 1.0]]], requires_grad=True
    )

    class Fixed(torch.nn.Module):
        _double_head = True

        def forward(self, x, **kwargs):
            return logits, None

    model._model = Fixed()
    batch = {
        "frame": torch.zeros((2, 1, 3, 2, 2)),
        "label": torch.tensor([[1], [2]]),
        "dataset": torch.tensor([1, 2]),
    }
    loss = model.loss(batch, training=False)
    expected = (
        torch.nn.functional.cross_entropy(
            logits[0, :, :2], torch.tensor([1]), weight=torch.tensor([1.0, 5.0])
        )
        + torch.nn.functional.cross_entropy(
            logits[1, :, 2:], torch.tensor([2]), weight=torch.tensor([1.0, 5.0, 5.0])
        )
    ) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert not logits.grad[0, :, 2:].any() and not logits.grad[1, :, :2].any()


def test_deterministic_resampling_backward_matches_analytic_gradients():
    from opensportslib.models.utils.impl.tdeed.deterministic import (
        _Linear,
        _AdaptiveMax,
    )

    torch.manual_seed(19)
    for length, output in [(1, 1), (3, 7), (7, 4), (25, 50)]:
        x = torch.randn(2, 3, length, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda value: _Linear.apply(value, output), (x,)
        )
        y = x.detach().clone().requires_grad_(True)
        upstream = torch.randn(2, 3, output, dtype=torch.float64)
        _Linear.apply(x, output).backward(upstream)
        torch.nn.functional.interpolate(
            y, size=output, mode="linear", align_corners=True
        ).backward(upstream)
        torch.testing.assert_close(x.grad, y.grad)
    x = torch.randn(2, 3, 9, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda value: _AdaptiveMax.apply(value, 5), (x,))


@pytest.mark.parametrize(
    "variant",
    [
        "plain_gru",
        "plain_sgp",
        "plain_transformer",
        "skip_none",
        "skip_sum",
        "skip_concat",
        "sgp_mixer_sum",
    ],
)
def test_temporal_ablation_preserves_resolution_and_backpropagates(variant):
    from opensportslib.models.utils.impl.tdeed.ablations import build_temporal

    torch.set_num_threads(2)
    model = build_temporal(variant, 32, 25, 2, 3, 2)
    value = torch.randn(2, 25, 32, requires_grad=True)
    output = model(value)
    assert output.shape == value.shape
    output.square().mean().backward()
    assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_finegym_start_protocol_includes_interior_vault_events():
    from opensportslib.metrics.tdeed import finegym_start_classes

    assert finegym_start_classes(
        ["BB_turns_start", "BB_turns_end", "VT_0", "VT_1", "VT_2", "VT_3"]
    ) == ["BB_turns_start", "VT_1", "VT_2"]
