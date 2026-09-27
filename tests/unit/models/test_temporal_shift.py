"""CPU contracts for the legacy temporal-shift model utility."""

from __future__ import annotations

import torch

from opensportslib.models.utils.impl.tsm import TemporalShift


def test_temporal_shift_moves_channel_folds_in_expected_directions():
    # One batch, three temporal positions, six channels, one spatial cell.
    values = torch.arange(18, dtype=torch.float32).reshape(3, 6, 1, 1)
    shifted = TemporalShift.shift(values, n_segment=3, fold_div=3, inplace=False)

    # First fold shifts left, second shifts right, remaining channels stay put.
    assert torch.equal(shifted[0, :2], values[1, :2])
    assert torch.equal(shifted[2, 2:4], values[1, 2:4])
    assert torch.equal(shifted[:, 4:], values[:, 4:])


def test_temporal_shift_wrapper_preserves_shape_and_backpropagates(capsys):
    layer = TemporalShift(torch.nn.Identity(), n_segment=2, n_div=2, inplace=False)
    values = torch.randn(4, 4, 2, 2, requires_grad=True)
    output = layer(values)
    output.sum().backward()

    assert output.shape == values.shape
    assert values.grad is not None
    assert "fold div" in capsys.readouterr().out
