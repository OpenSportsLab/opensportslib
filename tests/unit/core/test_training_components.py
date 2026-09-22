"""Fast contracts for losses, optimizers, schedulers, and checkpoints."""

from __future__ import annotations

import torch

from opensportslib.core.loss.builder import build_criterion
from opensportslib.core.loss.combine import Combined2x
from opensportslib.core.loss.nll import NLLLoss
from opensportslib.core.optimizer.builder import build_optimizer
from opensportslib.core.scheduler.builder import build_scheduler
from opensportslib.core.utils.checkpoint import load_checkpoint, save_checkpoint


def test_cross_entropy_backward_optimizer_and_scheduler_step():
    model = torch.nn.Linear(3, 2)
    criterion = build_criterion({"type": "CrossEntropyLoss"})
    optimizer = build_optimizer(model.parameters(), {"type": "SGD", "lr": 0.1})
    scheduler = build_scheduler(
        optimizer,
        {"type": "StepLR", "step_size": 1, "gamma": 0.5},
    )

    loss = criterion(model(torch.ones(2, 3)), torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.05


def test_checkpoint_roundtrip_restores_model_optimizer_and_epoch(tmp_path):
    original = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(original.parameters(), lr=0.25, momentum=0.9)
    expected = original(torch.tensor([[1.0, 2.0]])).detach()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(original, str(path), optimizer=optimizer, epoch=7)

    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01, momentum=0.9)
    loaded = load_checkpoint(
        restored,
        str(path),
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
    )

    assert loaded[4] == 7
    assert torch.equal(restored(torch.tensor([[1.0, 2.0]])).detach(), expected)
    assert restored_optimizer.param_groups[0]["lr"] == 0.25


def test_legacy_loss_helpers_produce_weighted_finite_values():
    labels = (torch.tensor([1.0]), torch.tensor([0.0]))
    outputs = (torch.tensor([0.8]), torch.tensor([0.2]))
    combined = Combined2x(torch.nn.L1Loss(), torch.nn.L1Loss(), 2.0, 3.0)
    assert torch.isclose(combined(labels, outputs), torch.tensor(1.0))

    nll = NLLLoss()
    loss = nll(torch.tensor([1.0, 0.0]), torch.tensor([0.8, 0.2]))
    assert torch.isfinite(loss)
    assert loss > 0
