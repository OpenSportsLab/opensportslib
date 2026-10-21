from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from opensportslib.core.trainer import classification_trainer


class _FakeTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train(self, epoch_start=0, save_every=1):
        del epoch_start, save_every


class _FakeDataset:
    label_map = {0: "PASS"}

    def __len__(self):
        return 1

    def num_classes(self):
        return 1

    def get_class_weights(self, num_classes=None, sqrt=False):
        del num_classes, sqrt
        return torch.ones(1)

    def get_sample_weights(self):
        return torch.ones(1)


def _make_config(mp_context=None):
    dataloader = {
        "batch_size": 1,
        "num_workers": 1,
        "pin_memory": False,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    if mp_context is not None:
        dataloader["mp_context"] = mp_context

    return OmegaConf.create(
        {
            "TASK": "classification",
            "VERSION": 3,
            "DATA": {
                "common": {
                    "dataset_name": "smoke",
                    "data_root": ".",
                    "classes": ["PASS"],
                    "runtime": {"loader_backend": "opencv"},
                    "splits": {
                        "train": {"dataloader": dict(dataloader)},
                        "valid": {"dataloader": dict(dataloader)},
                    },
                },
                "inputs": {
                    "video": {
                        "modality": "video",
                        "representation": "raw",
                        "source": {"format": "mp4"},
                        "sampling": {},
                        "transform": {},
                        "augmentations": {},
                        "params": {},
                    }
                },
            },
            "MODEL": {
                "schema_version": 3,
                "task": "classification",
                "components": {
                    "video_encoder": {
                        "kind": "encoder",
                        "source": {
                            "provider": "opensportslib",
                            "registry": "backbone",
                            "name": "smoke_backbone",
                        },
                        "params": {},
                        "overrides": {},
                    }
                },
                "topology": [],
            },
            "TRAIN": {
                "trainer": {"type": "classification"},
                "optimizer": {"type": "SGD", "lr": 0.1},
                "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.1},
                "criterion": {"type": "CrossEntropyLoss"},
                "epochs": 1,
                "sampling": {
                    "use_weighted_loss": False,
                    "use_weighted_sampler": False,
                },
                "checkpoint": {"save_every": 1},
            },
            "SYSTEM": {
                "reproducibility": {"seed": 0, "use_seed": False},
                "device": "cpu",
                "paths": {"save_dir": "."},
            },
        }
    )


def _run_train(monkeypatch, config):
    dataloader_calls = []

    monkeypatch.setattr(
        classification_trainer,
        "select_device",
        lambda system: torch.device("cpu"),
    )
    monkeypatch.setattr(
        classification_trainer,
        "DataLoader",
        lambda dataset, **kwargs: dataloader_calls.append(kwargs) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        classification_trainer,
        "MVTrainerClassification",
        _FakeTrainer,
    )
    monkeypatch.setattr(
        "opensportslib.core.optimizer.builder.build_optimizer",
        lambda params, cfg: object(),
    )
    monkeypatch.setattr(
        "opensportslib.core.scheduler.builder.build_scheduler",
        lambda optimizer, cfg: object(),
    )
    monkeypatch.setattr(
        "opensportslib.core.loss.builder.build_criterion",
        lambda cfg: object(),
    )

    trainer = classification_trainer.Trainer_Classification(config)
    trainer.train(torch.nn.Linear(1, 1), _FakeDataset(), _FakeDataset())

    return dataloader_calls


def test_video_train_loader_respects_explicit_spawn_context(monkeypatch):
    dataloader_calls = _run_train(monkeypatch, _make_config("spawn"))

    assert len(dataloader_calls) == 2
    assert dataloader_calls[0]["num_workers"] == 1
    assert dataloader_calls[0]["pin_memory"] is False
    assert (dataloader_calls[0]["multiprocessing_context"].get_start_method() == "spawn")
    assert (dataloader_calls[1]["multiprocessing_context"].get_start_method() == "spawn")
    


def test_video_train_loader_respects_explicit_context(monkeypatch):
    dataloader_calls = _run_train(monkeypatch, _make_config("forkserver"))

    assert len(dataloader_calls) == 2
    assert (dataloader_calls[0]["multiprocessing_context"].get_start_method() == "forkserver")
    assert (dataloader_calls[1]["multiprocessing_context"].get_start_method() == "forkserver")


def test_classification_dispatches_validation_stopping_settings(monkeypatch):
    captured = {}
    monkeypatch.setattr(_FakeTrainer, "__init__", lambda self, **kwargs: captured.update(kwargs))
    config = _make_config()
    config.TRAIN.selection = {"monitor": "loss", "mode": "min", "patience": 12, "min_delta": .002}
    _run_train(monkeypatch, config)
    assert captured["patience"] == 12 and captured["min_delta"] == .002


@pytest.fixture
def cpu_trainer(tmp_path, monkeypatch):
    from opensportslib.core.loss.builder import build_criterion
    from torch.utils.data import DataLoader

    monkeypatch.setattr(classification_trainer.wandb, "watch", lambda *args, **kwargs: None)
    monkeypatch.setattr(classification_trainer.wandb, "run", None)
    monkeypatch.setattr(classification_trainer, "log_confusion_matrix_wandb", lambda **kwargs: None)
    data = [{"id": f"clip-{i}", "pixel_values": torch.eye(3)[i % 3], "labels": i % 3} for i in range(6)]
    loader = DataLoader(data, batch_size=3, shuffle=False)
    model = torch.nn.Linear(3, 3)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3))
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    return classification_trainer.MVTrainerClassification(
        train_loader=loader, val_loader=loader, test_loader=loader,
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=build_criterion({"type": "CrossEntropyLoss"}),
        class_weights=None, class_names={0: "PASS", 1: "SHOT", 2: "HEADER"},
        save_dir=str(tmp_path), model_name="tiny", max_epochs=2, device="cpu",
        monitor="loss", mode="min",
    )


@pytest.mark.parametrize("mode,values,patience,min_delta,completed,best_epoch,reason", [
    ("min", [1., .8, .8, .9, .7], 2, 0., 4, 2, "validation_plateau"),
    ("max", [.1, .5, .49, .49, .8], 2, 0., 4, 2, "validation_plateau"),
    ("min", [1., .96, .92, .88, .87, .86, .85, .1], 3, .1, 7, 7, "validation_plateau"),
    ("min", [1., 1., 1., 1.], 0, 0., 4, 1, "max_epochs"),
])
def test_validation_plateau_stops_and_keeps_strict_best_checkpoint(
        cpu_trainer, monkeypatch, mode, values, patience, min_delta, completed, best_epoch, reason):
    import json
    from pathlib import Path

    trainer = cpu_trainer
    trainer.mode, trainer.monitor = mode, "loss" if mode == "min" else "balanced_accuracy"
    trainer.patience, trainer.min_delta, trainer.max_epochs = patience, min_delta, len(values)
    seen = []

    def scripted_epoch(loader, epoch, *, train, **kwargs):
        if train:
            seen.append(epoch)
            with torch.no_grad():
                trainer.model.weight.fill_(epoch)
            trainer.optimizer.step()
        return [], [], values[epoch-1], {"balanced_accuracy": values[epoch-1]}

    monkeypatch.setattr(trainer, "_run_epoch", scripted_epoch)
    trainer.train()
    assert seen == list(range(1, completed+1))
    checkpoint = torch.load(trainer.best_checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == best_epoch
    assert checkpoint["best_metric"] == values[best_epoch-1]
    torch.testing.assert_close(checkpoint["state_dict"]["weight"], torch.full_like(trainer.model.weight, best_epoch))
    summary = json.loads((Path(trainer.save_dir) / "training_summary.json").read_text())
    assert summary["epochs_completed"] == completed and summary["stop_reason"] == reason
    assert len(summary["history"]) == completed
    assert summary["min_delta"] == min_delta and summary["patience"] == patience


@pytest.mark.parametrize("min_lr,completed", [(0., 3), (.025, 2)])
def test_lr_reduction_and_minimum_lr_save_the_evaluated_best_weights(cpu_trainer, monkeypatch, min_lr, completed):
    trainer = cpu_trainer
    trainer.max_epochs, trainer.monitor, trainer.mode = 3, "balanced_accuracy", "max"
    trainer.revert_on_lr_reduction = True
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode="min", patience=0, factor=.5, min_lr=min_lr)

    def scripted_epoch(loader, epoch, *, train, **kwargs):
        if train:
            with torch.no_grad():
                trainer.model.weight.fill_(epoch)
            trainer.optimizer.step()
        return [], [], 1., {"balanced_accuracy": epoch / 10}

    monkeypatch.setattr(trainer, "_run_epoch", scripted_epoch)
    trainer.train()
    assert trainer.epochs_completed == completed
    assert trainer.stop_reason == ("minimum_learning_rate" if min_lr else "max_epochs")
    checkpoint = torch.load(trainer.best_checkpoint_path, weights_only=False)
    assert checkpoint["epoch"] == completed
    torch.testing.assert_close(checkpoint["state_dict"]["weight"], torch.full_like(trainer.model.weight, completed))


@pytest.mark.parametrize("weighted", [False, True])
def test_cpu_training_checkpoint_resumes_identical_next_update(cpu_trainer, weighted):
    from opensportslib.core.utils.checkpoint import load_checkpoint

    trainer = cpu_trainer
    if weighted:
        trainer.class_weights = torch.tensor([1., 2., 3.])
    before = trainer.model.weight.detach().clone()
    trainer.train()
    assert not torch.equal(before, trainer.model.weight)
    assert trainer.best_checkpoint_path is not None
    restored = torch.nn.Linear(3, 3)
    optimizer = torch.optim.SGD(restored.parameters(), lr=.9, momentum=.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    _, _, _, _, epoch, checkpoint = load_checkpoint(
        restored, trainer.best_checkpoint_path, optimizer=optimizer,
        scheduler=scheduler, device=torch.device("cpu"),
    )
    assert epoch == 2
    assert checkpoint["best_metric"] == pytest.approx(trainer.best_metric)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.025)
    assert scheduler.state_dict() == trainer.scheduler.state_dict()
    assert optimizer.state  # Momentum must be populated, not an empty-state round trip.
    batch = next(iter(trainer.train_loader))
    expected_before = trainer.model(batch["pixel_values"]).detach()
    torch.testing.assert_close(restored(batch["pixel_values"]), expected_before)
    trainer._process_batch(batch, train=True)
    optimizer.zero_grad()
    loss = trainer.criterion(output=restored(batch["pixel_values"]), labels=batch["labels"], weight=trainer.class_weights)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(restored.parameters(), 1.)
    optimizer.step()
    for actual, expected in zip(restored.parameters(), trainer.model.parameters()):
        torch.testing.assert_close(actual, expected)
    test_loss, scores = trainer.test()
    assert test_loss > 0 and scores["accuracy"] == 1.


@pytest.mark.parametrize("labelled", [False, True])
def test_evaluation_keeps_weights_and_exports_sample_predictions(cpu_trainer, labelled):
    trainer = cpu_trainer
    batch = next(iter(trainer.test_loader))
    if not labelled:
        batch.pop("labels")
    original = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    logits, labels, loss, scores = trainer._run_epoch([batch], epoch="check", set_name="test")
    assert logits.shape == (3, 3)
    assert len(labels) == (3 if labelled else 0)
    assert loss > 0 if labelled else loss == 0
    assert scores["accuracy"] == 1. if labelled else scores == {}
    assert [item["id"] for item in trainer.predictions_payload["data"]] == batch["id"]
    assert [item["labels"]["action"]["label"] for item in trainer.predictions_payload["data"]] == ["PASS", "SHOT", "HEADER"]
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_empty_evaluation_returns_empty_predictions(cpu_trainer):
    logits, labels, loss, scores = cpu_trainer._run_epoch([], epoch="empty", set_name="test")
    assert len(logits) == len(labels) == 0
    assert loss == 0 and scores == {}
    assert cpu_trainer.predictions_payload["data"] == []


@pytest.mark.parametrize("canonical", [True, False])
def test_frames_trainer_resolves_amp_and_mixup_configuration(cpu_trainer, monkeypatch, canonical):
    from types import SimpleNamespace

    captured = []
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: captured.append(kwargs["enabled"]))
    train = {"use_amp": not canonical, "mixup_alpha": .7}
    if canonical:
        train["execution"] = SimpleNamespace(use_amp=True, mixup_alpha=.2)
    config = SimpleNamespace(TRAIN=SimpleNamespace(**train))
    base = cpu_trainer
    trainer = classification_trainer.FramesTrainerClassification(
        train_loader=base.train_loader, val_loader=base.val_loader, test_loader=base.test_loader,
        model=base.model, optimizer=base.optimizer, scheduler=base.scheduler,
        criterion=base.criterion, class_weights=None, class_names=base.class_names,
        save_dir=base.save_dir, model_name="frames", device="cpu", config=config)
    assert trainer.use_amp is True and captured == [True]
    assert trainer.mixup_alpha == pytest.approx(.2 if canonical else .7)
