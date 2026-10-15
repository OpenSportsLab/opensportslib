"""Proves classification training can resume from a checkpoint.

Before this, `Trainer_Classification.load()`/`.train()` restored only the
epoch counter from a checkpoint: `.train()` always called `build_optimizer()`
/`build_scheduler()` fresh, so Adam's momentum and the scheduler's internal
state (e.g. StepLR's `last_epoch`) silently reset on every resume even though
`_save_checkpoint()` had been writing them out all along.

These tests exercise the real optimizer/scheduler objects and the real
checkpoint file on disk (no mocking of torch internals) so a regression that
reintroduces "resume rebuilds everything from scratch" fails loudly instead
of merely failing to crash.
"""

from types import SimpleNamespace

import pytest
import torch

from opensportslib.core.loss.ce import CELoss
from opensportslib.core.trainer import classification_trainer
from opensportslib.core.trainer.classification_trainer import (
    BaseTrainerClassification,
    Trainer_Classification,
)


def _ns(obj):
    """Recursively turn a nested dict/list literal into a SimpleNamespace tree.

    opensportslib.core.config.accessors walks configs generically via
    vars(obj); a real OmegaConf DictConfig's child nodes carry a `_parent`
    back-reference, so that walk cycles forever on nested OmegaConf configs.
    SimpleNamespace has no such cycle, and this is the same fake-config style
    tests/test_task_model_api_contract.py already uses for LocalizationModel.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


class _ListLoader:
    """Minimal DataLoader stand-in.

    BaseTrainerClassification.train() probes `self.train_loader.sampler` for
    a DistributedSampler-style `set_epoch()`, so a plain list won't do --
    it needs a `.sampler` attribute (None is fine) plus `__len__`/`__iter__`.
    """

    def __init__(self, batches):
        self._batches = batches
        self.sampler = None

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class _TinyTrainer(BaseTrainerClassification):
    """Concrete trainer with a trivial forward pass: batch["x"] -> model."""

    def _forward_batch(self, batch):
        return self.model(batch["x"]), batch["labels"]


def _make_batches(n_batches=3, batch_size=4, in_features=3):
    torch.manual_seed(0)
    return [
        {
            "x": torch.randn(batch_size, in_features),
            "labels": torch.randint(0, 2, (batch_size,)),
            "id": [f"s{b}_{i}" for i in range(batch_size)],
        }
        for b in range(n_batches)
    ]


def _resume_config(save_dir, epochs=5):
    return _ns(
        {
            "TRAIN": {
                "optimizer": {"type": "Adam", "lr": 0.05},
                "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.5},
                "epochs": epochs,
            },
            "SYSTEM": {"paths": {"save_dir": str(save_dir)}},
        }
    )


def test_resume_training_restores_optimizer_and_scheduler_state(tmp_path, monkeypatch):
    """Train one real epoch, save, then resume via
    Trainer_Classification.load(..., resume_training=True) and check the
    restored optimizer/scheduler carry the trained state -- not a
    freshly-initialized one.
    """
    # compute_classification_metrics pulls in the `evaluate` library's
    # network-backed metric scripts; that's orthogonal to what this test
    # proves (optimizer/scheduler restoration), so stub it out.
    monkeypatch.setattr(
        classification_trainer,
        "compute_classification_metrics",
        lambda *args, **kwargs: {"balanced_accuracy": 0.7},
    )
    # load_checkpoint() probes huggingface_hub.whoami() before falling back
    # to the local path; keep that offline/instant for the test.
    monkeypatch.setattr(
        "huggingface_hub.whoami",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no auth in tests")),
    )

    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    config = _resume_config(tmp_path, epochs=5)

    trainer = _TinyTrainer(
        train_loader=_ListLoader(_make_batches()),
        val_loader=_ListLoader(_make_batches()),
        test_loader=None,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=CELoss(),
        class_weights=None,
        class_names={0: "a", 1: "b"},
        save_dir=str(tmp_path),
        model_name="tiny",
        max_epochs=1,
        device="cpu",
        top_k=1,
        monitor="balanced_accuracy",
        mode="max",
        config=config,
    )

    # Sanity: a real Adam optimizer has no per-param state until .step() runs.
    assert optimizer.state_dict()["state"] == {}

    trainer.train(epoch_start=0, save_every=1)

    # One epoch, one improvement over -inf -> exactly one "best" checkpoint.
    assert trainer.best_checkpoint_path is not None
    assert optimizer.state_dict()["state"] != {}

    saved = torch.load(trainer.best_checkpoint_path, weights_only=False)
    assert saved["epoch"] == 1
    assert saved["best_metric"] == pytest.approx(0.7)

    # --- resume into a brand-new Trainer_Classification/model pair ---
    resumed_model = torch.nn.Linear(3, 2)
    resumer = Trainer_Classification(config)
    resumer.model = resumed_model  # skip build_model(); nothing else needs it
    resumer.device = torch.device("cpu")

    resumer.load(trainer.best_checkpoint_path, resume_training=True)

    assert resumer.epoch == 1
    assert resumer._resume_state["epoch"] == 1
    assert resumer._resume_state["best_metric"] == pytest.approx(0.7)
    assert resumer._resume_state["optimizer"] is resumer.optimizer
    assert resumer._resume_state["scheduler"] is resumer.scheduler

    restored_state = resumer.optimizer.state_dict()["state"]
    trained_state = saved["optimizer"]["state"]
    assert restored_state != {}
    assert len(restored_state) == len(trained_state)
    for key, trained_entry in trained_state.items():
        restored_entry = restored_state[key]
        for field, value in trained_entry.items():
            if torch.is_tensor(value):
                assert torch.equal(restored_entry[field], value)
            else:
                assert restored_entry[field] == value

    assert resumer.scheduler.state_dict() == saved["scheduler"]

    # The crucial negative: a *freshly built* optimizer over the same model
    # is not what we got -- proving the restore is doing real work, not
    # just failing to crash.
    fresh_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.05)
    assert fresh_optimizer.state_dict()["state"] == {}
    assert fresh_optimizer.state_dict()["state"] != restored_state


class _FakeInnerTrainer:
    """Stand-in for MVTrainerClassification: records what it was built with
    and what epoch_start .train() was told to continue from."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.epoch_start = None
        self.best_checkpoint_path = "fake-best.pt"

    def train(self, epoch_start=0, save_every=1):
        del save_every
        self.epoch_start = epoch_start


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


def _plumbing_config(epochs=5):
    dataloader = {
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
    }
    return _ns(
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
                "optimizer": {"type": "Adam", "lr": 0.1},
                "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.1},
                "criterion": {"type": "CrossEntropyLoss"},
                "epochs": epochs,
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


def test_train_with_resume_from_reuses_state_instead_of_rebuilding(monkeypatch):
    """Plumbing check: when `resume_from` is given, .train() must reuse its
    optimizer/scheduler (not call build_optimizer()/build_scheduler() again),
    hand them unchanged to the inner trainer, continue from the saved epoch,
    and seed best_metric tracking -- mirroring build_trainer()'s resume_from
    handling in localization_trainer.py.
    """
    build_optimizer_calls = []
    build_scheduler_calls = []

    monkeypatch.setattr(classification_trainer, "select_device", lambda system: torch.device("cpu"))
    monkeypatch.setattr(
        classification_trainer,
        "DataLoader",
        lambda dataset, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(classification_trainer, "MVTrainerClassification", _FakeInnerTrainer)
    monkeypatch.setattr(
        "opensportslib.core.optimizer.builder.build_optimizer",
        lambda params, cfg: build_optimizer_calls.append(cfg) or object(),
    )
    monkeypatch.setattr(
        "opensportslib.core.scheduler.builder.build_scheduler",
        lambda optimizer, cfg: build_scheduler_calls.append(cfg) or object(),
    )
    monkeypatch.setattr(
        "opensportslib.core.loss.builder.build_criterion",
        lambda cfg: object(),
    )

    sentinel_optimizer = object()
    sentinel_scheduler = object()
    resume_from = {
        "optimizer": sentinel_optimizer,
        "scheduler": sentinel_scheduler,
        "epoch": 2,
        "best_metric": 0.42,
    }

    trainer = Trainer_Classification(_plumbing_config(epochs=5))
    trainer.train(
        torch.nn.Linear(1, 1), _FakeDataset(), _FakeDataset(), resume_from=resume_from
    )

    assert build_optimizer_calls == []
    assert build_scheduler_calls == []
    assert trainer.trainer.kwargs["optimizer"] is sentinel_optimizer
    assert trainer.trainer.kwargs["scheduler"] is sentinel_scheduler
    assert trainer.epoch == 2
    assert trainer.trainer.epoch_start == 2
    assert trainer.trainer.best_metric == 0.42


def test_train_with_resume_from_past_target_epochs_raises(monkeypatch):
    """Mirrors localization's build_trainer() guard: resuming a checkpoint
    that already reached TRAIN.epochs must fail loudly instead of silently
    running a zero-epoch "training" that looks like success.
    """
    monkeypatch.setattr(classification_trainer, "select_device", lambda system: torch.device("cpu"))

    resume_from = {
        "optimizer": object(),
        "scheduler": object(),
        "epoch": 5,
        "best_metric": None,
    }

    trainer = Trainer_Classification(_plumbing_config(epochs=5))
    with pytest.raises(ValueError, match="TRAIN.epochs"):
        trainer.train(
            torch.nn.Linear(1, 1), _FakeDataset(), _FakeDataset(), resume_from=resume_from
        )


def test_train_without_resume_from_still_builds_fresh_optimizer(monkeypatch):
    """The untouched default path: no resume_from -> build_optimizer()/
    build_scheduler() are called exactly as before this change, and training
    starts at self.epoch (0 for a Trainer_Classification that never loaded a
    checkpoint)."""
    build_optimizer_calls = []
    build_scheduler_calls = []

    monkeypatch.setattr(classification_trainer, "select_device", lambda system: torch.device("cpu"))
    monkeypatch.setattr(
        classification_trainer,
        "DataLoader",
        lambda dataset, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(classification_trainer, "MVTrainerClassification", _FakeInnerTrainer)
    monkeypatch.setattr(
        "opensportslib.core.optimizer.builder.build_optimizer",
        lambda params, cfg: build_optimizer_calls.append(cfg) or object(),
    )
    monkeypatch.setattr(
        "opensportslib.core.scheduler.builder.build_scheduler",
        lambda optimizer, cfg: build_scheduler_calls.append(cfg) or object(),
    )
    monkeypatch.setattr(
        "opensportslib.core.loss.builder.build_criterion",
        lambda cfg: object(),
    )

    trainer = Trainer_Classification(_plumbing_config(epochs=5))
    trainer.train(torch.nn.Linear(1, 1), _FakeDataset(), _FakeDataset())

    assert len(build_optimizer_calls) == 1
    assert len(build_scheduler_calls) == 1
    assert trainer.epoch == 0
    assert trainer.trainer.epoch_start == 0
    assert not hasattr(trainer.trainer, "best_metric") or trainer.trainer.best_metric is None
