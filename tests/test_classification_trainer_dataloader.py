from types import SimpleNamespace

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
            "DATA": {
                "data_modality": "video",
                "train": {"dataloader": dict(dataloader)},
                "valid": {"dataloader": dict(dataloader)},
            },
            "MODEL": {
                "type": "custom",
                "backbone": {"type": "smoke_backbone"},
            },
            "TRAIN": {
                "use_weighted_loss": False,
                "use_weighted_sampler": False,
                "optimizer": {"type": "SGD", "lr": 0.1},
                "scheduler": {"type": "StepLR", "step_size": 1, "gamma": 0.1},
                "criterion": {"type": "CrossEntropyLoss"},
                "epochs": 1,
                "save_every": 1,
            },
            "SYSTEM": {
                "seed": 0,
                "device": "cpu",
                "save_dir": ".",
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