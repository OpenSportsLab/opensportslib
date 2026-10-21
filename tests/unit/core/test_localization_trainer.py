"""Validation stopping and checkpoint contracts for the native E2E trainer."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from opensportslib.core.trainer import localization_trainer


@pytest.fixture
def e2e_trainer(tmp_path, monkeypatch):
    from opensportslib.apis import Config
    from opensportslib.core.config.runtime_adapter import dict_to_namespace

    monkeypatch.setattr(localization_trainer.wandb, "watch", lambda *args, **kwargs: None)
    monkeypatch.setattr(localization_trainer.wandb, "run", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class EpochModel:
        def __init__(self, values):
            self._model = torch.nn.Linear(1, 1)
            self.values, self.trained = values, 0

        def epoch(self, loader, dali, optimizer=None, *args, **kwargs):
            if optimizer is not None:
                self.trained += 1
                with torch.no_grad():
                    self._model.weight.fill_(self.trained)
            return self.values[self.trained - 1]

        def state_dict(self):
            return self._model.state_dict()

    def build(values, *, patience=2, min_delta=0., criterion="loss", map_every=2):
        payload = Config.from_file("examples/tennis/localization_skeleton.yaml").get_config()
        payload["TRAIN"]["epochs"] = len(values)
        payload["TRAIN"]["selection"].update(patience=patience, min_delta=min_delta)
        payload["TRAIN"]["execution"].update(criterion_valid=criterion, valid_map_every=map_every)
        model = EpochModel(values)
        optimizer = torch.optim.SGD(model._model.parameters(), lr=.01)
        return localization_trainer.Trainer_e2e(
            dict_to_namespace(payload), model, optimizer, torch.amp.GradScaler("cpu", enabled=False),
            torch.optim.lr_scheduler.StepLR(optimizer, 1), str(tmp_path), False)

    return build


@pytest.mark.parametrize("values,patience,min_delta,completed,selected", [
    ([1., .7, .71, .72, .6], 2, 0., 4, 2),
    ([1., .96, .92, .88, .87, .86, .85, .1], 3, .1, 7, 7),
    ([1., 1., 1.], 0, 0., 3, 1),
])
def test_e2e_validation_plateau_preserves_selected_and_latest_checkpoints(
        e2e_trainer, values, patience, min_delta, completed, selected):
    trainer = e2e_trainer(values, patience=patience, min_delta=min_delta)
    trainer.train([], [], ["swing"])
    summary = json.loads((Path(trainer.save_dir) / "training_summary.json").read_text())
    assert summary["epochs_completed"] == completed and summary["selected_epoch"] == selected
    assert summary["stop_reason"] == ("validation_plateau" if patience else "max_epochs")
    assert [row["epoch"] for row in summary["history"]] == list(range(1, completed+1))
    best = torch.load(trainer.best_checkpoint_path, weights_only=False)
    latest = torch.load(Path(trainer.save_dir) / "latest_checkpoint.pt", weights_only=False)
    assert best["epoch"] == selected-1 and latest["epoch"] == completed-1
    assert best["best_criterion_valid"] == values[selected-1]
    # Final validation uses the actual selected model, including tiny gains on
    # the stopping epoch; checkpoint epoch numbers retain their existing base.
    assert trainer.model._model.weight.item() == selected


def test_e2e_map_patience_counts_only_measurements_and_keeps_zero_map(e2e_trainer, monkeypatch):
    trainer = e2e_trainer([1.] * 7, patience=1, criterion="map", map_every=2)
    trainer.cfg_valid_data_frames = SimpleNamespace(dataloader=None)
    dataset = SimpleNamespace()
    monkeypatch.setattr(localization_trainer, "build_dataset", lambda *args, **kwargs:
                        SimpleNamespace(cfg=None, building_dataset=lambda *args, **kwargs: dataset))
    measured = []
    monkeypatch.setattr(localization_trainer, "infer_and_process_predictions_e2e",
                        lambda *args, **kwargs: measured.append(trainer.model.trained) or 0.)
    trainer.train([], [], ["swing"])
    assert measured == [1, 3] and trainer.epochs_completed == 3
    assert trainer.best_epoch == 0 and trainer.best_checkpoint_path
    assert trainer.stop_reason == "validation_plateau"


def test_e2e_rejects_nonfinite_validation_loss(e2e_trainer):
    trainer = e2e_trainer([float("nan")])
    with pytest.raises(ValueError, match="not finite at epoch 1"):
        trainer.train([], [], ["swing"])


@pytest.mark.parametrize("settings", [{"patience": -1}, {"min_delta": -.1}, {"min_delta": float("nan")}])
def test_e2e_rejects_invalid_stopping_settings(e2e_trainer, settings):
    with pytest.raises(ValueError, match="Early-stopping"):
        e2e_trainer([1.], **settings)
