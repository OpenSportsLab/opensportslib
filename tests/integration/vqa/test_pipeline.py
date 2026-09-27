from pathlib import Path

from opensportslib.apis.vqa import VQAModel


def _install_vqa_pipeline_stub(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "vqa-adapter"

    def train(self, train_set=None, valid_set=None, **kwargs):
        del kwargs
        assert Path(self._resolve_split_path("train", train_set)).exists()
        assert Path(self._resolve_split_path("valid", valid_set)).exists()
        checkpoint_path.mkdir()
        (checkpoint_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.best_checkpoint = self.last_loaded_weights = str(checkpoint_path)
        return str(checkpoint_path)

    def infer(self, test_set=None, weights=None, **kwargs):
        del kwargs
        assert Path(self._resolve_split_path("test", test_set)).exists()
        self.last_loaded_weights = weights or self.last_loaded_weights
        return {
            "version": "2.0", "task": "vqa", "metadata": {"type": "predictions"},
            "data": [{"id": "action_0", "question": "What happened?", "answer_text": "Challenge"}],
        }

    def evaluate(self, test_set=None, weights=None, predictions=None, **kwargs):
        predictions = predictions or self.infer(test_set=test_set, weights=weights, **kwargs)
        return {"token_f1": 1.0, "count": len(predictions["data"])}

    monkeypatch.setattr(VQAModel, "train", train)
    monkeypatch.setattr(VQAModel, "infer", infer)
    monkeypatch.setattr(VQAModel, "evaluate", evaluate)


def test_vqa_synthetic_train_checkpoint_infer_evaluate(vqa_config_path, monkeypatch, tmp_path):
    _install_vqa_pipeline_stub(monkeypatch, tmp_path)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    api = VQAModel(config=vqa_config_path)
    checkpoint = api.train(use_wandb=False)
    assert checkpoint and Path(checkpoint).is_dir()
    predictions = api.infer(weights=checkpoint, use_wandb=False)
    assert predictions["task"] == "vqa"
    assert predictions["data"][0]["answer_text"]
    metrics = api.evaluate(predictions=predictions, use_wandb=False)
    assert metrics == {"token_f1": 1.0, "count": 1}
