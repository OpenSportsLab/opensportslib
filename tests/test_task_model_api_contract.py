import inspect
from types import SimpleNamespace

from opensportslib.apis import ClassificationModel, LocalizationModel


def test_method_signatures_expose_weights_and_no_pretrained_in_signature(
    classification_config_path,
    localization_config_path,
):
    cls_api = ClassificationModel(config=classification_config_path)
    loc_api = LocalizationModel(config=localization_config_path)

    for api in (cls_api, loc_api):
        for method_name in ("load_weights", "train", "infer", "evaluate"):
            sig = inspect.signature(getattr(api, method_name))
            assert "weights" in sig.parameters
            assert "pretrained" not in sig.parameters
            assert any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )

    load_sig = inspect.signature(cls_api.load_weights)
    assert "optimizer" not in load_sig.parameters
    assert "scheduler" not in load_sig.parameters

    infer_sig = inspect.signature(cls_api.infer)
    assert "output_path" not in infer_sig.parameters

    eval_sig = inspect.signature(cls_api.evaluate)
    assert "predictions" in eval_sig.parameters

    loc_train_sig = inspect.signature(loc_api.train)
    loc_infer_sig = inspect.signature(loc_api.infer)
    loc_eval_sig = inspect.signature(loc_api.evaluate)
    assert "use_ddp" not in loc_train_sig.parameters
    assert "use_ddp" not in loc_infer_sig.parameters
    assert "use_ddp" not in loc_eval_sig.parameters

    save_sig = inspect.signature(cls_api.save_predictions)
    assert "output_path" in save_sig.parameters
    assert "predictions" in save_sig.parameters
    assert save_sig.parameters["predictions"].default is inspect._empty


def test_save_predictions_writes_dict_payload(classification_config_path, tmp_path):
    api = ClassificationModel(config=classification_config_path)
    out_path = tmp_path / "predictions.json"

    saved = api.save_predictions(
        output_path=str(out_path),
        predictions={"items": [{"label": "PASS", "confidence": 0.9}]},
    )

    assert saved == str(out_path)
    assert out_path.exists()


def test_constructor_is_minimal_and_sets_run_id(
    classification_config_path,
    localization_config_path,
):
    cls_sig = inspect.signature(ClassificationModel)
    loc_sig = inspect.signature(LocalizationModel)

    assert list(cls_sig.parameters.keys()) == ["config", "weights"]
    assert list(loc_sig.parameters.keys()) == ["config", "weights"]

    cls_api = ClassificationModel(config=classification_config_path)
    loc_api = LocalizationModel(config=localization_config_path)

    assert cls_api.run_id
    assert loc_api.run_id


def test_classification_constructor_weights_are_default_for_train_and_infer(
    classification_config_path,
    tmp_path,
    monkeypatch,
):
    calls = []
    test_set = tmp_path / "test.json"
    test_set.write_text("{}", encoding="utf-8")

    def fake_load_weights(self, weights=None, **kwargs):
        del kwargs
        self.model = object()
        self.last_loaded_weights = weights
        self.best_checkpoint = weights

    def fake_worker_ddp(
        rank,
        world_size,
        mode,
        config_path,
        config,
        return_queue=None,
        train_set=None,
        valid_set=None,
        test_set=None,
        weights=None,
        use_wandb=False,
    ):
        del rank, world_size, config_path, config, train_set, valid_set
        del test_set, use_wandb
        calls.append((mode, weights))
        if mode == "train":
            return_queue.put("trained-checkpoint.pt")
        else:
            return_queue.put({"task": "action_classification"})

    monkeypatch.setattr(ClassificationModel, "load_weights", fake_load_weights)
    monkeypatch.setattr(ClassificationModel, "_worker_ddp", staticmethod(fake_worker_ddp))
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)

    api = ClassificationModel(
        config=classification_config_path,
        weights="OpenSportsLab/OSL-cls-action-mvitv2",
    )

    predictions = api.infer(test_set=str(test_set), use_wandb=False)
    assert predictions == {"task": "action_classification"}
    assert calls[-1] == ("infer", "OpenSportsLab/OSL-cls-action-mvitv2")

    api.infer(test_set=str(test_set), weights="override", use_wandb=False)
    assert calls[-1] == ("infer", "override")

    api.train(use_wandb=False)
    assert calls[-1] == ("train", "OpenSportsLab/OSL-cls-action-mvitv2")

    api.train(weights="override", use_wandb=False)
    assert calls[-1] == ("train", "override")


def test_classification_evaluate_uses_provided_predictions(
    classification_config_path,
    tmp_path,
    monkeypatch,
):
    test_set = tmp_path / "test.json"
    test_set.write_text("{}", encoding="utf-8")
    provided_predictions = {"task": "action_classification", "data": []}
    evaluated = {}

    class FakeDataset:
        label_map = {0: "PASS"}
        exclude_labels = []

    class FakeTrainer:
        def evaluate(self, pred_path, gt_path, class_names, exclude_labels):
            evaluated["pred_path"] = pred_path
            evaluated["gt_path"] = gt_path
            evaluated["class_names"] = class_names
            evaluated["exclude_labels"] = exclude_labels
            return {"f1": 1.0}

    def fail_infer(*args, **kwargs):
        del args, kwargs
        raise AssertionError("infer should not run when predictions are provided")

    monkeypatch.setattr(ClassificationModel, "infer", fail_infer)
    monkeypatch.setattr(
        "opensportslib.core.trainer.classification_trainer.Trainer_Classification",
        lambda config: FakeTrainer(),
    )
    monkeypatch.setattr(
        "opensportslib.datasets.builder.build_dataset",
        lambda config, path, processor, split: FakeDataset(),
    )

    api = ClassificationModel(config=classification_config_path)
    metrics = api.evaluate(
        test_set=str(test_set),
        predictions=provided_predictions,
        use_wandb=False,
    )

    assert metrics == {"f1": 1.0}
    assert evaluated["pred_path"] is provided_predictions
    assert evaluated["gt_path"] == str(test_set)


def test_localization_evaluate_uses_provided_predictions(
    localization_config_path,
    tmp_path,
    monkeypatch,
):
    test_set = tmp_path / "test.json"
    test_set.write_text("{}", encoding="utf-8")
    provided_predictions = str(tmp_path / "predictions.json")
    evaluated = {}

    class FakeEvaluator:
        def evaluate(self, cfg_testset, json_gz_file=None):
            evaluated["cfg_testset"] = cfg_testset
            evaluated["json_gz_file"] = json_gz_file
            return {"a_mAP": 1.0}

    def fail_infer(*args, **kwargs):
        del args, kwargs
        raise AssertionError("infer should not run when predictions are provided")

    monkeypatch.setattr(LocalizationModel, "infer", fail_infer)
    monkeypatch.setattr(
        "opensportslib.core.trainer.localization_trainer.build_evaluator",
        lambda cfg: FakeEvaluator(),
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.config.resolve_config_omega",
        lambda config: config,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.load_annotations.check_config",
        lambda config, split: None,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.load_annotations.whether_infer_split",
        lambda test_cfg: False,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.load_annotations.has_localization_events",
        lambda path: True,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda *args, **kwargs: None,
    )

    api = LocalizationModel(config=localization_config_path)
    api.config = SimpleNamespace(
        DATA=SimpleNamespace(
            test=SimpleNamespace(
                path=str(test_set),
                results="default_predictions.json",
            )
        ),
        MODEL=SimpleNamespace(multi_gpu=True),
    )

    metrics = api.evaluate(
        test_set=str(test_set),
        predictions=provided_predictions,
        use_wandb=False,
    )

    assert metrics == {"a_mAP": 1.0}
    assert evaluated["json_gz_file"] == provided_predictions


def test_localization_constructor_weights_are_default_for_train_and_infer(
    localization_config_path,
    tmp_path,
    monkeypatch,
):
    load_calls = []
    trainer_resume_from = []

    def make_config():
        def split(name):
            path = tmp_path / f"{name}.json"
            path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(path=str(path), dataloader=SimpleNamespace())

        return SimpleNamespace(
            DATA=SimpleNamespace(
                train=split("train"),
                valid=split("valid"),
                test=split("test"),
                classes=["PASS", "SHOT"],
            ),
            MODEL=SimpleNamespace(multi_gpu=True),
            SYSTEM=SimpleNamespace(seed=42, GPU=1),
            TRAIN=SimpleNamespace(type="trainer"),
            dali=False,
        )

    class FakeData:
        cfg = SimpleNamespace(dataloader=SimpleNamespace())
        default_args = {}

        def building_dataset(self, cfg, gpu, default_args):
            del cfg, gpu, default_args
            return ["dataset"]

        def building_dataloader(self, dataset, cfg, gpu, dali):
            del dataset, cfg, gpu, dali
            return ["batch"]

    class FakeTrainer:
        best_checkpoint_path = "trained-localization.ckpt"

        def train(self, **kwargs):
            del kwargs

    class FakeInferer:
        def infer(self, cfg, data, dataloader):
            del cfg, data, dataloader
            return {"task": "localization"}

    def fake_load_weights(self, weights=None, **kwargs):
        del kwargs
        load_calls.append(weights)
        self.model = object()
        self.last_loaded_weights = weights
        self.best_checkpoint = weights

    def fake_build_trainer(cfg, model, default_args, resume_from=None):
        del cfg, model, default_args
        trainer_resume_from.append(resume_from)
        return FakeTrainer()

    monkeypatch.setattr(LocalizationModel, "load_weights", fake_load_weights)
    monkeypatch.setattr(
        "opensportslib.models.builder.build_model",
        lambda config, device: object(),
    )
    monkeypatch.setattr(
        "opensportslib.datasets.builder.build_dataset",
        lambda config, split: FakeData(),
    )
    monkeypatch.setattr(
        "opensportslib.core.trainer.localization_trainer.build_trainer",
        fake_build_trainer,
    )
    monkeypatch.setattr(
        "opensportslib.core.trainer.localization_trainer.build_inferer",
        lambda cfg, model: FakeInferer(),
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.config.resolve_config_omega",
        lambda config: config,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.config.select_device",
        lambda system: "cpu",
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.load_annotations.check_config",
        lambda config, split: None,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.load_annotations.whether_infer_split",
        lambda test_cfg: False,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.default_args.get_default_args_trainer",
        lambda config, loader_len: {},
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.default_args.get_default_args_train",
        lambda model, train_loader, valid_loader, classes, trainer_type: {},
    )

    infer_api = LocalizationModel(config=localization_config_path, weights="default")
    infer_api.config = make_config()
    assert infer_api.infer(use_wandb=False) == {"task": "localization"}
    assert load_calls == ["default"]

    infer_api.infer(weights="override", use_wandb=False)
    assert load_calls[-1] == "override"

    train_api = LocalizationModel(config=localization_config_path, weights="default")
    train_api.config = make_config()
    train_api.train(use_wandb=False)
    assert trainer_resume_from[-1] == "default"

    train_api = LocalizationModel(config=localization_config_path, weights="default")
    train_api.config = make_config()
    train_api.train(weights="override", use_wandb=False)
    assert trainer_resume_from[-1] == "override"
