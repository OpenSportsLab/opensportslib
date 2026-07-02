import inspect
from types import SimpleNamespace

from opensportslib.apis import ClassificationModel, LocalizationModel, VQAModel


def test_method_signatures_expose_weights_and_no_pretrained_in_signature(
    classification_config_path,
    localization_config_path,
    vqa_config_path,
):
    cls_api = ClassificationModel(config=classification_config_path)
    loc_api = LocalizationModel(config=localization_config_path)
    vqa_api = VQAModel(config=vqa_config_path)

    for api in (cls_api, loc_api, vqa_api):
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
    vqa_config_path,
):
    cls_sig = inspect.signature(ClassificationModel)
    loc_sig = inspect.signature(LocalizationModel)
    vqa_sig = inspect.signature(VQAModel)

    assert list(cls_sig.parameters.keys()) == ["config", "weights"]
    assert list(loc_sig.parameters.keys()) == ["config", "weights"]
    assert list(vqa_sig.parameters.keys()) == ["config", "weights"]

    cls_api = ClassificationModel(config=classification_config_path)
    loc_api = LocalizationModel(config=localization_config_path)
    vqa_api = VQAModel(config=vqa_config_path)

    assert cls_api.run_id
    assert loc_api.run_id
    assert vqa_api.run_id


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
        lambda config, weights=None: config,
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
            return SimpleNamespace(
                annotation_path=str(path),
                source_path=str(tmp_path),
                dataloader=SimpleNamespace(),
            )

        return SimpleNamespace(
            DATA=SimpleNamespace(
                common=SimpleNamespace(
                    classes=["PASS", "SHOT"],
                    runtime=SimpleNamespace(loader_backend="opencv"),
                    splits=SimpleNamespace(
                        train=split("train"),
                        valid=split("valid"),
                        test=split("test"),
                    ),
                ),
                inputs=SimpleNamespace(
                    video=SimpleNamespace(
                        modality="video",
                        representation="raw",
                        sampling=SimpleNamespace(),
                        transform=SimpleNamespace(),
                        params=SimpleNamespace(),
                    )
                ),
            ),
            MODEL=SimpleNamespace(
                task="localization",
                components=SimpleNamespace(),
                topology=[],
            ),
            SYSTEM=SimpleNamespace(
                reproducibility=SimpleNamespace(seed=42, use_seed=False),
                gpu=SimpleNamespace(count=1, id=0),
                device="cpu",
                paths=SimpleNamespace(save_dir=str(tmp_path)),
            ),
            TRAIN=SimpleNamespace(trainer=SimpleNamespace(type="trainer_e2e")),
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
        self._resume_state = {"source_weights": weights}

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
        lambda config, weights=None: config,
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
    assert trainer_resume_from[-1]["source_weights"] == "default"

    train_api = LocalizationModel(config=localization_config_path, weights="default")
    train_api.config = make_config()
    train_api.train(weights="override", use_wandb=False)
    assert trainer_resume_from[-1]["source_weights"] == "override"


def test_vqa_api_uses_wandb_for_train_infer_and_evaluate(vqa_config_path, tmp_path, monkeypatch):
    wandb_inits = []
    train_calls = []
    infer_calls = []
    evaluate_calls = []

    def split(name):
        path = tmp_path / f"{name}.json"
        path.write_text("[]", encoding="utf-8")
        return SimpleNamespace(
            annotation_path=str(path),
            source_path=str(tmp_path),
            dataloader=SimpleNamespace(batch_size=1),
        )

    config = SimpleNamespace(
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                splits=SimpleNamespace(
                    train=split("train"),
                    valid=split("valid"),
                    test=split("test"),
                )
            )
        ),
        MODEL=SimpleNamespace(load=SimpleNamespace(checkpoint_path=None)),
        SYSTEM=SimpleNamespace(device="cpu", gpu=SimpleNamespace(count=0, id=0)),
        TRAIN=SimpleNamespace(execution={"training_backend": "baseline", "prompt": {}, "generation": {}}),
        TASK="VQA",
    )

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def train(self, model, train_data, valid_data=None, *, rank=0, world_size=1, use_wandb=False):
            del model, train_data, valid_data, rank, world_size
            train_calls.append(use_wandb)
            return "trained.ckpt"

        def infer(self, model, dataset, *, use_wandb=False):
            del model, dataset
            infer_calls.append(use_wandb)
            return {"task": "vqa", "data": [{"id": "1", "question": "Q", "answer_text": "A"}]}

        def evaluate(self, predictions, dataset, *, use_wandb=False):
            del predictions, dataset
            evaluate_calls.append(use_wandb)
            return {"exact_match": 1.0}

        def load(self, weights):
            return weights

    monkeypatch.setattr(
        "opensportslib.apis.vqa.resolve_config_omega",
        lambda cfg, weights=None: cfg,
    )
    monkeypatch.setattr(
        "opensportslib.datasets.builder.build_dataset",
        lambda *args, **kwargs: [{"id": kwargs.get("split", "x")}],
    )
    monkeypatch.setattr(
        "opensportslib.models.builder.build_model",
        lambda cfg, device: (object(), None),
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.config.select_device",
        lambda system: "cpu",
    )
    monkeypatch.setattr(
        "opensportslib.core.trainer.vqa_trainer.Trainer_VQA",
        FakeTrainer,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda cfg_path, cfg, run_id, use_wandb=False: wandb_inits.append(use_wandb),
    )

    api = VQAModel(config=vqa_config_path)
    api.config = config

    assert api.train(use_wandb=True) == "trained.ckpt"
    predictions = api.infer(use_wandb=True)
    assert predictions["task"] == "vqa"
    metrics = api.evaluate(predictions=predictions, use_wandb=True)
    assert metrics == {"exact_match": 1.0}

    assert wandb_inits == [True, True, True]
    assert train_calls == [True]
    assert infer_calls == [True]
    assert evaluate_calls == [True]


def test_vqa_worker_ddp_initializes_wandb_on_rank_zero(vqa_config_path, tmp_path, monkeypatch):
    wandb_inits = []

    train_path = tmp_path / "train.json"
    valid_path = tmp_path / "valid.json"
    train_path.write_text("[]", encoding="utf-8")
    valid_path.write_text("[]", encoding="utf-8")

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def train(self, model, train_data, valid_data=None, *, rank=0, world_size=1, use_wandb=False):
            del model, train_data, valid_data, rank, world_size, use_wandb
            return "trained.ckpt"

    class FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    monkeypatch.setattr("torch.cuda.set_device", lambda rank: None)
    monkeypatch.setattr("opensportslib.core.utils.ddp.ddp_setup", lambda rank, world_size: None)
    monkeypatch.setattr("opensportslib.core.utils.ddp.ddp_cleanup", lambda: None)
    monkeypatch.setattr(
        "opensportslib.datasets.builder.build_dataset",
        lambda *args, **kwargs: [{"id": kwargs.get("split", "x")}],
    )
    monkeypatch.setattr(
        "opensportslib.core.trainer.vqa_trainer.Trainer_VQA",
        FakeTrainer,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda cfg_path, cfg, run_id, use_wandb=False: wandb_inits.append((cfg_path, use_wandb)),
    )

    os_environ = __import__("os").environ
    os_environ["RUN_ID"] = "testrun"
    queue = FakeQueue()
    config = SimpleNamespace()

    VQAModel._worker_ddp(
        rank=0,
        world_size=1,
        config_path=vqa_config_path,
        config=config,
        return_queue=queue,
        train_set=str(train_path),
        valid_set=str(valid_path),
        use_wandb=True,
    )

    assert wandb_inits == [(vqa_config_path, True)]


def test_vqa_worker_ddp_sets_distributed_debug_when_unset(vqa_config_path, tmp_path, monkeypatch):
    train_path = tmp_path / "train.json"
    valid_path = tmp_path / "valid.json"
    train_path.write_text("[]", encoding="utf-8")
    valid_path.write_text("[]", encoding="utf-8")

    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def train(self, model, train_data, valid_data=None, *, rank=0, world_size=1, use_wandb=False):
            del model, train_data, valid_data, rank, world_size, use_wandb
            return "trained.ckpt"

    class FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    monkeypatch.setattr("torch.cuda.set_device", lambda rank: None)
    monkeypatch.setattr("opensportslib.core.utils.ddp.ddp_setup", lambda rank, world_size: None)
    monkeypatch.setattr("opensportslib.core.utils.ddp.ddp_cleanup", lambda: None)
    monkeypatch.setattr(
        "opensportslib.datasets.builder.build_dataset",
        lambda *args, **kwargs: [{"id": kwargs.get("split", "x")}],
    )
    monkeypatch.setattr(
        "opensportslib.core.trainer.vqa_trainer.Trainer_VQA",
        FakeTrainer,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda cfg_path, cfg, run_id, use_wandb=False: None,
    )

    os_environ = __import__("os").environ
    prior_debug = os_environ.pop("TORCH_DISTRIBUTED_DEBUG", None)
    os_environ["RUN_ID"] = "testrun"
    queue = FakeQueue()
    config = SimpleNamespace()

    try:
        VQAModel._worker_ddp(
            rank=0,
            world_size=2,
            config_path=vqa_config_path,
            config=config,
            return_queue=queue,
            train_set=str(train_path),
            valid_set=str(valid_path),
            use_wandb=False,
        )
        assert os_environ["TORCH_DISTRIBUTED_DEBUG"] == "INFO"
    finally:
        if prior_debug is None:
            os_environ.pop("TORCH_DISTRIBUTED_DEBUG", None)
        else:
            os_environ["TORCH_DISTRIBUTED_DEBUG"] = prior_debug


def test_vqa_direct_xvars_infer_uses_native_model_path(vqa_config_path, tmp_path, monkeypatch):
    from opensportslib.apis.vqa import VQAModel

    video_path = tmp_path / "clip_0.mp4"
    video_path.write_bytes(b"video")

    config = SimpleNamespace(
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                splits=SimpleNamespace(
                    test=SimpleNamespace(annotation_path=str(tmp_path / "test.json")),
                )
            )
        ),
        MODEL=SimpleNamespace(load=SimpleNamespace(checkpoint_path=None)),
        SYSTEM=SimpleNamespace(device="cuda", gpu=SimpleNamespace(count=1, id=0)),
        TRAIN=SimpleNamespace(execution={"training_backend": "xvars_videochatgpt_lora", "prompt": {}, "generation": {}}),
        TASK="VQA",
    )

    fake_model = object()
    captured = {}

    def fake_trainer_infer(model, dataset, use_wandb=False):
        captured["infer"] = (model, list(dataset), use_wandb)
        return {
            "task": "vqa",
            "data": [
                {
                    "id": "clip_0",
                    "question": "What happened?",
                    "answer_text": "native-answer",
                    "video_path": str(video_path),
                }
            ],
        }

    monkeypatch.setattr(
        "opensportslib.apis.vqa.resolve_config_omega",
        lambda cfg, weights=None: cfg,
    )
    monkeypatch.setattr(
        "opensportslib.apis.vqa.get_vqa_backend",
        lambda cfg: "xvars_videochatgpt",
    )
    monkeypatch.setattr(
        "opensportslib.models.base.xvars_videochatgpt.run_upstream_xvars_demo_direct_infer",
        lambda cfg, *, video_path, question: (_ for _ in ()).throw(
            AssertionError("upstream demo helper should not run from public infer()")
        ),
    )
    monkeypatch.setattr(
        "opensportslib.core.trainer.vqa_trainer.Trainer_VQA",
        lambda cfg: SimpleNamespace(
            load=lambda weights: weights,
            infer=fake_trainer_infer,
        ),
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda cfg_path, cfg, run_id, use_wandb=False: None,
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.config.select_device",
        lambda cfg: "cpu",
    )
    monkeypatch.setattr(
        "opensportslib.models.builder.build_model",
        lambda *args, **kwargs: (fake_model, None),
    )

    api = VQAModel(config=vqa_config_path)
    api.config = config
    out = api.infer(video_path=str(video_path), question="What happened?", use_wandb=False)

    infer_model, infer_dataset, infer_use_wandb = captured["infer"]
    assert infer_model is fake_model
    assert infer_use_wandb is False
    assert infer_dataset == [
        {
            "id": "clip_0",
            "question": "What happened?",
            "references": [],
            "video_path": str(video_path),
            "video_spatio_temporal_features": None,
            "prior_prediction_text": "",
            "labels": {},
            "metadata": {},
            "_xvars_demo_parity_direct_infer": True,
        }
    ]
    assert out["task"] == "vqa"
    assert out["data"][0]["question"] == "What happened?"
    assert out["data"][0]["answer_text"] == "native-answer"


def test_vqa_xvars_test_set_infer_uses_native_model_and_not_upstream_repo(vqa_config_path, tmp_path, monkeypatch):
    from opensportslib.apis.vqa import VQAModel

    test_path = tmp_path / "test.json"
    test_path.write_text("{}", encoding="utf-8")

    config = SimpleNamespace(
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                splits=SimpleNamespace(
                    test=SimpleNamespace(annotation_path=str(test_path)),
                )
            )
        ),
        MODEL=SimpleNamespace(load=SimpleNamespace(checkpoint_path=None)),
        SYSTEM=SimpleNamespace(device="cpu", gpu=SimpleNamespace(count=0, id=0)),
        TRAIN=SimpleNamespace(execution={"training_backend": "xvars_videochatgpt_lora", "prompt": {}, "generation": {}}),
        TASK="VQA",
    )

    dataset_rows = [{"id": "action_0", "question": "What happened?", "video_path": str(tmp_path / "clip_0.mp4")}]
    fake_model = object()
    captured = {}

    def fake_trainer_infer(model, dataset, use_wandb=False):
        captured["infer"] = (model, list(dataset), use_wandb)
        return {
            "task": "vqa",
            "data": [
                {
                    "id": "action_0",
                    "question": "What happened?",
                    "answer_text": "dataset-native-answer",
                    "video_path": dataset_rows[0]["video_path"],
                }
            ],
        }

    monkeypatch.setattr("opensportslib.apis.vqa.resolve_config_omega", lambda cfg, weights=None: cfg)
    monkeypatch.setattr("opensportslib.apis.vqa.get_vqa_backend", lambda cfg: "xvars_videochatgpt")
    monkeypatch.setattr("opensportslib.core.utils.config.select_device", lambda cfg: "cpu")
    monkeypatch.setattr(
        "opensportslib.models.base.xvars_videochatgpt._UPSTREAM_XVARS_REPO_ROOT",
        str(tmp_path / "missing-xvars-repo"),
    )
    monkeypatch.setattr(
        "opensportslib.models.base.xvars_videochatgpt.run_upstream_xvars_demo_direct_infer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("upstream helper should not run for test_set inference")
        ),
    )
    monkeypatch.setattr(
        "opensportslib.core.trainer.vqa_trainer.Trainer_VQA",
        lambda cfg: SimpleNamespace(
            load=lambda weights: weights,
            infer=fake_trainer_infer,
        ),
    )
    monkeypatch.setattr(
        "opensportslib.datasets.builder.build_dataset",
        lambda cfg, path, weights, split="test": dataset_rows,
    )
    monkeypatch.setattr(
        "opensportslib.models.builder.build_model",
        lambda *args, **kwargs: (fake_model, None),
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda cfg_path, cfg, run_id, use_wandb=False: None,
    )

    api = VQAModel(config=vqa_config_path)
    api.config = config
    out = api.infer(test_set=str(test_path), use_wandb=False)

    infer_model, infer_dataset, infer_use_wandb = captured["infer"]
    assert infer_model is fake_model
    assert infer_dataset == dataset_rows
    assert infer_use_wandb is False
    assert out["task"] == "vqa"
    assert out["data"][0]["answer_text"] == "dataset-native-answer"
