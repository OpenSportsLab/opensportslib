import os
import sys
from types import ModuleType, SimpleNamespace

from opensportslib.apis.localization import LocalizationModel
from opensportslib.core.config.accessors import get_loader_backend


def _split(split_type, batch_size=4, shuffle=False):
    return SimpleNamespace(
        type=split_type,
        annotation_path="/tmp/annotations.json",
        source_path="/tmp/data",
        dataloader=SimpleNamespace(batch_size=batch_size, shuffle=shuffle),
    )


def _make_config(loader_backend="dali"):
    return SimpleNamespace(
        TASK="localization",
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                runtime=SimpleNamespace(loader_backend=loader_backend),
                classes=["PASS", "SHOT"],
                splits=SimpleNamespace(
                    train=_split("VideoGameWithDali", shuffle=True),
                    valid=_split("VideoGameWithDali", shuffle=True),
                    test=_split("VideoGameWithDaliVideo"),
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
        MODEL=SimpleNamespace(task="localization", components=SimpleNamespace(), topology=[]),
        SYSTEM=SimpleNamespace(
            device="cpu",
            gpu=SimpleNamespace(count=0, id=0),
            reproducibility=SimpleNamespace(seed=0, use_seed=False),
            paths=SimpleNamespace(save_dir="/tmp", work_dir="/tmp", log_dir="/tmp"),
        ),
        TRAIN=SimpleNamespace(
            execution=SimpleNamespace(multi_gpu=False),
            trainer=SimpleNamespace(type="trainer_e2e"),
        ),
    )


def _make_api(config):
    api = LocalizationModel.__new__(LocalizationModel)
    api.config = config
    api.config_path = "/tmp/localization.yaml"
    api.model = None
    api.processor = None
    api.trainer = None
    api.last_loaded_weights = None
    api.best_checkpoint = None
    api.train_flag = False
    return api


def test_hf_localization_cpu_switches_to_opencv_and_normalizes_dataloader(monkeypatch):
    api = _make_api(_make_config(loader_backend="dali"))
    api.config = _make_config(loader_backend="dali")

    monkeypatch.setattr(
        "opensportslib.core.utils.config.select_device",
        lambda system: SimpleNamespace(type="cpu"),
    )

    api._adapt_hf_backend_for_device("OpenSportsLab/OSL-loc-snbas-2025-e2e")

    assert get_loader_backend(api.config) == "opencv"
    assert api.config.DATA.common.splits.train.type == "VideoGameWithOpencv"
    assert api.config.DATA.common.splits.valid.type == "VideoGameWithOpencv"
    assert api.config.DATA.common.splits.test.type == "VideoGameWithOpencvVideo"
    assert api.config.DATA.common.splits.valid.dataloader.num_workers == 0
    assert api.config.DATA.common.splits.valid.dataloader.pin_memory is False


def test_hf_localization_auto_cuda_switches_to_dali(monkeypatch):
    api = _make_api(_make_config(loader_backend="opencv"))
    api.config.DATA.common.splits.train.type = "VideoGameWithOpencv"
    api.config.DATA.common.splits.valid.type = "VideoGameWithOpencv"
    api.config.DATA.common.splits.test.type = "VideoGameWithOpencvVideo"
    api.config.SYSTEM.device = "auto"

    monkeypatch.setattr(
        "opensportslib.core.utils.config.select_device",
        lambda system: SimpleNamespace(type="cuda"),
    )
    monkeypatch.setattr("opensportslib.apis.localization._dali_available", lambda: True)

    api._adapt_hf_backend_for_device("OpenSportsLab/OSL-loc-snbas-2023-e2e")

    assert get_loader_backend(api.config) == "dali"
    assert api.config.DATA.common.splits.train.type == "VideoGameWithDali"
    assert api.config.DATA.common.splits.valid.type == "VideoGameWithDali"
    assert api.config.DATA.common.splits.test.type == "VideoGameWithDaliVideo"


def test_hf_localization_cuda_without_dali_falls_back_to_opencv(monkeypatch):
    api = _make_api(_make_config(loader_backend="dali"))

    monkeypatch.setattr(
        "opensportslib.core.utils.config.select_device",
        lambda system: SimpleNamespace(type="cuda"),
    )
    monkeypatch.setattr("opensportslib.apis.localization._dali_available", lambda: False)

    api._adapt_hf_backend_for_device("OpenSportsLab/OSL-loc-snbas-2023-e2e")

    assert get_loader_backend(api.config) == "opencv"
    assert api.config.DATA.common.splits.train.type == "VideoGameWithOpencv"
    assert api.config.DATA.common.splits.valid.type == "VideoGameWithOpencv"
    assert api.config.DATA.common.splits.test.type == "VideoGameWithOpencvVideo"


def test_local_weights_do_not_trigger_hf_backend_override(monkeypatch):
    api = _make_api(_make_config(loader_backend="dali"))

    monkeypatch.setattr(
        "opensportslib.core.utils.config.select_device",
        lambda system: SimpleNamespace(type="cpu"),
    )

    api._adapt_hf_backend_for_device("/tmp/model.pt")

    assert get_loader_backend(api.config) == "dali"
    assert api.config.DATA.common.splits.train.type == "VideoGameWithDali"
    assert not hasattr(api.config.DATA.common.splits.valid.dataloader, "num_workers")


def test_localization_infer_applies_hf_backend_override_before_dataset_build(
    tmp_path,
    monkeypatch,
):
    test_set = tmp_path / "test.json"
    test_set.write_text("{}", encoding="utf-8")
    observed = {}

    class FakeData:
        cfg = SimpleNamespace(dataloader=SimpleNamespace(batch_size=1, shuffle=False))
        default_args = {}

        def building_dataset(self, cfg, gpu, default_args):
            del cfg, gpu, default_args
            return ["dataset"]

        def building_dataloader(self, dataset, cfg, gpu, dali):
            del dataset, cfg, gpu
            observed["dali"] = dali
            return ["batch"]

    class FakeInferer:
        def infer(self, cfg, data, dataloader):
            del data, dataloader
            observed["backend"] = get_loader_backend(cfg)
            observed["dataset_type"] = cfg.DATA.common.splits.test.type
            return {"task": "localization"}

    fake_load_annotations = ModuleType("opensportslib.core.utils.load_annotations")
    fake_load_annotations.check_config = lambda config, split: None
    fake_load_annotations.whether_infer_split = lambda test_cfg: False
    monkeypatch.setitem(sys.modules, "opensportslib.core.utils.load_annotations", fake_load_annotations)

    fake_localization_trainer = ModuleType("opensportslib.core.trainer.localization_trainer")
    fake_localization_trainer.build_inferer = lambda cfg, model: FakeInferer()
    monkeypatch.setitem(
        sys.modules,
        "opensportslib.core.trainer.localization_trainer",
        fake_localization_trainer,
    )

    fake_builder = ModuleType("opensportslib.datasets.builder")
    fake_builder.build_dataset = lambda config, split: FakeData()
    monkeypatch.setitem(sys.modules, "opensportslib.datasets.builder", fake_builder)

    fake_wandb = ModuleType("opensportslib.core.utils.wandb")
    fake_wandb.init_wandb = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "opensportslib.core.utils.wandb", fake_wandb)

    monkeypatch.setattr("opensportslib.core.utils.config.resolve_config_omega", lambda config, weights=None: config)
    monkeypatch.setattr(
        "opensportslib.core.utils.config.select_device",
        lambda system: SimpleNamespace(type="cpu"),
    )
    monkeypatch.setattr(LocalizationModel, "load_weights", lambda self, weights=None, **kwargs: setattr(self, "model", object()))

    api = _make_api(_make_config(loader_backend="dali"))
    os.environ["RUN_ID"] = "test-run"
    api.last_loaded_weights = "OpenSportsLab/OSL-loc-snbas-2025-e2e"

    predictions = api.infer(test_set=str(test_set), use_wandb=False)

    assert predictions == {"task": "localization"}
    assert observed["backend"] == "opencv"
    assert observed["dataset_type"] == "VideoGameWithOpencvVideo"
    assert observed["dali"] is False
