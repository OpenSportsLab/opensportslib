from pathlib import Path
from types import SimpleNamespace
import inspect
import json
import sys
import types

import pytest

from opensportslib.apis import VQAModel


class _FakeRuntime:
    def __init__(self):
        self.model = object()
        self.image_processor = object()
        self.calls = []

    def infer_one(self, *, video_path, question, conv_mode, temperature, max_new_tokens):
        self.calls.append(
            {
                "video_path": video_path,
                "question": question,
                "conv_mode": conv_mode,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
        )
        return SimpleNamespace(
            answer_text=f"answer::{question}",
            action_class="Tackling",
            offence_class="Offence",
            severity_class="3.0",
        )


def test_vqa_method_signatures_expose_weights(vqa_config_path):
    api = VQAModel(config=vqa_config_path)

    for method_name in ("load_weights", "train", "infer", "evaluate"):
        sig = inspect.signature(getattr(api, method_name))
        assert "weights" in sig.parameters
        assert any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

    infer_sig = inspect.signature(api.infer)
    assert "test_set" in infer_sig.parameters


def test_vqa_constructor_weights_are_default_for_infer(vqa_config_path, monkeypatch):
    calls = []
    captured = {}
    fake_model = object()
    api = VQAModel(config=vqa_config_path, weights="default-adapter")
    api.config = SimpleNamespace(
        SYSTEM=SimpleNamespace(device="cpu", gpu=SimpleNamespace(count=0, id=0)),
        MODEL=SimpleNamespace(load=SimpleNamespace(checkpoint_path=None)),
        TRAIN=SimpleNamespace(execution={"training_backend": "xvars_videochatgpt_lora", "prompt": {}, "generation": {}, "xvars": {"feature_source": "raw_video"}}),
        TASK="VQA",
    )

    monkeypatch.setattr("opensportslib.apis.vqa.resolve_config_omega", lambda cfg, weights=None: cfg)
    monkeypatch.setattr("opensportslib.apis.vqa.get_vqa_backend", lambda cfg: "xvars_videochatgpt")
    monkeypatch.setattr("opensportslib.core.utils.config.select_device", lambda cfg: "cpu")
    monkeypatch.setattr(
        "opensportslib.models.builder.build_model",
        lambda *args, **kwargs: (fake_model, None),
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda cfg_path, cfg, run_id, use_wandb=False: None,
    )

    def _fake_trainer_factory(cfg):
        return SimpleNamespace(
            load=lambda weights: calls.append(weights),
            infer=lambda model, dataset, use_wandb=False: {
                "task": "vqa",
                "data": [
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "answer_text": f"answer::{row['question']}",
                        "video_path": row["video_path"],
                    }
                    for row in dataset
                ],
            },
        )

    monkeypatch.setattr("opensportslib.core.trainer.vqa_trainer.Trainer_VQA", _fake_trainer_factory)

    video_path = Path("/tmp/sample.mp4")
    video_path.write_bytes(b"video")
    predictions = api.infer(
        video_path=str(video_path),
        question="Is it a foul or not? Why?",
        use_wandb=False,
    )

    assert predictions["task"] == "vqa"
    assert predictions["data"][0]["answer_text"] == "answer::Is it a foul or not? Why?"
    assert calls[-1] == "default-adapter"

    api.infer(
        video_path=str(video_path),
        question="What card would you give? Why?",
        weights="override-adapter",
        use_wandb=False,
    )
    assert calls[-1] == "override-adapter"


def test_vqa_single_video_multi_question_infer(vqa_config_path, monkeypatch):
    api = VQAModel(config=vqa_config_path)
    video_path = Path("/tmp/clip_0.mp4")
    video_path.write_bytes(b"video")
    api.config = SimpleNamespace(
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                splits=SimpleNamespace(
                    test=SimpleNamespace(annotation_path="/tmp/test.json"),
                )
            )
        ),
        MODEL=SimpleNamespace(
            load=SimpleNamespace(checkpoint_path=None),
            components=SimpleNamespace(
                video_encoder=SimpleNamespace(params=SimpleNamespace(feature_source="raw_video")),
            ),
        ),
        SYSTEM=SimpleNamespace(device="cpu", gpu=SimpleNamespace(count=0, id=0)),
        TRAIN=SimpleNamespace(execution={"training_backend": "xvars_videochatgpt_lora", "prompt": {}, "generation": {}, "xvars": {"feature_source": "raw_video"}}),
        TASK="VQA",
    )
    fake_model = object()
    monkeypatch.setattr("opensportslib.apis.vqa.resolve_config_omega", lambda cfg, weights=None: cfg)
    monkeypatch.setattr("opensportslib.apis.vqa.get_vqa_backend", lambda cfg: "xvars_videochatgpt")
    monkeypatch.setattr("opensportslib.core.utils.config.select_device", lambda cfg: "cpu")
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda cfg_path, cfg, run_id, use_wandb=False: None,
    )
    monkeypatch.setattr(
        "opensportslib.models.builder.build_model",
        lambda *args, **kwargs: (fake_model, None),
    )
    monkeypatch.setattr(
        "opensportslib.core.trainer.vqa_trainer.Trainer_VQA",
        lambda cfg: SimpleNamespace(
            infer=lambda model, dataset, use_wandb=False: {
                "task": "vqa",
                "data": [
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "answer_text": f"answer::{row['question']}",
                        "video_path": row["video_path"],
                    }
                    for row in dataset
                ],
            },
        ),
    )
    dataset_path = Path("/tmp/vqa-multi-test.json")
    dataset_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "clip_0",
                        "video_path": str(video_path),
                        "questions": [
                            "Is it a foul or not? Why?",
                            "What card would you give? Why?",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    predictions = api.infer(test_set=str(dataset_path), use_wandb=False)

    assert predictions["task"] == "vqa"
    assert [row["id"] for row in predictions["data"]] == ["clip_0:0", "clip_0:1"]
    assert [row["question"] for row in predictions["data"]] == [
        "Is it a foul or not? Why?",
        "What card would you give? Why?",
    ]
    assert predictions["data"][0]["video_path"] == "/tmp/clip_0.mp4"


def test_vqa_dataset_infer_expands_all_questions(vqa_config_path, tmp_path, monkeypatch):
    dataset_path = tmp_path / "test.json"
    dataset_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "action_0",
                        "video_path": "/tmp/action_0.mp4",
                        "questions": [
                            "Is it a foul or not? Why?",
                            "What card would you give? Why?",
                        ],
                    },
                    {
                        "id": "action_1",
                        "inputs": [{"type": "video", "path": "relative/action_1.mp4"}],
                        "question": "Could the referee have given advantage? Why?",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    api = VQAModel(config=vqa_config_path)
    api.config = SimpleNamespace(
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                splits=SimpleNamespace(
                    test=SimpleNamespace(annotation_path=str(dataset_path)),
                )
            )
        ),
        MODEL=SimpleNamespace(
            load=SimpleNamespace(checkpoint_path=None),
            components=SimpleNamespace(
                video_encoder=SimpleNamespace(params=SimpleNamespace(feature_source="raw_video")),
            ),
        ),
        SYSTEM=SimpleNamespace(device="cpu", gpu=SimpleNamespace(count=0, id=0)),
        TRAIN=SimpleNamespace(execution={"training_backend": "xvars_videochatgpt_lora", "prompt": {}, "generation": {}, "xvars": {"feature_source": "raw_video"}}),
        TASK="VQA",
    )
    fake_model = object()
    monkeypatch.setattr("opensportslib.apis.vqa.resolve_config_omega", lambda cfg, weights=None: cfg)
    monkeypatch.setattr("opensportslib.apis.vqa.get_vqa_backend", lambda cfg: "xvars_videochatgpt")
    monkeypatch.setattr("opensportslib.core.utils.config.select_device", lambda cfg: "cpu")
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda cfg_path, cfg, run_id, use_wandb=False: None,
    )
    monkeypatch.setattr(
        "opensportslib.models.builder.build_model",
        lambda *args, **kwargs: (fake_model, None),
    )
    monkeypatch.setattr(
        "opensportslib.core.trainer.vqa_trainer.Trainer_VQA",
        lambda cfg: SimpleNamespace(
            infer=lambda model, dataset, use_wandb=False: {
                "task": "vqa",
                "data": [
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "answer_text": f"answer::{row['question']}",
                        "video_path": row["video_path"],
                    }
                    for row in dataset
                ],
            },
        ),
    )
    predictions = api.infer(test_set=str(dataset_path), use_wandb=False)

    assert predictions["task"] == "vqa"
    assert len(predictions["data"]) == 3
    assert [row["id"] for row in predictions["data"]] == [
        "action_0:0",
        "action_0:1",
        "action_1",
    ]
    assert predictions["data"][2]["video_path"] == str(
        (dataset_path.parent / "relative/action_1.mp4").resolve()
    )


def test_build_xvars_prompt_injects_classifier_priors():
    from opensportslib.models.utils.vqa_prompting import build_xvars_prompt

    class FakeState:
        def __init__(self):
            self.roles = ("USER", "ASSISTANT")
            self.messages = []
            self.sep = " "
            self.sep2 = "</s>"
            self.sep_style = SimpleNamespace(name="TWO")
            self.action_prompt = ""
            self.offence_prompt = ""

        def copy(self):
            return FakeState()

        def append_message(self, role, message):
            self.messages.append([role, message])

        def set_predictions(self, action_class, offence_class):
            self.action_prompt = action_class
            self.offence_prompt = offence_class

        def get_prompt(self):
            return (
                f"{self.messages[0][0]}:{self.messages[0][1]}"
                f"|priors:{self.action_prompt}{self.offence_prompt}"
            )

    prompt = build_xvars_prompt(
        system_prompt="System.",
        question="What card would you give? Why?",
        prior_text="a tackle, foul and a yellow card",
        video_token_len=2,
    )

    stop_str = "</s>"
    assert "<vid_patch>" in prompt
    assert "a tackle, foul and a yellow card" in prompt
    assert "<video>" not in prompt
    assert stop_str == "</s>"


def test_vqa_save_predictions_round_trip(vqa_config_path, tmp_path, monkeypatch):
    api = VQAModel(config=vqa_config_path)
    api.config = SimpleNamespace(
        SYSTEM=SimpleNamespace(device="cpu", gpu=SimpleNamespace(count=0, id=0)),
        MODEL=SimpleNamespace(load=SimpleNamespace(checkpoint_path=None)),
        TRAIN=SimpleNamespace(execution={"training_backend": "xvars_videochatgpt_lora", "prompt": {}, "generation": {}}),
        TASK="VQA",
    )
    fake_model = object()
    monkeypatch.setattr("opensportslib.apis.vqa.resolve_config_omega", lambda cfg, weights=None: cfg)
    monkeypatch.setattr("opensportslib.apis.vqa.get_vqa_backend", lambda cfg: "xvars_videochatgpt")
    monkeypatch.setattr("opensportslib.core.utils.config.select_device", lambda cfg: "cpu")
    monkeypatch.setattr(
        "opensportslib.models.builder.build_model",
        lambda *args, **kwargs: (fake_model, None),
    )
    monkeypatch.setattr(
        "opensportslib.core.utils.wandb.init_wandb",
        lambda cfg_path, cfg, run_id, use_wandb=False: None,
    )
    monkeypatch.setattr(
        "opensportslib.core.trainer.vqa_trainer.Trainer_VQA",
        lambda cfg: SimpleNamespace(
            infer=lambda model, dataset, use_wandb=False: {
                "task": "vqa",
                "data": [
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "answer_text": f"answer::{row['question']}",
                        "video_path": row["video_path"],
                    }
                    for row in dataset
                ],
            },
        ),
    )
    video_path = tmp_path / "clip_0.mp4"
    video_path.write_bytes(b"video")
    predictions = api.infer(
        video_path=str(video_path),
        question="Is it a foul or not? Why?",
        use_wandb=False,
    )

    out_path = tmp_path / "predictions.json"
    saved = api.save_predictions(str(out_path), predictions)

    assert saved == str(out_path)
    assert json.loads(out_path.read_text(encoding="utf-8"))["task"] == "vqa"


def test_vqa_train_and_evaluate_methods_exist(vqa_config_path):
    api = VQAModel(config=vqa_config_path)
    assert callable(api.train)
    assert callable(api.evaluate)


def test_vqa_import_does_not_pull_gradio_or_demo_modules():
    assert "gradio" not in sys.modules
    assert "video_chatgpt.demo.chat" not in sys.modules
    assert "video_chatgpt.demo" not in sys.modules


def test_vqa_runtime_dependency_error_names_missing_package(monkeypatch):
    del monkeypatch


def test_vqa_qwen_train_requires_qwen_training_backend(vqa_config_path, monkeypatch):
    api = VQAModel(config=vqa_config_path)
    api.config = SimpleNamespace(
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                splits=SimpleNamespace(
                    train=SimpleNamespace(annotation_path="/tmp/train.json"),
                    valid=SimpleNamespace(annotation_path="/tmp/valid.json"),
                )
            )
        ),
        SYSTEM=SimpleNamespace(gpu=SimpleNamespace(count=0)),
        TRAIN=SimpleNamespace(execution={"training_backend": "xvars_videochatgpt_lora"}),
        TASK="VQA",
    )

    monkeypatch.setattr("opensportslib.apis.vqa.resolve_config_omega", lambda cfg, weights=None: cfg)
    monkeypatch.setattr("opensportslib.apis.vqa.get_vqa_backend", lambda cfg: "qwen_xvars_infer")

    with pytest.raises(ValueError, match="requires TRAIN.execution.training_backend='qwen_xvars_lora'"):
        api.train(use_wandb=False)


def test_vqa_qwen_infer_accepts_adapter_weights(vqa_config_path, tmp_path, monkeypatch):
    api = VQAModel(config=vqa_config_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "training_metadata.json").write_text(json.dumps({"backend": "qwen_xvars_lora"}), encoding="utf-8")
    api.config = SimpleNamespace(
        SYSTEM=SimpleNamespace(device="cpu", gpu=SimpleNamespace(count=0, id=0)),
        MODEL=SimpleNamespace(load=SimpleNamespace(checkpoint_path=None)),
        TRAIN=SimpleNamespace(execution={"training_backend": "qwen_xvars_lora", "prompt": {}, "generation": {}}),
        TASK="VQA",
    )
    captured = {}
    fake_model = object()

    def _fake_load(weights):
        captured["loaded_weights"] = weights
        return weights

    def _fake_infer(model, dataset, use_wandb=False):
        captured["infer_model"] = model
        captured["infer_dataset"] = list(dataset)
        return {"task": "vqa", "data": [{"id": "clip", "question": "Was it a foul?", "answer_text": "yes", "video_path": str(video_path)}]}

    monkeypatch.setattr("opensportslib.apis.vqa.resolve_config_omega", lambda cfg, weights=None: cfg)
    monkeypatch.setattr("opensportslib.apis.vqa.get_vqa_backend", lambda cfg: "qwen_xvars_infer")
    monkeypatch.setattr("opensportslib.core.utils.config.select_device", lambda cfg: "cpu")
    fake_trainer_module = types.ModuleType("opensportslib.core.trainer.vqa_trainer")
    fake_trainer_module.Trainer_VQA = lambda cfg: SimpleNamespace(load=_fake_load, infer=_fake_infer)
    monkeypatch.setitem(sys.modules, "opensportslib.core.trainer.vqa_trainer", fake_trainer_module)
    monkeypatch.setattr(VQAModel, "_init_wandb", lambda self, use_wandb=False: None)
    monkeypatch.setattr(
        "opensportslib.models.builder.build_model",
        lambda *args, **kwargs: (fake_model, None),
    )

    out = api.infer(video_path=str(video_path), question="Was it a foul?", weights=str(adapter_dir), use_wandb=False)

    assert captured["loaded_weights"] == str(adapter_dir)
    assert captured["infer_model"] is fake_model
    assert captured["infer_dataset"][0]["question"] == "Was it a foul?"
    assert out["data"][0]["answer_text"] == "yes"


def test_vqa_qwen_vl_native_train_requires_native_training_backend(vqa_config_path, monkeypatch):
    api = VQAModel(config=vqa_config_path)
    api.config = SimpleNamespace(
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                splits=SimpleNamespace(
                    train=SimpleNamespace(annotation_path="/tmp/train.json"),
                    valid=SimpleNamespace(annotation_path="/tmp/valid.json"),
                )
            )
        ),
        SYSTEM=SimpleNamespace(gpu=SimpleNamespace(count=0)),
        TRAIN=SimpleNamespace(execution={"training_backend": "qwen_xvars_lora"}),
        TASK="VQA",
    )

    monkeypatch.setattr("opensportslib.apis.vqa.resolve_config_omega", lambda cfg, weights=None: cfg)
    monkeypatch.setattr("opensportslib.apis.vqa.get_vqa_backend", lambda cfg: "qwen_vl_native_infer")

    with pytest.raises(ValueError, match="requires TRAIN.execution.training_backend='qwen_vl_native_lora'"):
        api.train(use_wandb=False)


def test_vqa_qwen_vl_native_infer_accepts_adapter_weights(vqa_config_path, tmp_path, monkeypatch):
    api = VQAModel(config=vqa_config_path)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "training_metadata.json").write_text(
        json.dumps({"backend": "qwen_vl_native_lora"}),
        encoding="utf-8",
    )
    api.config = SimpleNamespace(
        SYSTEM=SimpleNamespace(device="cpu", gpu=SimpleNamespace(count=0, id=0)),
        MODEL=SimpleNamespace(load=SimpleNamespace(checkpoint_path=None)),
        TRAIN=SimpleNamespace(execution={"training_backend": "qwen_vl_native_lora", "prompt": {}, "generation": {}}),
        TASK="VQA",
    )
    captured = {}
    fake_model = object()

    def _fake_load(weights):
        captured["loaded_weights"] = weights
        return weights

    def _fake_infer(model, dataset, use_wandb=False):
        captured["infer_model"] = model
        captured["infer_dataset"] = list(dataset)
        return {"task": "vqa", "data": [{"id": "clip", "question": "Was it a foul?", "answer_text": "yes", "video_path": str(video_path)}]}

    monkeypatch.setattr("opensportslib.apis.vqa.resolve_config_omega", lambda cfg, weights=None: cfg)
    monkeypatch.setattr("opensportslib.apis.vqa.get_vqa_backend", lambda cfg: "qwen_vl_native_infer")
    monkeypatch.setattr("opensportslib.core.utils.config.select_device", lambda cfg: "cpu")
    fake_trainer_module = types.ModuleType("opensportslib.core.trainer.vqa_trainer")
    fake_trainer_module.Trainer_VQA = lambda cfg: SimpleNamespace(load=_fake_load, infer=_fake_infer)
    monkeypatch.setitem(sys.modules, "opensportslib.core.trainer.vqa_trainer", fake_trainer_module)
    monkeypatch.setattr(VQAModel, "_init_wandb", lambda self, use_wandb=False: None)
    monkeypatch.setattr(
        "opensportslib.models.builder.build_model",
        lambda *args, **kwargs: (fake_model, None),
    )

    out = api.infer(video_path=str(video_path), question="Was it a foul?", weights=str(adapter_dir), use_wandb=False)

    assert captured["loaded_weights"] == str(adapter_dir)
    assert captured["infer_model"] is fake_model
    assert captured["infer_dataset"][0]["frame_paths"] == []
    assert out["data"][0]["answer_text"] == "yes"


def test_vqa_train_ddp_uses_requested_gpu_count(vqa_config_path, monkeypatch):
    api = VQAModel(config=vqa_config_path)
    api.config = SimpleNamespace(
        DATA=SimpleNamespace(
            common=SimpleNamespace(
                splits=SimpleNamespace(
                    train=SimpleNamespace(annotation_path="/tmp/train.json"),
                    valid=SimpleNamespace(annotation_path="/tmp/valid.json"),
                )
            )
        ),
        SYSTEM=SimpleNamespace(gpu=SimpleNamespace(count=2)),
        TRAIN=SimpleNamespace(execution={"training_backend": "qwen_xvars_lora"}),
        TASK="VQA",
    )

    monkeypatch.setattr("opensportslib.apis.vqa.resolve_config_omega", lambda cfg, weights=None: cfg)
    monkeypatch.setattr("opensportslib.apis.vqa.get_vqa_backend", lambda cfg: "qwen_xvars_infer")

    captured = {}

    class _FakeQueue:
        def __init__(self):
            self.value = None

        def put(self, value):
            self.value = value

        def get(self):
            return self.value

    fake_queue = _FakeQueue()

    class _FakeContext:
        def SimpleQueue(self):
            return fake_queue

    class _FakeMP:
        @staticmethod
        def get_context(method):
            captured["context_method"] = method
            return _FakeContext()

        @staticmethod
        def spawn(fn, args=(), nprocs=1):
            captured["spawn_fn"] = fn
            captured["spawn_args"] = args
            captured["nprocs"] = nprocs
            args[3].put("/tmp/qwen-ddp-ckpt")

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(device_count=lambda: 8),
        multiprocessing=_FakeMP,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", _FakeMP)

    ckpt = api.train(use_wandb=False)

    assert ckpt == "/tmp/qwen-ddp-ckpt"
    assert captured["context_method"] == "spawn"
    assert captured["nprocs"] == 2
    assert captured["spawn_args"][0] == 2
