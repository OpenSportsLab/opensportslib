from pathlib import Path
from types import SimpleNamespace
import inspect
import json
import sys

import pytest

from opensportslib.apis import VQAModel
from opensportslib.apis._xvars_runtime import (
    XVarsHeadlessRuntime,
    XVarsPrediction,
    _dependency_error,
    build_xvars_prompt,
)


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
        return XVarsPrediction(
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
    runtime = _FakeRuntime()

    def fake_get_runtime(self, weights=None):
        calls.append(weights)
        self.last_loaded_weights = weights
        self.best_checkpoint = weights
        return runtime

    monkeypatch.setattr(VQAModel, "_get_runtime", fake_get_runtime)
    api = VQAModel(config=vqa_config_path, weights="default-adapter")

    predictions = api.infer(
        video_path="/tmp/sample.mp4",
        question="Is it a foul or not? Why?",
        use_wandb=False,
    )

    assert predictions["task"] == "vqa"
    assert predictions["data"][0]["answer_text"] == "answer::Is it a foul or not? Why?"
    assert calls[-1] == "default-adapter"

    api.infer(
        video_path="/tmp/sample.mp4",
        question="What card would you give? Why?",
        weights="override-adapter",
        use_wandb=False,
    )
    assert calls[-1] == "override-adapter"


def test_vqa_single_video_multi_question_infer(vqa_config_path, monkeypatch):
    runtime = _FakeRuntime()
    monkeypatch.setattr(VQAModel, "_get_runtime", lambda self, weights=None: runtime)

    api = VQAModel(config=vqa_config_path)
    predictions = api.infer(
        video_path="/tmp/clip_0.mp4",
        questions=[
            "Is it a foul or not? Why?",
            "What card would you give? Why?",
        ],
        use_wandb=False,
    )

    assert predictions["task"] == "vqa"
    assert [row["id"] for row in predictions["data"]] == ["clip_0:0", "clip_0:1"]
    assert [row["question"] for row in predictions["data"]] == [
        "Is it a foul or not? Why?",
        "What card would you give? Why?",
    ]
    assert predictions["data"][0]["video_path"] == "/tmp/clip_0.mp4"


def test_vqa_dataset_infer_expands_all_questions(vqa_config_path, tmp_path, monkeypatch):
    runtime = _FakeRuntime()
    monkeypatch.setattr(VQAModel, "_get_runtime", lambda self, weights=None: runtime)

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

    prompt, stop_str = build_xvars_prompt(
        conv_templates={"video-chatgpt_v1": FakeState()},
        conv_mode="video-chatgpt_v1",
        question="What card would you give? Why?",
        replace_token="<vid_tokens>",
        video_token="<video>",
        action_prompt="a tackle",
        offence_prompt=", foul and a yellow card",
    )

    assert "<vid_tokens>" in prompt
    assert "a tackle, foul and a yellow card" in prompt
    assert "<video>" not in prompt
    assert stop_str == "</s>"


def test_vqa_save_predictions_round_trip(vqa_config_path, tmp_path, monkeypatch):
    runtime = _FakeRuntime()
    monkeypatch.setattr(VQAModel, "_get_runtime", lambda self, weights=None: runtime)

    api = VQAModel(config=vqa_config_path)
    predictions = api.infer(
        video_path="/tmp/clip_0.mp4",
        question="Is it a foul or not? Why?",
        use_wandb=False,
    )

    out_path = tmp_path / "predictions.json"
    saved = api.save_predictions(str(out_path), predictions)

    assert saved == str(out_path)
    assert json.loads(out_path.read_text(encoding="utf-8"))["task"] == "vqa"


def test_vqa_train_and_evaluate_raise_not_implemented(vqa_config_path):
    api = VQAModel(config=vqa_config_path)

    with pytest.raises(NotImplementedError):
        api.train(use_wandb=False)

    with pytest.raises(NotImplementedError):
        api.evaluate(use_wandb=False)


def test_vqa_import_does_not_pull_gradio_or_demo_modules():
    assert "gradio" not in sys.modules
    assert "video_chatgpt.demo.chat" not in sys.modules
    assert "video_chatgpt.demo" not in sys.modules


def test_vqa_runtime_dependency_error_names_missing_package(monkeypatch):
    runtime = XVarsHeadlessRuntime()

    def fail_bootstrap():
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    monkeypatch.setattr(runtime, "_bootstrap_components", fail_bootstrap)
    error = _dependency_error(ModuleNotFoundError("No module named 'numpy'", name="numpy"))
    assert "numpy" in str(error)
