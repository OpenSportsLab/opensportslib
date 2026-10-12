"""Offline decoder lifecycle and multimodal generation contracts."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from opensportslib.core.utils import hf_runtime
from tests.helpers.causal_lm import TinyGenerateLM, TinyTokenizer


@pytest.fixture
def decoder(monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = TinyTokenizer()
    tokenizer.decode = Mock(side_effect=lambda ids, **kwargs: ",".join(map(str, ids.tolist())))
    model = TinyGenerateLM()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", Mock(return_value=model))
    runtime = hf_runtime.HFCausalDecoderRuntime("example/decoder", prefer_cuda=False)
    assert runtime.is_ready and runtime.error is None
    assert not model.training
    return runtime, tokenizer, model


@pytest.mark.parametrize("previous", [None, "0"])
def test_offline_mode_restores_environment_after_failure(monkeypatch, previous):
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    for key in keys:
        if previous is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, previous)
    with pytest.raises(RuntimeError, match="load failed"):
        with hf_runtime.hf_offline_if_requested(True):
            assert all(os.environ[key] == "1" for key in keys)
            with hf_runtime.hf_offline_if_requested(False):
                assert all(os.environ[key] == "1" for key in keys)
            raise RuntimeError("load failed")
    assert all(os.environ.get(key) == previous for key in keys)


def test_generation_truncates_prompt_and_honors_token_cap(decoder):
    runtime, tokenizer, model = decoder
    model.config.max_position_embeddings = 6
    prompt = "one two three four five six seven"
    expected = tokenizer(prompt, return_tensors="pt")["input_ids"][:, -4:]
    answer = runtime.generate(prompt, {"max_new_tokens": 20, "max_new_tokens_cap": 2, "temperature": 0})
    assert answer == "9,1"
    assert torch.equal(model.generate_kwargs["input_ids"], expected)
    assert model.generate_kwargs["max_new_tokens"] == 2
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["temperature"] is None
    assert tokenizer.decode.call_args.args[0].tolist() == [9, 1]


def test_video_features_replace_only_patch_embeddings(decoder):
    runtime, tokenizer, model = decoder
    prompt = "before <vid_start><vid_patch><vid_patch><vid_end> after"
    features = [[1., 2., 3., 4.], [5., 6., 7., 8.]]
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    expected = model.get_input_embeddings()(ids).detach()
    answer = runtime.generate(prompt, {"temperature": 0.5}, video_features=features)
    actual = model.generate_kwargs["inputs_embeds"]
    assert torch.equal(actual[0, 2:4], torch.tensor(features))
    assert torch.equal(actual[0, [0, 1, 4, 5]], expected[0, [0, 1, 4, 5]])
    assert model.generate_kwargs["do_sample"] is True
    assert answer == "9,1"


@pytest.mark.parametrize("prompt,features,message", [
    ("no video block", [[1.] * 4], "Missing required"),
    ("<vid_end><vid_patch><vid_start>", [[1.] * 4], "before"),
    ("<vid_patch><vid_start><vid_end>", [[1.] * 4], "between"),
    ("<vid_start><vid_patch><vid_end>", [1.] * 4, "2D tensor"),
    ("<vid_start><vid_patch><vid_end>", [[1.] * 4] * 2, "Patch-feature mismatch"),
    ("<vid_start><vid_patch><vid_end>", [[1.] * 3], "Embedding dim mismatch"),
])
def test_invalid_video_alignment_fails_before_generation(decoder, prompt, features, message):
    runtime, _, model = decoder
    with pytest.raises(ValueError, match=message):
        runtime.generate(prompt, video_features=features)
    assert model.generate_kwargs is None


def test_generation_retries_transient_failure_with_backoff(decoder, monkeypatch):
    runtime, _, model = decoder
    generate = model.generate
    model.generate = Mock(side_effect=[RuntimeError("temporary"), generate(input_ids=torch.tensor([[5]]))])
    sleep = Mock()
    monkeypatch.setattr(hf_runtime.time, "sleep", sleep)
    assert runtime.generate("hello", {"retry_count": 1, "retry_backoff_s": 0.25}) == "9,1"
    assert model.generate.call_count == 2
    sleep.assert_called_once_with(0.25)


def test_generation_raises_last_error_after_retry_budget(decoder):
    runtime, _, model = decoder
    model.generate = Mock(side_effect=RuntimeError("decoder failure"))
    with pytest.raises(RuntimeError, match="decoder failure"):
        runtime.generate("hello", {"retry_count": 2})
    assert model.generate.call_count == 3


def test_generation_timeout_does_not_return_late_answer(decoder, monkeypatch):
    runtime, _, _ = decoder
    monkeypatch.setattr(hf_runtime.time, "time", Mock(side_effect=[10., 12.]))
    with pytest.raises(TimeoutError, match="Generation exceeded"):
        runtime.generate("hello", {"timeout_s": 1})


def test_generation_falls_back_to_full_output_for_older_decoders(decoder):
    runtime, tokenizer, _ = decoder
    tokenizer.decode.side_effect = ["", " fallback answer "]
    assert runtime.generate("hello") == "fallback answer"
    assert tokenizer.decode.call_count == 2


def test_failed_decoder_load_exposes_reason_and_cannot_generate(monkeypatch):
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(side_effect=OSError("missing local model")))
    runtime = hf_runtime.HFCausalDecoderRuntime("example/missing", prefer_cuda=False, local_files_only=True)
    assert not runtime.is_ready
    assert runtime.hidden_size == 0
    assert "missing local model" in runtime.error
    with pytest.raises(RuntimeError, match="missing local model"):
        runtime.generate("hello")


@pytest.mark.parametrize("path_kind", ["absent", "empty", "local", "hub"])
def test_adapter_loading_requires_artifacts_and_preserves_base_model(tmp_path, monkeypatch, path_kind):
    base, adapted = object(), object()
    load = Mock(return_value=adapted)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load)))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    if path_kind == "local":
        (adapter / "adapter_model.safetensors").write_bytes(b"test")
    path = None if path_kind == "absent" else ("example/adapter" if path_kind == "hub" else str(adapter))
    result, status = hf_runtime.load_peft_adapter_if_available(base, path)
    if path_kind in {"local", "hub"}:
        assert result is adapted and status == "loaded"
        load.assert_called_once_with(base, path)
    else:
        assert result is base and status == "not_found"
        load.assert_not_called()


def test_missing_peft_is_reported_without_losing_base_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "peft", None)
    base = object()
    assert hf_runtime.load_peft_adapter_if_available(base, "example/adapter") == (base, "missing_peft")
    with pytest.raises(hf_runtime.OptionalDependencyError, match="pip install peft"):
        hf_runtime.require_optional_package("peft")
