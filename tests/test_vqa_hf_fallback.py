from types import SimpleNamespace


def _cfg():
    return SimpleNamespace(
        DATA=SimpleNamespace(inputs=SimpleNamespace(video=SimpleNamespace(sampling=SimpleNamespace()))),
        TRAIN=SimpleNamespace(execution=SimpleNamespace(hf=SimpleNamespace(local_files_only=True, prefer_cuda=False))),
    )


def test_hf_backend_falls_back_to_baseline(monkeypatch):
    import opensportslib.models.base.vqa as mm

    class DummyDecoder:
        def __init__(self, *args, **kwargs):
            self._ready = False
            self._error = "offline"

        @property
        def is_ready(self):
            return self._ready

        @property
        def error(self):
            return self._error

    monkeypatch.setattr(mm, "HFCausalDecoderRuntime", DummyDecoder)

    model = mm.MultimodalHFVQAModel(_cfg(), model_id="distilgpt2", projector_params={"input_dim": 270, "output_dim": 8})
    sample = {
        "question": "What card?",
        "labels": {"offence": {"label": "Offence: No card"}, "action": {"label": "Challenge"}},
        "metadata": {},
    }
    out = model.generate_answer(sample, prompt_cfg={"style": "short"}, generation_cfg={})
    assert isinstance(out, str)
    assert out


def test_hf_backend_receives_adapter_checkpoint_path(monkeypatch, tmp_path):
    import opensportslib.models.base.vqa as mm

    captured = {}

    class DummyDecoder:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self._ready = False
            self._error = "offline"

        @property
        def is_ready(self):
            return self._ready

        @property
        def error(self):
            return self._error

    cfg = _cfg()
    cfg.MODEL = SimpleNamespace(load=SimpleNamespace(checkpoint_path=str(tmp_path)))
    monkeypatch.setattr(mm, "HFCausalDecoderRuntime", DummyDecoder)

    mm.MultimodalHFVQAModel(cfg, model_id="distilgpt2", projector_params={"input_dim": 270, "output_dim": 8})
    assert captured["adapter_path"] == str(tmp_path)


def test_hf_backend_respects_fallback_policy_none(monkeypatch):
    import opensportslib.models.base.vqa as mm

    class DummyDecoder:
        def __init__(self, *args, **kwargs):
            self._ready = False
            self._error = "offline"

        @property
        def is_ready(self):
            return self._ready

        @property
        def error(self):
            return self._error

    monkeypatch.setattr(mm, "HFCausalDecoderRuntime", DummyDecoder)
    model = mm.MultimodalHFVQAModel(_cfg(), model_id="distilgpt2", projector_params={"input_dim": 270, "output_dim": 8})

    sample = {"question": "What card?", "labels": {}, "metadata": {}}
    try:
        model.generate_answer(sample, prompt_cfg={"style": "short"}, generation_cfg={"fallback_policy": "none"})
        assert False, "Expected RuntimeError when fallback_policy=none and HF decoder unavailable"
    except RuntimeError:
        assert True
