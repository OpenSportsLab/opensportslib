from types import SimpleNamespace
import torch


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
        "video_spatio_temporal_features": torch.ones((8, 1024), dtype=torch.float32),
    }
    out = model.generate_answer(sample, prompt_cfg={"style": "short"}, generation_cfg={"fallback_policy": "baseline_on_failure"})
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

    sample = {"question": "What card?", "labels": {}, "metadata": {}, "video_spatio_temporal_features": torch.ones((8, 1024))}
    try:
        model.generate_answer(sample, prompt_cfg={"style": "short"}, generation_cfg={"fallback_policy": "none"})
        assert False, "Expected RuntimeError when fallback_policy=none and HF decoder unavailable"
    except RuntimeError:
        assert True


def test_hf_backend_passes_patch_aligned_features(monkeypatch):
    import opensportslib.models.base.vqa as mm
    captured = {}

    class DummyDecoder:
        def __init__(self, *args, **kwargs):
            self._ready = True
            self._error = None

        @property
        def is_ready(self):
            return self._ready

        @property
        def error(self):
            return self._error

        @property
        def hidden_size(self):
            return 32

        def generate(self, prompt, generation_cfg=None, video_features=None):
            del generation_cfg
            captured["prompt"] = prompt
            captured["video_features"] = video_features
            return "ok"

    monkeypatch.setattr(mm, "HFCausalDecoderRuntime", DummyDecoder)
    model = mm.MultimodalHFVQAModel(_cfg(), model_id="distilgpt2", projector_params={"input_dim": 270, "output_dim": 8})
    sample = {"question": "What card?", "labels": {}, "metadata": {}, "video_spatio_temporal_features": torch.ones((12, 1024))}
    out = model.generate_answer(sample, prompt_cfg={"video_token_len": 7}, generation_cfg={})
    assert out == "ok"
    assert isinstance(captured["video_features"], torch.Tensor)
    assert tuple(captured["video_features"].shape) == (7, 32)


def test_ensure_video_special_tokens_resizes_embeddings():
    from opensportslib.core.utils.hf_runtime import _ensure_video_special_tokens

    class DummyTokenizer:
        def __init__(self):
            self._vocab = {"hello": 0}
            self._added = []

        def get_vocab(self):
            return dict(self._vocab)

        def add_special_tokens(self, payload):
            tokens = payload.get("additional_special_tokens", [])
            added = 0
            for tok in tokens:
                if tok not in self._vocab:
                    self._vocab[tok] = len(self._vocab)
                    self._added.append(tok)
                    added += 1
            return added

        def __len__(self):
            return len(self._vocab)

    class DummyModel:
        def __init__(self):
            self.resized_to = None

        def resize_token_embeddings(self, size):
            self.resized_to = int(size)

    tok = DummyTokenizer()
    model = DummyModel()
    added = _ensure_video_special_tokens(tok, model)
    assert added == 3
    assert model.resized_to == len(tok)


def test_hf_backend_generation_mismatch_falls_back(monkeypatch):
    import opensportslib.models.base.vqa as mm

    class DummyDecoder:
        def __init__(self, *args, **kwargs):
            self._ready = True
            self._error = None

        @property
        def is_ready(self):
            return self._ready

        @property
        def error(self):
            return self._error

        @property
        def hidden_size(self):
            return 16

        def generate(self, prompt, generation_cfg=None, video_features=None):
            del prompt, generation_cfg, video_features
            raise ValueError("Patch-feature mismatch")

    monkeypatch.setattr(mm, "HFCausalDecoderRuntime", DummyDecoder)
    model = mm.MultimodalHFVQAModel(_cfg(), model_id="distilgpt2", projector_params={"input_dim": 270, "output_dim": 8})
    sample = {
        "question": "What card?",
        "labels": {"offence": {"label": "Offence: No card"}, "action": {"label": "Challenge"}},
        "metadata": {},
        "video_spatio_temporal_features": torch.ones((8, 1024), dtype=torch.float32),
    }
    out = model.generate_answer(
        sample,
        prompt_cfg={"style": "short", "video_token_len": 4},
        generation_cfg={"fallback_policy": "baseline_on_failure"},
    )
    assert isinstance(out, str) and out


def test_hf_backend_empty_generation_falls_back(monkeypatch):
    import opensportslib.models.base.vqa as mm

    class DummyDecoder:
        def __init__(self, *args, **kwargs):
            self._ready = True
            self._error = None

        @property
        def is_ready(self):
            return self._ready

        @property
        def error(self):
            return self._error

        @property
        def hidden_size(self):
            return 16

        def generate(self, prompt, generation_cfg=None, video_features=None):
            del prompt, generation_cfg, video_features
            return ""

    monkeypatch.setattr(mm, "HFCausalDecoderRuntime", DummyDecoder)
    model = mm.MultimodalHFVQAModel(_cfg(), model_id="distilgpt2", projector_params={"input_dim": 270, "output_dim": 8})
    sample = {
        "question": "What card?",
        "labels": {"offence": {"label": "Offence: No card"}, "action": {"label": "Challenge"}},
        "metadata": {},
        "video_spatio_temporal_features": torch.ones((8, 1024), dtype=torch.float32),
    }
    out = model.generate_answer(
        sample,
        prompt_cfg={"style": "short", "video_token_len": 4},
        generation_cfg={"fallback_policy": "baseline_on_failure"},
    )
    assert isinstance(out, str) and out


def test_hf_backend_rejects_shape_mismatch_when_feature_mode_is_explicit(monkeypatch):
    import opensportslib.models.base.vqa as mm

    class DummyDecoder:
        def __init__(self, *args, **kwargs):
            self._ready = True
            self._error = None

        @property
        def is_ready(self):
            return self._ready

        @property
        def error(self):
            return self._error

        @property
        def hidden_size(self):
            return 16

        def generate(self, prompt, generation_cfg=None, video_features=None):
            del prompt, generation_cfg, video_features
            return "ok"

    monkeypatch.setattr(mm, "HFCausalDecoderRuntime", DummyDecoder)
    cfg = _cfg()
    cfg.TRAIN.execution = SimpleNamespace(
        hf=SimpleNamespace(local_files_only=True, prefer_cuda=False),
        xvars=SimpleNamespace(feature_mode="strict_xvars"),
    )
    model = mm.MultimodalHFVQAModel(cfg, model_id="distilgpt2", projector_params={"input_dim": 270, "output_dim": 8})
    sample = {"question": "What card?", "labels": {}, "metadata": {}, "video_spatio_temporal_features": torch.ones((356, 1024))}

    with pytest.raises(ValueError, match="token count mismatch"):
        model.generate_answer(sample, prompt_cfg={"video_token_len": 300}, generation_cfg={})
