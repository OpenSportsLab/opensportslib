from types import SimpleNamespace
import types
import sys

from opensportslib.models.base.vqa import VQABaselineModel
from opensportslib.models.utils.vqa_prompting import build_prior_text
from opensportslib.models.builder import build_model


def test_build_prior_text_from_labels_and_metadata():
    labels = {
        "action": {"label": "Tackling"},
        "offence": {"label": "Offence: No card"},
        "contact": {"label": "With contact"},
    }
    metadata = {"league": "TestLeague"}
    text = build_prior_text(labels, metadata)
    assert "action=Tackling" in text
    assert "offence=Offence: No card" in text
    assert "league=TestLeague" in text


def test_vqa_builder_defaults_to_baseline(vqa_config_path):
    from opensportslib.core.config import load_config_omega
    from opensportslib.core.utils.config import select_device

    cfg = load_config_omega(vqa_config_path)
    model, _ = build_model(cfg, select_device(cfg.SYSTEM))
    assert isinstance(model, VQABaselineModel)


def test_vqa_builder_selects_xvars_hf_backend(monkeypatch, vqa_config_path):
    from opensportslib.core.config import load_config_omega
    from opensportslib.core.utils.config import select_device

    class FakeMM:
        def __init__(self, config, model_id, projector_params=None):
            self.config = config
            self.model_id = model_id
            self.projector_params = projector_params or {}

    fake_module = types.ModuleType("opensportslib.models.base.vqa")
    fake_module.MultimodalHFVQAModel = FakeMM
    monkeypatch.setitem(sys.modules, "opensportslib.models.base.vqa", fake_module)

    cfg = load_config_omega(vqa_config_path)
    cfg.MODEL.metadata = SimpleNamespace(backend="xvars_hf")
    cfg.MODEL.components.llm_decoder.source.provider = "huggingface"
    cfg.MODEL.components.llm_decoder.params = SimpleNamespace(repo_id="distilgpt2")

    model, _ = build_model(cfg, select_device(cfg.SYSTEM))
    assert isinstance(model, FakeMM)
    assert model.model_id == "distilgpt2"

