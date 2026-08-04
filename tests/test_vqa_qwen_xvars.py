from types import SimpleNamespace

import torch

from opensportslib.models.base.qwen_xvars import QWEN_MM_PROJECTOR_FILE, QwenXVarsCausalLM, QwenXVarsModel
from opensportslib.models.builder import build_model


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "</s>"

    def __init__(self):
        self.vocab = {"<pad>": 0, "</s>": 1, "<vid_patch>": 2, "<vid_start>": 3, "<vid_end>": 4}

    def get_vocab(self):
        return dict(self.vocab)

    def add_special_tokens(self, payload):
        added = 0
        for token in payload.get("additional_special_tokens", []):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, return_tensors=None, **kwargs):
        del kwargs
        if isinstance(text, list):
            text = text[0]
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for token in ("<vid_start>", "<vid_patch>", "<vid_end>", "</s>"):
                if text.startswith(token, i):
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace() and not any(
                text.startswith(token, j) for token in ("<vid_start>", "<vid_patch>", "<vid_end>", "</s>")
            ):
                j += 1
            tokens.append(text[i:j])
            i = j
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        attn = [1] * len(ids)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([attn])}
        return SimpleNamespace(input_ids=ids, attention_mask=attn)

    def batch_decode(self, sequences, skip_special_tokens=True):
        del sequences, skip_special_tokens
        return ["qwen answer </s>"]


class TinyGenerateLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, vocab_size=64, use_cache=True)
        self.generation_config = SimpleNamespace(use_cache=False)
        self.emb = torch.nn.Embedding(128, 4)
        self.lm = torch.nn.Linear(4, 64)
        self.seen_inputs_embeds = None
        self.generate_kwargs = None

    def get_input_embeddings(self):
        return self.emb

    def resize_token_embeddings(self, size):
        return self.emb

    def save_pretrained(self, output_dir):
        del output_dir

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        del input_ids, attention_mask, kwargs
        self.seen_inputs_embeds = inputs_embeds.detach().clone()
        logits = self.lm(inputs_embeds)
        loss = logits.sum() * 0
        if labels is not None:
            loss = loss + 0.123
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        del attention_mask
        self.generate_kwargs = {"input_ids": input_ids, "inputs_embeds": inputs_embeds, **kwargs}
        extra = torch.tensor([[9, 1]], device=input_ids.device if input_ids is not None else inputs_embeds.device)
        if input_ids is None:
            prefix = torch.zeros((1, inputs_embeds.shape[1]), dtype=torch.long, device=extra.device)
            return torch.cat((prefix, extra), dim=1)
        return torch.cat((input_ids, extra), dim=1)


def _cfg(tmp_path):
    return SimpleNamespace(
        TASK="VQA",
        SYSTEM=SimpleNamespace(device="cpu", gpu=SimpleNamespace(count=0, id=0)),
        MODEL=SimpleNamespace(
            runtime=SimpleNamespace(dtype="fp32"),
            load=SimpleNamespace(checkpoint_path=None),
            components=SimpleNamespace(
                video_encoder=SimpleNamespace(
                    kind="encoder",
                    source=SimpleNamespace(provider="opensportslib", name="xvars_clip_features"),
                    load=SimpleNamespace(weights_path=str(tmp_path / "vision.pt")),
                    params=SimpleNamespace(feature_source="indexed_or_raw_clip", vision_tower="clip", feature_dim=1024),
                    overrides=SimpleNamespace(),
                ),
                mm_projector=SimpleNamespace(
                    kind="projector",
                    source=SimpleNamespace(provider="opensportslib"),
                    params=SimpleNamespace(input_dim=1024),
                    overrides=SimpleNamespace(),
                ),
                llm_decoder=SimpleNamespace(
                    kind="decoder",
                    source=SimpleNamespace(provider="huggingface", name="Qwen/Qwen3.5-9B-Base"),
                    params=SimpleNamespace(repo_id="Qwen/Qwen3.5-9B-Base"),
                    overrides=SimpleNamespace(),
                ),
            ),
            metadata=SimpleNamespace(backend="qwen_xvars_infer"),
        ),
        DATA=SimpleNamespace(
            inputs=SimpleNamespace(video=SimpleNamespace(sampling=SimpleNamespace(num_frames=100, start_frame=63, end_frame=87, input_fps=25, target_fps=17))),
        ),
        TRAIN=SimpleNamespace(
            execution={
                "prompt": {"include_priors": True, "video_token_len": 3},
                "generation": {"max_new_tokens": 8, "temperature": 0.0},
                "xvars": {"feature_mode": "strict_xvars"},
                "hf": {"local_files_only": True, "prefer_cuda": False, "tokenizer_id": "Qwen/Qwen3.5-9B-Base"},
                "quantization": {"enabled": False},
            }
        ),
    )


def test_qwen_xvars_wrapper_projects_features_into_hidden_size():
    tokenizer = TinyTokenizer()
    base = TinyGenerateLM()
    model = QwenXVarsCausalLM(base, mm_hidden_size=1024)
    input_ids = torch.tensor([[tokenizer.convert_tokens_to_ids("<vid_start>"), tokenizer.convert_tokens_to_ids("<vid_patch>"), tokenizer.convert_tokens_to_ids("<vid_patch>"), tokenizer.convert_tokens_to_ids("<vid_end>")]])
    features = torch.ones((1, 2, 1024))

    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids.clone(),
        video_spatio_temporal_features=features,
        tokenizer=tokenizer,
    )

    assert abs(out.loss.item() - 0.123) < 1e-6
    assert model.mm_projector.out_features == base.config.hidden_size
    assert tuple(base.seen_inputs_embeds.shape) == (1, 4, base.config.hidden_size)


def test_qwen_xvars_wrapper_raises_on_patch_feature_mismatch():
    tokenizer = TinyTokenizer()
    base = TinyGenerateLM()
    model = QwenXVarsCausalLM(base, mm_hidden_size=1024)
    input_ids = torch.tensor(
        [[
            tokenizer.convert_tokens_to_ids("<vid_start>"),
            tokenizer.convert_tokens_to_ids("<vid_patch>"),
            tokenizer.convert_tokens_to_ids("<vid_patch>"),
            tokenizer.convert_tokens_to_ids("<vid_end>"),
        ]]
    )
    features = torch.ones((1, 3, 1024))

    try:
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids.clone(),
            video_spatio_temporal_features=features,
            tokenizer=tokenizer,
        )
    except ValueError as exc:
        assert "Patch-feature mismatch" in str(exc)
    else:
        raise AssertionError("Expected patch-feature mismatch to raise ValueError.")


def test_qwen_xvars_model_generate_answer_uses_existing_feature_contract(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    tokenizer = TinyTokenizer()
    base_model = TinyGenerateLM()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: base_model)

    model = QwenXVarsModel(cfg, model_id="Qwen/Qwen3.5-9B-Base", projector_params={"input_dim": 1024})
    answer = model.generate_answer(
        {
            "id": "action_0",
            "question": "Was this a foul?",
            "references": [],
            "labels": {},
            "metadata": {},
            "prior_prediction_text": "",
            "video_spatio_temporal_features": torch.ones((3, 1024)),
        },
        prompt_cfg={"video_token_len": 3},
        generation_cfg={"max_new_tokens": 8, "temperature": 0.0},
    )

    assert model._ready is True
    assert answer == "qwen answer"
    assert base_model.generate_kwargs is not None
    assert base_model.generate_kwargs["inputs_embeds"].shape[-1] == base_model.config.hidden_size
    assert base_model.generate_kwargs["inputs_embeds"].shape[-2] == base_model.generate_kwargs["input_ids"].shape[-1]


def test_qwen_loader_does_not_pass_use_cache_to_from_pretrained(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    tokenizer = TinyTokenizer()
    base_model = TinyGenerateLM()
    captured = {}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _load_tokenizer(*args, **kwargs):
        del args, kwargs
        return tokenizer

    def _load_model(model_id, **kwargs):
        captured["model_id"] = model_id
        captured["model_kwargs"] = dict(kwargs)
        if "use_cache" in kwargs:
            raise TypeError("unexpected load kwarg")
        return base_model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _load_tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _load_model)

    model = QwenXVarsModel(cfg, model_id="Qwen/Qwen3.5-9B-Base", projector_params={"input_dim": 1024})

    assert model._ready is True
    assert captured["model_id"] == "Qwen/Qwen3.5-9B-Base"
    assert "use_cache" not in captured["model_kwargs"]
    assert base_model.config.use_cache is True
    assert base_model.generation_config.use_cache is True


def test_builder_routes_qwen_backend(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    tokenizer = TinyTokenizer()
    base_model = TinyGenerateLM()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: base_model)

    model, _ = build_model(cfg, device="cpu")

    assert isinstance(model, QwenXVarsModel)


def test_qwen_wrapper_save_and_reload_projector(tmp_path):
    base = TinyGenerateLM()
    model = QwenXVarsCausalLM(base, mm_hidden_size=1024)
    with torch.no_grad():
        model.mm_projector.weight.fill_(0.25)
        model.mm_projector.bias.fill_(0.5)

    model.save_pretrained(str(tmp_path))
    reloaded = QwenXVarsCausalLM.from_pretrained_projector(
        TinyGenerateLM(),
        str(tmp_path),
        mm_hidden_size=1024,
    )

    assert (tmp_path / QWEN_MM_PROJECTOR_FILE).exists()
    assert torch.allclose(reloaded.mm_projector.weight, model.mm_projector.weight)
    assert torch.allclose(reloaded.mm_projector.bias, model.mm_projector.bias)


def test_qwen_model_loads_adapter_checkpoint_without_hard_failure(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    cfg.MODEL.load.checkpoint_path = str(adapter_dir)
    tokenizer = TinyTokenizer()
    base_model = TinyGenerateLM()
    captured = {}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import opensportslib.models.base.qwen_xvars as mod

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: base_model)

    def _fake_load(model, adapter_path):
        captured["adapter_path"] = adapter_path
        return model, "loaded"

    monkeypatch.setattr(mod, "load_peft_adapter_if_available", _fake_load)

    model = QwenXVarsModel(cfg, model_id="Qwen/Qwen3.5-9B-Base", projector_params={"input_dim": 1024})

    assert model._ready is True
    assert captured["adapter_path"] == str(adapter_dir)
