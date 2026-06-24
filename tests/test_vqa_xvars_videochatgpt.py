import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from opensportslib.core.trainer.vqa_trainer import (
    VQAXVarsVideoChatGPTLoraTrainer,
    VQAXVarsVideoChatGPTSFTDataset,
    XVarsVideoChatGPTDataCollator,
    _resolve_sft_per_device_batch_sizes,
    _score_xvars_generated_answers,
)
from opensportslib.models.base.xvars_videochatgpt import XVarsVideoChatGPTCausalLM
from opensportslib.models.utils.vqa_prompting import VIDEO_CHATGPT_SYSTEM_PROMPT, build_xvars_prompt
from opensportslib.models.utils.xvars_clip_index import load_feature_index, load_prediction_index


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "</s>"

    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "</s>": 1,
            "<vid_start>": 2,
            "<vid_patch>": 3,
            "<vid_end>": 4,
        }

    def convert_tokens_to_ids(self, tok):
        return self.vocab[tok]

    def __call__(self, text, truncation=True, max_length=64, padding=None, return_tensors=None):
        toks = []
        i = 0
        while i < len(text):
            matched = False
            for tok in ("<vid_start>", "<vid_patch>", "<vid_end>", "</s>"):
                if text.startswith(tok, i):
                    toks.append(tok)
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace() and not any(
                text.startswith(t, j) for t in ("<vid_start>", "<vid_patch>", "<vid_end>", "</s>")
            ):
                j += 1
            toks.append(text[i:j])
            i = j
        ids = []
        for tok in toks:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab[tok])
        if truncation:
            ids = ids[:max_length]
        attn = [1] * len(ids)
        if padding == "max_length":
            pad = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad
            attn = attn + [0] * pad
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([attn])}
        return {"input_ids": ids, "attention_mask": attn}


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, vocab_size=32)
        self.emb = torch.nn.Embedding(64, 4)
        self.lm = torch.nn.Linear(4, 64)
        self.seen_inputs_embeds = None
        self.gradient_checkpointing_kwargs = None
        self.gradient_checkpointing_disabled = False
        self.input_require_grads = False

    def get_input_embeddings(self):
        return self.emb

    def resize_token_embeddings(self, size):
        return self.emb

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_disabled = True

    def enable_input_require_grads(self):
        self.input_require_grads = True

    def disable_input_require_grads(self):
        self.input_require_grads = False

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        del input_ids, attention_mask, kwargs
        self.seen_inputs_embeds = inputs_embeds.detach().clone()
        logits = self.lm(inputs_embeds)
        loss = logits.sum() * 0
        if labels is not None:
            loss = loss + 0.123
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)


class TinyHalfEmbeddingLM(TinyLM):
    def __init__(self):
        super().__init__()
        self.emb = self.emb.to(torch.float16)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        del input_ids, attention_mask, labels, kwargs
        self.seen_inputs_embeds = inputs_embeds.detach().clone()
        return SimpleNamespace(loss=inputs_embeds.float().sum() * 0 + 0.123, logits=inputs_embeds.float())


class TinyGenerateLM(TinyLM):
    def __init__(self):
        super().__init__()
        self.generate_kwargs = None
        self.generation_config = SimpleNamespace(marker="original")

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4]])


def _sample():
    return {
        "id": "action_0",
        "question": "What card?",
        "references": ["Yellow card because the challenge is reckless."],
        "labels": {"action": {"label": "Challenge"}},
        "metadata": {},
        "prior_prediction_text": "Challenge, foul, yellow card",
        "video_spatio_temporal_features": torch.ones((3, 1024)),
    }


def _cfg(tmp_path, *, dry_run=True):
    return SimpleNamespace(
        SYSTEM=SimpleNamespace(paths=SimpleNamespace(save_dir=str(tmp_path / "ckpt"))),
        MODEL=SimpleNamespace(
            runtime=SimpleNamespace(dtype="fp32"),
            components=SimpleNamespace(
                video_encoder=SimpleNamespace(
                    kind="encoder",
                    source=SimpleNamespace(provider="opensportslib", name="xvars_clip_features"),
                    params=SimpleNamespace(feature_source="indexed"),
                    overrides=SimpleNamespace(),
                ),
                mm_projector=SimpleNamespace(
                    kind="projector",
                    source=SimpleNamespace(provider="opensportslib", name="xvars_mm_projector"),
                    params=SimpleNamespace(input_dim=1024),
                    overrides=SimpleNamespace(),
                ),
                llm_decoder=SimpleNamespace(
                    kind="decoder",
                    source=SimpleNamespace(provider="huggingface", name="tiny"),
                    params=SimpleNamespace(repo_id="tiny"),
                    overrides=SimpleNamespace(),
                ),
            ),
        ),
        TRAIN=SimpleNamespace(
            epochs=1,
            optimizer=SimpleNamespace(type="AdamW", lr=1e-4),
            execution={
                "training_backend": "xvars_videochatgpt_lora",
                "dry_run": dry_run,
                "acc_grad_iter": 1,
                "log_interval": 1,
                "prompt": {"include_priors": True, "video_token_len": 3},
                "sft": {"include_video_tokens": True, "max_seq_length": 64},
                "xvars": {"projection_path": None, "feature_mode": "strict_xvars"},
                "hf": {"local_files_only": True, "prefer_cuda": False},
                "lora": {},
                "quantization": {"enabled": False},
                "checkpoint": {"save_adapter": True},
            }
        ),
    )


def test_sft_batch_sizes_default_to_split_dataloaders(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.DATA = SimpleNamespace(
        common=SimpleNamespace(
            splits=SimpleNamespace(
                train=SimpleNamespace(dataloader=SimpleNamespace(batch_size=4)),
                valid=SimpleNamespace(dataloader=SimpleNamespace(batch_size=2)),
            )
        )
    )
    assert _resolve_sft_per_device_batch_sizes(cfg, {}) == (4, 2)


def test_sft_batch_size_overrides_win_over_split_dataloaders(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.DATA = SimpleNamespace(
        common=SimpleNamespace(
            splits=SimpleNamespace(
                train=SimpleNamespace(dataloader=SimpleNamespace(batch_size=4)),
                valid=SimpleNamespace(dataloader=SimpleNamespace(batch_size=2)),
            )
        )
    )
    assert _resolve_sft_per_device_batch_sizes(
        cfg,
        {"per_device_train_batch_size": 8, "per_device_eval_batch_size": 6},
    ) == (8, 6)


def test_xvars_sft_dataset_and_collator_keep_video_features():
    tok = TinyTokenizer()
    ds = VQAXVarsVideoChatGPTSFTDataset(
        [_sample()],
        tokenizer=tok,
        prompt_cfg={"include_priors": True, "video_token_len": 3},
        sft_cfg={"max_seq_length": 64},
        xvars_cfg={},
    )
    batch = XVarsVideoChatGPTDataCollator(tok)([ds[0]])
    assert tuple(batch["video_spatio_temporal_features"].shape) == (1, 3, 1024)
    assert batch["labels"].shape == batch["input_ids"].shape
    eos_positions = (batch["input_ids"] == tok.eos_token_id).nonzero(as_tuple=False)
    assert eos_positions.numel() > 0
    assert all(batch["labels"][tuple(position)] == tok.eos_token_id for position in eos_positions)


def test_xvars_sft_dataset_flattens_all_reference_answers():
    tok = TinyTokenizer()
    sample = _sample()
    sample["references"] = ["First referee answer.", "Second referee answer."]

    ds = VQAXVarsVideoChatGPTSFTDataset(
        [sample],
        tokenizer=tok,
        prompt_cfg={"include_priors": True, "video_token_len": 3},
        sft_cfg={"max_seq_length": 64, "reference_mode": "all", "append_eos_token": True},
        xvars_cfg={},
    )

    assert len(ds) == 2


def test_xvars_model_forward_injects_video_features_at_patch_tokens():
    tok = TinyTokenizer()
    base = TinyLM()
    model = XVarsVideoChatGPTCausalLM(base, mm_hidden_size=1024)
    with torch.no_grad():
        model.mm_projector.weight.fill_(0.0)
        model.mm_projector.bias.fill_(7.0)
    encoded = tok("USER: q <vid_start><vid_patch><vid_patch><vid_patch><vid_end> ASSISTANT: a", padding="max_length", max_length=16)
    input_ids = torch.tensor([encoded["input_ids"]])
    labels = input_ids.clone()
    features = torch.ones((1, 3, 1024))
    out = model(input_ids=input_ids, labels=labels, video_spatio_temporal_features=features, tokenizer=tok)
    assert float(out.loss) > 0
    patch_positions = (input_ids[0] == tok.convert_tokens_to_ids("<vid_patch>")).nonzero(as_tuple=False).flatten()
    assert torch.all(base.seen_inputs_embeds[0, patch_positions, :] == 7.0)


def test_xvars_wrapper_delegates_gradient_checkpointing_to_decoder():
    base = TinyLM()
    model = XVarsVideoChatGPTCausalLM(base, mm_hidden_size=1024)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    assert base.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert base.input_require_grads is True

    model.gradient_checkpointing_disable()
    model.disable_input_require_grads()
    assert base.gradient_checkpointing_disabled is True
    assert base.input_require_grads is False


def test_xvars_wrapper_reuses_embedded_videochatgpt_projector():
    base = TinyLM()
    base.model = SimpleNamespace(mm_projector=torch.nn.Linear(1024, 4))
    with torch.no_grad():
        base.model.mm_projector.weight.fill_(0.25)
        base.model.mm_projector.bias.fill_(2.0)

    model = XVarsVideoChatGPTCausalLM.from_pretrained_projector(base, mm_hidden_size=1024)

    assert torch.all(model.mm_projector.weight == 0.25)
    assert torch.all(model.mm_projector.bias == 2.0)


def test_xvars_wrapper_loads_raw_projector_when_embedded_copy_is_quantized(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    checkpoint = tmp_path / "videochatgpt"
    checkpoint.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    safetensors.save_file(
        {
            "model.mm_projector.weight": torch.full((4, 1024), 0.5, dtype=torch.float16),
            "model.mm_projector.bias": torch.full((4,), 1.5, dtype=torch.float16),
        },
        checkpoint / shard_name,
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.mm_projector.weight": shard_name,
                    "model.mm_projector.bias": shard_name,
                }
            }
        ),
        encoding="utf-8",
    )

    base = TinyLM()
    base.config._name_or_path = str(checkpoint)
    base.model = SimpleNamespace(mm_projector=torch.nn.Linear(1, 1))
    model = XVarsVideoChatGPTCausalLM.from_pretrained_projector(base, mm_hidden_size=1024)

    assert torch.all(model.mm_projector.weight == 0.5)
    assert torch.all(model.mm_projector.bias == 1.5)


def test_xvars_model_forward_ignores_peft_inputs_embeds_kwarg():
    tok = TinyTokenizer()
    base = TinyLM()
    model = XVarsVideoChatGPTCausalLM(base, mm_hidden_size=1024)
    with torch.no_grad():
        model.mm_projector.weight.fill_(0.0)
        model.mm_projector.bias.fill_(5.0)
    encoded = tok("USER: q <vid_start><vid_patch><vid_patch><vid_patch><vid_end> ASSISTANT: a", padding="max_length", max_length=16)
    input_ids = torch.tensor([encoded["input_ids"]])
    features = torch.ones((1, 3, 1024))

    out = model(
        input_ids=input_ids,
        labels=input_ids.clone(),
        inputs_embeds=torch.zeros((1, input_ids.shape[1], 4)),
        video_spatio_temporal_features=features,
        tokenizer=tok,
    )

    assert float(out.loss) > 0
    patch_positions = (input_ids[0] == tok.convert_tokens_to_ids("<vid_patch>")).nonzero(as_tuple=False).flatten()
    assert torch.all(base.seen_inputs_embeds[0, patch_positions, :] == 5.0)


def test_xvars_model_projects_float_features_for_half_embeddings_without_dtype_mismatch():
    tok = TinyTokenizer()
    base = TinyHalfEmbeddingLM()
    model = XVarsVideoChatGPTCausalLM(base, mm_hidden_size=1024)
    assert model.mm_projector.weight.dtype == torch.float32

    with torch.no_grad():
        model.mm_projector.weight.fill_(0.0)
        model.mm_projector.bias.fill_(3.0)
    encoded = tok("USER: q <vid_start><vid_patch><vid_patch><vid_patch><vid_end> ASSISTANT: a", padding="max_length", max_length=16)
    input_ids = torch.tensor([encoded["input_ids"]])
    features = torch.ones((1, 3, 1024), dtype=torch.float32)

    out = model(input_ids=input_ids, labels=input_ids.clone(), video_spatio_temporal_features=features, tokenizer=tok)

    assert float(out.loss) > 0
    assert base.seen_inputs_embeds.dtype == torch.float16
    patch_positions = (input_ids[0] == tok.convert_tokens_to_ids("<vid_patch>")).nonzero(as_tuple=False).flatten()
    assert torch.all(base.seen_inputs_embeds[0, patch_positions, :] == torch.tensor(3.0, dtype=torch.float16))


def test_xvars_model_backward_with_video_features_avoids_in_place_leaf_error():
    tok = TinyTokenizer()
    model = XVarsVideoChatGPTCausalLM(TinyLM(), mm_hidden_size=1024)
    encoded = tok("USER: q <vid_start><vid_patch><vid_patch><vid_patch><vid_end> ASSISTANT: a", padding="max_length", max_length=16)
    input_ids = torch.tensor([encoded["input_ids"]])
    labels = input_ids.clone()
    features = torch.ones((1, 3, 1024))

    out = model(input_ids=input_ids, labels=labels, video_spatio_temporal_features=features, tokenizer=tok)
    loss = out.logits.sum() + out.loss
    loss.backward()

    assert model.mm_projector.weight.grad is not None


def test_xvars_model_prepare_inputs_for_generation_keeps_video_features():
    tok = TinyTokenizer()
    model = XVarsVideoChatGPTCausalLM(TinyLM(), mm_hidden_size=1024)
    input_ids = torch.tensor([[5, 6, 7]])
    attention_mask = torch.ones_like(input_ids)
    features = torch.ones((1, 3, 1024))

    first_step = model.prepare_inputs_for_generation(
        input_ids,
        attention_mask=attention_mask,
        inputs_embeds=torch.zeros((1, 3, 4)),
        video_spatio_temporal_features=features,
        tokenizer=tok,
        use_cache=True,
    )
    assert "inputs_embeds" in first_step
    assert "input_ids" not in first_step
    assert first_step["video_spatio_temporal_features"] is features
    assert first_step["tokenizer"] is tok
    assert first_step["use_cache"] is True

    next_step = model.prepare_inputs_for_generation(
        input_ids,
        past_key_values=(("cached",),),
        attention_mask=attention_mask,
        video_spatio_temporal_features=features,
        _xvars_tokenizer=tok,
    )
    assert next_step["input_ids"].tolist() == [[7]]
    assert next_step["video_spatio_temporal_features"] is features
    assert next_step["_xvars_tokenizer"] is tok


def test_xvars_model_generate_preserves_input_ids_with_inputs_embeds():
    tok = TinyTokenizer()
    base = TinyGenerateLM()
    model = XVarsVideoChatGPTCausalLM(base, mm_hidden_size=1024)
    encoded = tok("USER: q <vid_start><vid_patch><vid_patch><vid_patch><vid_end> ASSISTANT: a", padding="max_length", max_length=16)
    input_ids = torch.tensor([encoded["input_ids"]])
    attention_mask = torch.ones_like(input_ids)
    features = torch.ones((1, 3, 1024))

    output = model.generate(
        input_ids,
        tokenizer=tok,
        video_spatio_temporal_features=features,
        attention_mask=attention_mask,
        max_new_tokens=2,
    )

    assert output.tolist() == [[1, 2, 3, 4]]
    assert torch.equal(base.generate_kwargs["input_ids"], input_ids)
    assert base.generate_kwargs["inputs_embeds"].shape == (1, input_ids.shape[1], 4)
    assert torch.equal(base.generate_kwargs["attention_mask"], attention_mask)
    assert base.generate_kwargs["max_new_tokens"] == 2


def test_xvars_wrapper_delegates_generation_config_to_decoder():
    base = TinyGenerateLM()
    model = XVarsVideoChatGPTCausalLM(base, mm_hidden_size=1024)

    assert model.generation_config is base.generation_config
    replacement = SimpleNamespace(marker="replacement")
    model.generation_config = replacement

    assert model.generation_config is replacement
    assert base.generation_config is replacement


def test_xvars_peft_generate_uses_delegated_generation_config():
    peft = pytest.importorskip("peft")
    tok = TinyTokenizer()
    base = TinyGenerateLM()
    model = XVarsVideoChatGPTCausalLM(base, mm_hidden_size=1024)
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            task_type="CAUSAL_LM",
            target_modules=["mm_projector"],
        ),
    )
    encoded = tok(
        "USER: q <vid_start><vid_patch><vid_patch><vid_patch><vid_end> ASSISTANT: a",
        padding="max_length",
        max_length=16,
    )
    input_ids = torch.tensor([encoded["input_ids"]])

    output = model.generate(
        input_ids,
        tokenizer=tok,
        video_spatio_temporal_features=torch.ones((1, 3, 1024)),
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=2,
    )

    assert output.tolist() == [[1, 2, 3, 4]]
    assert model.generation_config is base.generation_config


def test_xvars_videochatgpt_lora_dry_run_marks_multimodal(tmp_path):
    out = VQAXVarsVideoChatGPTLoraTrainer(_cfg(tmp_path, dry_run=True)).train([_sample()], [_sample()])
    metadata = Path(out) / "training_metadata.json"
    assert metadata.exists()
    text = metadata.read_text(encoding="utf-8")
    assert '"backend": "xvars_videochatgpt_lora"' in text
    assert '"multimodal_training": true' in text


def test_vqa_xvars_prediction_export(tmp_path):
    from opensportslib.apis.vqa import VQAModel

    predictions = {
        "task": "vqa",
        "data": [
            {
                "id": "action_0",
                "question": "What card?",
                "answer_text": "Yellow card.",
                "video_path": "/tmp/action_0.mp4",
            }
        ],
    }
    rows = VQAModel._to_xvars_prediction_rows(predictions)
    assert rows == [{"id": "action_0", "video_name": "action_0", "Q": "What card?", "pred": "Yellow card."}]

    api = VQAModel.__new__(VQAModel)
    out_path = tmp_path / "xvars_predictions.json"
    saved = api.save_predictions(str(out_path), predictions, output_format="xvars")
    assert saved == str(out_path)
    assert out_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8")) == rows


def test_xvars_raw_num_frames_prefers_data_video_sampling():
    from opensportslib.models.base.xvars_videochatgpt import resolve_xvars_raw_num_frames

    cfg = SimpleNamespace(
        DATA=SimpleNamespace(inputs=SimpleNamespace(video=SimpleNamespace(sampling=SimpleNamespace(num_frames=77)))),
    )
    assert resolve_xvars_raw_num_frames(cfg, {"raw_num_frames": 13}) == 77


def test_xvars_raw_num_frames_falls_back_to_legacy_xvars_value():
    from opensportslib.models.base.xvars_videochatgpt import resolve_xvars_raw_num_frames

    cfg = SimpleNamespace(DATA=SimpleNamespace(inputs=SimpleNamespace(video=SimpleNamespace(sampling=SimpleNamespace()))))
    assert resolve_xvars_raw_num_frames(cfg, {"raw_num_frames": 13}) == 13


def test_xvars_dataset_and_model_prefer_canonical_vqa_fields(tmp_path):
    import pickle

    from opensportslib.datasets.vqa_dataset import VQADataset
    from opensportslib.models.base.xvars_videochatgpt import XVarsVideoChatGPTModel

    annotation_path = tmp_path / "train.json"
    annotation_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "action_0",
                        "inputs": [{"type": "video", "path": "clip.mp4"}],
                        "answers": [{"question": "What card?", "answers": ["Yellow"]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cfg = _cfg(tmp_path, dry_run=True)
    indexed_feature = tmp_path / "indexed.pkl"
    with indexed_feature.open("wb") as f:
        pickle.dump(torch.ones((300, 1024)).numpy(), f)
    feature_index = tmp_path / "feature_index.json"
    feature_index.write_text(
        json.dumps([{"id": "action_0", "feature_paths": [str(indexed_feature)]}]),
        encoding="utf-8",
    )
    cfg.DATA = SimpleNamespace(
        common=SimpleNamespace(
            feature_index=str(feature_index),
            prediction_index="",
            splits=SimpleNamespace(
                train=SimpleNamespace(
                    annotation_path=str(annotation_path),
                    source_path=str(tmp_path),
                    dataloader=SimpleNamespace(batch_size=1),
                )
            ),
        )
    )
    cfg.MODEL.components.video_encoder.params.feature_source = "raw_video"
    cfg.TRAIN.execution["xvars"] = {"feature_source": "indexed", "video_token_len": 9, "mm_hidden_size": 12}
    cfg.TRAIN.execution["prompt"]["video_token_len"] = 5
    cfg.MODEL.components.mm_projector.params.input_dim = 7

    dataset = VQADataset(cfg, split="train")
    assert dataset.feature_source == "raw_video"
    assert dataset[0]["video_spatio_temporal_features"] is None
    assert dataset[0]["selected_feature_path"] is None

    model = XVarsVideoChatGPTModel.__new__(XVarsVideoChatGPTModel)
    model.config = cfg
    model.video_token_len = 5
    model.feature_source = "raw_video"
    features = torch.ones((5, 7))
    assert tuple(model._features_for_sample({"video_spatio_temporal_features": features}, {"video_token_len": 5}).shape) == (5, 7)


def test_xvars_model_init_uses_explicit_feature_mode_token_len(monkeypatch, tmp_path):
    from opensportslib.models.base.xvars_videochatgpt import XVarsVideoChatGPTModel

    captured = {}

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

    class FakeWrappedModel:
        def to(self, device):
            del device
            return self

        def eval(self):
            return self

    cfg = _cfg(tmp_path, dry_run=True)
    cfg.MODEL.components.llm_decoder.params.repo_id = "base_model_videoChatGPT"
    cfg.TRAIN.execution["prompt"]["video_token_len"] = 300

    def _load_tok(model_id, **kwargs):
        del kwargs
        captured["tokenizer"] = model_id
        return FakeTokenizer()

    def _load_model(model_id, **kwargs):
        del kwargs
        captured["model"] = model_id
        return FakeWrappedModel()

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=_load_tok),
        ),
    )
    monkeypatch.setattr(
        "opensportslib.models.base.xvars_videochatgpt.load_videochatgpt_compatible_causal_lm",
        _load_model,
    )
    monkeypatch.setattr(
        "opensportslib.models.base.xvars_videochatgpt._ensure_video_special_tokens",
        lambda tokenizer, model=None: 0,
    )
    monkeypatch.setattr(
        "opensportslib.models.base.xvars_videochatgpt._configure_native_videochatgpt",
        lambda base_lm, tokenizer, model_id: True,
    )

    model = XVarsVideoChatGPTModel(cfg, model_id="base_model_videoChatGPT", projector_params={"input_dim": 1024})

    assert captured["model"] == "base_model_videoChatGPT"
    assert captured["tokenizer"] == "LLaVA-7B-Lightening-v1-1"
    assert model.native_generation is True
    assert model.video_token_len == 300


def test_xvars_prompt_places_prior_and_video_tokens_in_user_turn():
    prompt = build_xvars_prompt(
        system_prompt="System.",
        question="What card?",
        prior_text="a tackle, foul and a yellow card",
        video_token_len=2,
    )
    assert prompt == (
        "System. USER: What card? The prediction for this video is a tackle, foul and a yellow card\n"
        "<vid_start><vid_patch><vid_patch><vid_end> ASSISTANT:"
    )


def test_xvars_demo_token_ids_match_base_checkpoint():
    from transformers import AutoTokenizer
    from opensportslib.core.utils.hf_runtime import _ensure_video_special_tokens
    from opensportslib.models.base.xvars_videochatgpt import XVARS_BASE_TOKEN_IDS

    tokenizer = AutoTokenizer.from_pretrained(
        "/home/vorajv/xvars-weights/llava",
        use_fast=False,
        local_files_only=True,
    )
    _ensure_video_special_tokens(tokenizer)

    assert {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in XVARS_BASE_TOKEN_IDS
    } == XVARS_BASE_TOKEN_IDS


def test_xvars_base_inference_uses_native_videochatgpt_generate():
    from opensportslib.models.base.xvars_videochatgpt import XVarsVideoChatGPTModel

    captured = {}

    class NativeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(16, 4)
            self.mm_projector = torch.nn.Linear(1024, 4).half()

        def get_input_embeddings(self):
            return self.embeddings

        def generate(self, input_ids, **kwargs):
            captured["input_ids"] = input_ids.detach().clone()
            captured.update(kwargs)
            return torch.cat((input_ids, torch.tensor([[9]], device=input_ids.device)), dim=1)

    class NativeTokenizer:
        eos_token_id = 2

        def __call__(self, value, return_tensors=None):
            if isinstance(value, str):
                return SimpleNamespace(input_ids=[2])
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[1, 5, 6]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)}

        def batch_decode(self, values, skip_special_tokens=True):
            del values, skip_special_tokens
            return ["football answer</s>"]

    model = XVarsVideoChatGPTModel.__new__(XVarsVideoChatGPTModel)
    torch.nn.Module.__init__(model)
    model._ready = True
    model._error = None
    model.native_generation = True
    model.model = NativeModel()
    model.tokenizer = NativeTokenizer()
    model.feature_source = "indexed"
    model.feature_mode = "strict_xvars"
    model.video_token_len = 300
    model.baseline = SimpleNamespace(generate_answer=lambda *args, **kwargs: "fallback")

    answer = model.generate_answer(
        {
            "id": "action_0",
            "question": "What card?",
            "prior_prediction_text": "a tackle, foul and no card",
            "video_spatio_temporal_features": torch.ones((300, 1024)),
        },
        prompt_cfg={"video_token_len": 300, "include_priors": True},
        generation_cfg={"temperature": 0.2, "max_new_tokens": 16},
    )

    assert answer == "football answer"
    assert tuple(captured["video_spatio_temporal_features"].shape) == (1, 300, 1024)
    assert captured["video_spatio_temporal_features"].dtype == torch.float16
    assert captured["do_sample"] is False
    assert captured["temperature"] == 0.2
    assert captured["max_new_tokens"] == 16
    assert "inputs_embeds" not in captured
    assert "tokenizer" not in captured
    assert "eos_token_id" not in captured
    assert "pad_token_id" not in captured


def test_restore_native_projector_preserves_checkpoint_tensors(monkeypatch):
    from opensportslib.models.base import xvars_videochatgpt as mod

    raw_state = {
        "weight": torch.arange(12, dtype=torch.float16).reshape(3, 4),
        "bias": torch.arange(3, dtype=torch.float16),
    }

    class BaseModel:
        def __init__(self):
            self.model = SimpleNamespace(mm_projector=torch.nn.Linear(4, 3))

        def get_model(self):
            return self.model

    base = BaseModel()
    monkeypatch.setattr(mod, "_load_raw_mm_projector_state", lambda model: raw_state)

    assert mod._restore_native_mm_projector(base, torch.device("cpu")) is True
    assert torch.equal(base.model.mm_projector.weight.detach(), raw_state["weight"])
    assert torch.equal(base.model.mm_projector.bias.detach(), raw_state["bias"])


def test_offloaded_module_uses_accelerate_execution_device_not_meta():
    from opensportslib.models.base.xvars_videochatgpt import _module_execution_device

    module = torch.nn.Linear(4, 4, device="meta")
    module._hf_hook = SimpleNamespace(execution_device="cpu")

    assert _module_execution_device(module, torch.device("meta")) == torch.device("cpu")


def test_xvars_prompt_matches_upstream_videochatgpt_v1_system_prefix():
    prompt = build_xvars_prompt(
        system_prompt=VIDEO_CHATGPT_SYSTEM_PROMPT,
        question="What card?",
        prior_text="a shoulder challenge, foul, yellow card",
        video_token_len=1,
    )

    assert prompt == (
        f"{VIDEO_CHATGPT_SYSTEM_PROMPT} USER: What card? The prediction for this video is "
        "a shoulder challenge, foul, yellow card\n<vid_start><vid_patch><vid_end> ASSISTANT:"
    )


def test_xvars_truncation_preserves_supervised_eos():
    tokenizer = TinyTokenizer()
    row = {
        "prompt": "USER: question ASSISTANT:",
        "completion": "one two three four five six seven eight</s>",
    }

    tokenized = VQAXVarsVideoChatGPTSFTDataset._tokenize_row(
        row,
        tokenizer=tokenizer,
        max_seq_length=7,
    )

    assert tokenized is not None
    assert tokenized["input_ids"][-1] == tokenizer.eos_token_id
    assert tokenized["labels"][-1] == tokenizer.eos_token_id


def test_xvars_eos_remains_supervised_when_pad_and_eos_ids_match():
    tokenizer = TinyTokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    row = {"prompt": "USER: question ASSISTANT:", "completion": "answer</s>"}

    tokenized = VQAXVarsVideoChatGPTSFTDataset._tokenize_row(
        row,
        tokenizer=tokenizer,
        max_seq_length=12,
    )

    assert tokenized is not None
    supervised = [label for label in tokenized["labels"] if label != -100]
    assert supervised[-1] == tokenizer.eos_token_id


def test_xvars_generated_answer_relevance_score_rejects_code_domain_text():
    score = _score_xvars_generated_answers(
        ["It is a foul.", "Yellow card.", "No DOGSO.", "Advantage was possible."],
        required_terms=["foul", "card", "dogso", "advantage"],
        forbidden_terms=["get_children", "django"],
    )
    rejected = _score_xvars_generated_answers(
        ["Use get_children to inspect the node."],
        required_terms=["foul", "card"],
        forbidden_terms=["get_children"],
    )

    assert score["accepted"] is True
    assert rejected["accepted"] is False
    assert rejected["forbidden_count"] == 1


def test_xvars_indexes_are_split_aware(tmp_path):
    feature_index = tmp_path / "features.json"
    prediction_index = tmp_path / "predictions.json"
    feature_index.write_text(
        json.dumps(
            [
                {"id": "action_0", "split": "train", "feature_paths": ["train.pkl"]},
                {"id": "action_0", "split": "test", "feature_paths": ["test.pkl"]},
            ]
        ),
        encoding="utf-8",
    )
    prediction_index.write_text(
        json.dumps(
            [
                {"id": "action_0", "split": "train", "Action class": "Challenge"},
                {"id": "action_0", "split": "test", "Action class": "Tackling"},
            ]
        ),
        encoding="utf-8",
    )

    assert load_feature_index(str(feature_index), split="train")["action_0"][0].endswith("train.pkl")
    assert load_prediction_index(str(prediction_index), split="train")["action_0"]["Action class"] == "Challenge"

def test_xvars_model_init_uses_quantized_device_map_for_inference(monkeypatch, tmp_path):
    from opensportslib.models.base import xvars_videochatgpt as mod
    from opensportslib.models.base.xvars_videochatgpt import XVarsVideoChatGPTModel

    captured = {}
    bnb_config = object()

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

    class FakeProjector:
        def to(self, device):
            captured["projector_device"] = str(device)
            return self

    class FakeWrappedModel:
        def __init__(self):
            self.mm_projector = FakeProjector()

        def to(self, device):
            captured["wrapper_to"] = str(device)
            return self

        def eval(self):
            captured["eval"] = True
            return self

    cfg = _cfg(tmp_path, dry_run=True)
    cfg.TRAIN.execution["hf"] = {
        "local_files_only": True,
        "prefer_cuda": True,
        "cuda_device_index": 1,
        "tokenizer_id": "/tmp/tokenizer",
    }
    cfg.TRAIN.execution["quantization"] = {"enabled": True, "load_in_4bit": True}

    def _load_tok(model_id, **kwargs):
        captured["tokenizer"] = model_id
        captured["tokenizer_kwargs"] = kwargs
        return FakeTokenizer()

    def _load_model(model_id, **kwargs):
        captured["model"] = model_id
        captured["model_kwargs"] = kwargs
        return FakeWrappedModel()

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=_load_tok),
        ),
    )
    monkeypatch.setattr(mod, "build_bitsandbytes_config", lambda quant_cfg: bnb_config)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mod.torch.cuda, "set_device", lambda idx: captured.__setitem__("set_device", idx))
    monkeypatch.setattr(mod.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(mod, "load_videochatgpt_compatible_causal_lm", _load_model)
    monkeypatch.setattr(mod, "_ensure_video_special_tokens", lambda tokenizer, model=None: 0)
    monkeypatch.setattr(mod, "_configure_native_videochatgpt", lambda base_lm, tokenizer, model_id: True)
    monkeypatch.setattr(
        mod,
        "_restore_native_mm_projector",
        lambda base_lm, device: captured.__setitem__("projector_device", str(device)) or True,
    )

    model = XVarsVideoChatGPTModel(cfg, model_id="quantized_xvars", projector_params={"input_dim": 1024})

    assert model._ready is True
    assert captured["tokenizer"] == "/tmp/tokenizer"
    assert captured["set_device"] == 1
    assert captured["model"] == "quantized_xvars"
    assert captured["model_kwargs"]["local_files_only"] is True
    assert captured["model_kwargs"]["quantization_config"] is bnb_config
    assert captured["model_kwargs"]["device_map"] == {"": 1}
    assert captured["projector_device"] == "cuda:1"
    assert "wrapper_to" not in captured
    assert captured["eval"] is True
    assert model.native_generation is True


def test_xvars_dataset_rejects_feature_mode_shape_mismatch(tmp_path):
    from opensportslib.datasets.vqa_dataset import VQADataset

    annotation_path = tmp_path / "train.json"
    annotation_path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "action_0",
                        "inputs": [{"type": "video", "path": "clip.mp4"}],
                        "answers": [{"question": "What card?", "answers": ["Yellow"]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    feature_dir = tmp_path / "features" / "action_0"
    feature_dir.mkdir(parents=True, exist_ok=True)
    bad_features = torch.ones((356, 1024), dtype=torch.float32).numpy()
    with (feature_dir / "PRE_CLIP_feature_clip_1.pkl").open("wb") as f:
        import pickle

        pickle.dump(bad_features, f)
    feature_index = tmp_path / "feature_index.json"
    feature_index.write_text(json.dumps([{"id": "action_0", "feature_dir": str(feature_dir)}]), encoding="utf-8")

    cfg = _cfg(tmp_path, dry_run=True)
    cfg.DATA = SimpleNamespace(
        common=SimpleNamespace(
            feature_index=str(feature_index),
            prediction_index="",
            splits=SimpleNamespace(
                train=SimpleNamespace(
                    annotation_path=str(annotation_path),
                    source_path=str(tmp_path),
                    dataloader=SimpleNamespace(batch_size=1),
                )
            ),
        )
    )

    dataset = VQADataset(cfg, split="train")
    with pytest.raises(ValueError, match="token count mismatch"):
        dataset[0]


def test_videochatgpt_loader_raises_clear_xvars_error(monkeypatch, tmp_path):
    from opensportslib.models.base import video_chatgpt_compat as compat

    ckpt_dir = tmp_path / "videochatgpt_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text('{"model_type": "VideoChatGPT"}', encoding="utf-8")

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            del model_id, kwargs
            raise ValueError("Transformers does not recognize this architecture")

    monkeypatch.setattr(compat, "ensure_videochatgpt_registered", lambda: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM),
    )

    with pytest.raises(ValueError, match="root cause"):
        compat.load_videochatgpt_compatible_causal_lm(str(ckpt_dir), local_files_only=True)
