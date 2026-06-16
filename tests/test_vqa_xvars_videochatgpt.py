import json
from pathlib import Path
from types import SimpleNamespace

import torch

from opensportslib.core.trainer.vqa_trainer import (
    VQAXVarsVideoChatGPTLoraTrainer,
    VQAXVarsVideoChatGPTSFTDataset,
    XVarsVideoChatGPTDataCollator,
    _resolve_sft_per_device_batch_sizes,
)
from opensportslib.models.base.xvars_videochatgpt import XVarsVideoChatGPTCausalLM


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

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
            for tok in ("<vid_start>", "<vid_patch>", "<vid_end>"):
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
            while j < len(text) and not text[j].isspace() and not any(text.startswith(t, j) for t in ("<vid_start>", "<vid_patch>", "<vid_end>")):
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

    def get_input_embeddings(self):
        return self.emb

    def resize_token_embeddings(self, size):
        return self.emb

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
        TRAIN=SimpleNamespace(
            execution={
                "training_backend": "xvars_videochatgpt_lora",
                "dry_run": dry_run,
                "prompt": {"include_priors": True},
                "sft": {"include_video_tokens": True, "video_token_len": 3, "max_seq_length": 64},
                "xvars": {"base_model": "tiny", "video_token_len": 3, "feature_source": "indexed"},
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
        prompt_cfg={"include_priors": True},
        sft_cfg={"video_token_len": 3, "max_seq_length": 64},
        xvars_cfg={"video_token_len": 3},
    )
    batch = XVarsVideoChatGPTDataCollator(tok)([ds[0]])
    assert tuple(batch["video_spatio_temporal_features"].shape) == (1, 3, 1024)
    assert batch["labels"].shape == batch["input_ids"].shape


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
