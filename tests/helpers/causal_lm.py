"""Small tokenizer and causal model doubles for offline generation tests."""

from types import SimpleNamespace

import torch


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
