"""Minimal local compatibility layer for original X-VARS VideoChatGPT checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


def _read_local_model_type(model_id: str) -> str | None:
    config_path = Path(os.path.expanduser(str(model_id))) / "config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model_type = payload.get("model_type")
    return str(model_type) if model_type is not None else None


class VisionConfig:
    def __init__(self):
        self.frame_size = 224
        self.patch_size = 14
        self.hidden_size = 1024
        self.use_vid_start_end = None
        self.vid_start_token = None
        self.vid_end_token = None
        self.vid_patch_token = None


def ensure_videochatgpt_registered() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, LlamaModel
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

    class VideoChatGPTConfig(LlamaConfig):
        model_type = "VideoChatGPT"

    class VideoChatGPTLlamaModel(LlamaModel):
        config_class = VideoChatGPTConfig

        def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
            del mm_vision_tower, mm_hidden_size
            super().__init__(config)
            if hasattr(config, "mm_vision_tower"):
                self.vision_config = VisionConfig()
            if hasattr(config, "use_mm_proj"):
                self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

        def initialize_vision_modules(self, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
            del pretrain_mm_mlp_adapter, tune_mm_mlp_adapter
            vision_config = self.vision_config
            num_patches = (vision_config.frame_size // vision_config.patch_size) ** 2
            self.config.use_mm_proj = True
            self.config.mm_hidden_size = vision_config.hidden_size
            return {"video_token_len": num_patches, "vision_config": vision_config}

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
        ) -> BaseModelOutputWithPast:
            orig_embeds_params = getattr(self, "orig_embeds_params", None)
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if (input_ids is not None and (input_ids.shape[1] != 1 or self.training)) and video_spatio_temporal_features is not None:
                video_features = self.mm_projector(video_spatio_temporal_features)
                dummy_video_features = torch.zeros(
                    video_features.shape[1],
                    1024,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                dummy_video_features = self.mm_projector(dummy_video_features)
                new_input_embeds = []
                cur_video_idx = 0
                for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                    if (cur_input_ids == self.vision_config.vid_patch_token).sum() == 0:
                        cur_input_embeds = cur_input_embeds + (0.0 * dummy_video_features).sum()
                        new_input_embeds.append(cur_input_embeds)
                        cur_video_idx += 1
                        continue
                    if self.vision_config.use_vid_start_end:
                        video_start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0]
                        for video_start_token_pos in video_start_tokens:
                            cur_video_features = video_features[cur_video_idx].to(device=cur_input_embeds.device)
                            num_patches = cur_video_features.shape[0]
                            if cur_input_ids[video_start_token_pos + num_patches + 1] != self.vision_config.vid_end_token:
                                raise ValueError("The video end token should follow the video start token.")
                            if orig_embeds_params is not None:
                                cur_new_input_embeds = torch.cat(
                                    (
                                        cur_input_embeds[:video_start_token_pos].detach(),
                                        cur_input_embeds[video_start_token_pos : video_start_token_pos + 1],
                                        cur_video_features,
                                        cur_input_embeds[
                                            video_start_token_pos + num_patches + 1 : video_start_token_pos + num_patches + 2
                                        ],
                                        cur_input_embeds[video_start_token_pos + num_patches + 2 :].detach(),
                                    ),
                                    dim=0,
                                )
                            else:
                                cur_new_input_embeds = torch.cat(
                                    (
                                        cur_input_embeds[: video_start_token_pos + 1],
                                        cur_video_features,
                                        cur_input_embeds[video_start_token_pos + num_patches + 1 :],
                                    ),
                                    dim=0,
                                )
                            cur_video_idx += 1
                        new_input_embeds.append(cur_new_input_embeds)
                    else:
                        cur_video_features = video_features[cur_video_idx]
                        num_patches = cur_video_features.shape[0]
                        masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0]
                        mask_index_start = masked_indices[0]
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:mask_index_start].detach(),
                                    cur_video_features,
                                    cur_input_embeds[mask_index_start + num_patches :].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:mask_index_start],
                                    cur_video_features,
                                    cur_input_embeds[mask_index_start + num_patches :],
                                ),
                                dim=0,
                            )
                        new_input_embeds.append(cur_new_input_embeds)
                        cur_video_idx += 1
                inputs_embeds = torch.stack(new_input_embeds, dim=0)

            return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    class VideoChatGPTLlamaForCausalLM(LlamaForCausalLM):
        config_class = VideoChatGPTConfig

        def __init__(self, config):
            super(LlamaForCausalLM, self).__init__(config)
            self.model = VideoChatGPTLlamaModel(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        def get_model(self):
            return self.model

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
        ) -> CausalLMOutputWithPast:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            return_dict = return_dict if return_dict is not None else bool(getattr(self.config, "return_dict", True))
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                video_spatio_temporal_features=video_spatio_temporal_features,
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
                shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
                loss = CrossEntropyLoss()(shift_logits, shift_labels)
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
            if past_key_values:
                input_ids = input_ids[:, -1:]
            model_inputs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None and past_key_values is None else {"input_ids": input_ids}
            model_inputs.update(
                {
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                    "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
                }
            )
            return model_inputs

        def initialize_vision_tokenizer(self, mm_use_vid_start_end, tokenizer, device, tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
            del pretrain_mm_mlp_adapter
            vision_config = self.get_model().vision_config
            vision_config.use_vid_start_end = mm_use_vid_start_end
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            if mm_use_vid_start_end:
                num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
                self.resize_token_embeddings(len(tokenizer))
                vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
                    [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN]
                )
                if num_new_tokens > 0:
                    input_embeddings = self.get_input_embeddings().weight.data
                    output_embeddings = self.get_output_embeddings().weight.data
                    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                    input_embeddings[-num_new_tokens:] = input_embeddings_avg
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg
                if tune_mm_mlp_adapter:
                    self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                    for param in self.get_input_embeddings().parameters():
                        param.requires_grad = True
                    for param in self.get_output_embeddings().parameters():
                        param.requires_grad = False
            vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]

    try:
        AutoConfig.register("VideoChatGPT", VideoChatGPTConfig)
    except Exception:
        pass
    try:
        AutoModelForCausalLM.register(VideoChatGPTConfig, VideoChatGPTLlamaForCausalLM)
    except Exception:
        pass


def load_videochatgpt_compatible_causal_lm(model_id: str, **model_kwargs):
    ensure_videochatgpt_registered()
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:
        if _read_local_model_type(model_id) == "VideoChatGPT":
            raise ValueError(
                f"Configured X-VARS decoder checkpoint at '{model_id}' declares model_type 'VideoChatGPT', "
                "but OpenSportsLib could not load the local compatibility class. "
                "This is the root cause of any downstream torch.multiprocessing.spawn termination."
            ) from exc
        raise


__all__ = [
    "DEFAULT_VIDEO_PATCH_TOKEN",
    "DEFAULT_VID_END_TOKEN",
    "DEFAULT_VID_START_TOKEN",
    "ensure_videochatgpt_registered",
    "load_videochatgpt_compatible_causal_lm",
]
