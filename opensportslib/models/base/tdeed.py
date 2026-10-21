"""T-DEED localization model, with the released displacement and mixup losses."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from opensportslib.models.utils.common import BaseRGBModel
from opensportslib.models.utils.impl.tdeed.network import TDEEDNetwork


def relocate_scores(logits, displacement=None):
    """Relocate softmax scores by rounded offsets, taking classwise maxima.

    This deliberately does not re-normalize relocated scores: multiple source
    frames can vote for one destination, while other destinations receive none.
    """
    scores = logits.softmax(-1)
    if displacement is None:
        return scores
    if displacement.shape != logits.shape[:2] or not torch.isfinite(displacement).all():
        raise ValueError("Displacement must be finite and have shape (batch, time)")
    index = (
        torch.arange(logits.shape[1], device=logits.device)[None]
        - displacement.round().long()
    )
    index = index.clamp(0, logits.shape[1] - 1).unsqueeze(-1).expand_as(scores)
    return torch.zeros_like(scores).scatter_reduce(
        1, index, scores, reduce="amax", include_self=True
    )


class TDEEDModel(BaseRGBModel):
    """Wrapper compatible with OSL's model construction and checkpoint APIs.

    Inputs are uint8-range RGB (B,T,C,H,W). Image augmentation, cropping and
    ImageNet normalization belong to the network, matching the reference code.
    """

    def __init__(self, args, device="cpu"):
        self.args = (
            args if isinstance(args, SimpleNamespace) else SimpleNamespace(**args)
        )
        self.device = torch.device(device)
        self._model = TDEEDNetwork(self.args).to(self.device)
        self._num_classes = self.args.num_classes + 1
        auxiliary = getattr(self.args, "auxiliary_num_classes", 0)
        if auxiliary:
            # Keep reference checkpoint keys, but avoid its hard-coded .cuda().
            from opensportslib.models.utils.impl.tdeed.modules import FC2Layers

            self._model._pred_fine = FC2Layers(
                self._model._feat_dim, [self._num_classes, auxiliary + 1]
            ).to(self.device)
            self._model._double_head = True

    def get_optimizer(self, opt_args):
        return torch.optim.AdamW(self._get_params(), **opt_args), torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda"
        )

    def loss(self, batch, *, training=True, fg_weight=5, return_outputs=False):
        frame = batch["frame"].to(self.device).float()
        label = batch["label"].to(self.device)
        target_d = batch.get("labelD")
        target_d = target_d.to(self.device).float() if target_d is not None else None
        head_ids = batch.get("dataset")
        widths = [self._num_classes]
        if self._model._double_head:
            widths.append(self.args.auxiliary_num_classes + 1)
            if head_ids is None:
                raise ValueError("Joint training requires dataset IDs 1 or 2")
        total = sum(widths)
        offsets = torch.zeros(len(label), device=self.device, dtype=torch.long)
        if len(widths) == 2:
            head_ids = head_ids.to(self.device)
            if not ((head_ids == 1) | (head_ids == 2)).all():
                raise ValueError("Joint dataset IDs must be 1 or 2")
            offsets[head_ids == 2] = widths[0]
            label = label + offsets[:, None]
        if training and "frame2" in batch:
            import random

            weight = frame.new_tensor([random.betavariate(0.2, 0.2) for _ in frame])
            frame = (
                weight[:, None, None, None, None] * frame
                + (1 - weight[:, None, None, None, None])
                * batch["frame2"].to(self.device).float()
            )
            second = batch["label2"].to(self.device) + offsets[:, None]
            label = F.one_hot(label, total) * weight[:, None, None] + F.one_hot(
                second, total
            ) * (1 - weight[:, None, None])
            if target_d is not None:
                target_d = weight[:, None] * target_d + (1 - weight[:, None]) * batch[
                    "labelD2"
                ].to(self.device)
        output, _ = self._model(frame, inference=not training)
        logits = output["im_feat"] if isinstance(output, dict) else output
        if len(widths) == 1:
            weights = logits.new_tensor([1] + [fg_weight] * (widths[0] - 1))
            target = label.flatten() if label.ndim == 2 else label.reshape(-1, total)
            loss = F.cross_entropy(logits.reshape(-1, total), target, weight=weights)
        else:
            loss = logits.sum() * 0
            for i, head in enumerate(head_ids.tolist()):
                start = 0 if head == 1 else widths[0]
                stop = start + widths[head - 1]
                target = (
                    label[i] - start if label.ndim == 2 else label[i, :, start:stop]
                )
                weights = logits.new_tensor([1] + [fg_weight] * (stop - start - 1))
                loss = loss + F.cross_entropy(
                    logits[i, :, start:stop], target, weight=weights
                ) / len(label)
        if target_d is not None:
            loss = loss + F.mse_loss(output["displ_feat"], target_d)
        return (loss, output) if return_outputs else loss

    def epoch(
        self,
        loader,
        dali=False,
        optimizer=None,
        scaler=None,
        lr_scheduler=None,
        acc_grad_iter=1,
        fg_weight=5,
        progress_callback=None,
        collect_predictions=False,
    ):
        if dali:
            raise ValueError(
                "T-DEED requires its raw RGB loader; DALI preprocessing is incompatible"
            )
        training = optimizer is not None
        self._model.train(training)
        if training:
            optimizer.zero_grad(set_to_none=True)
        if not len(loader):
            raise ValueError("Empty T-DEED loader")
        total = 0.0
        collected_labels, collected_scores = [], []
        with nullcontext() if training else torch.no_grad():
            for i, batch in enumerate(loader):
                amp = (
                    torch.autocast("cuda")
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with amp:
                    result = self.loss(
                        batch,
                        training=training,
                        fg_weight=fg_weight,
                        return_outputs=collect_predictions,
                    )
                    if collect_predictions:
                        loss, output = result
                        if training or not isinstance(output, dict):
                            raise ValueError(
                                "Clip validation collection requires displacement validation outputs"
                            )
                        labels = batch["label"].to(self.device)
                        destination = (
                            torch.arange(labels.shape[1], device=self.device)[None]
                            - batch["labelD"].to(self.device).long()
                        )
                        valid = (
                            (labels != 0)
                            & (destination >= 0)
                            & (destination < labels.shape[1])
                        )
                        restored = output["im_feat"].new_zeros(output["im_feat"].shape)
                        restored[..., 0] = 1
                        b, t = valid.nonzero(as_tuple=True)
                        restored[b, destination[b, t], labels[b, t]] = 1
                        restored[b, destination[b, t], 0] = 0
                        collected_labels.append(restored.float().cpu())
                        collected_scores.append(
                            relocate_scores(output["im_feat"], output["displ_feat"])
                            .float()
                            .cpu()
                        )
                    else:
                        loss = result
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite T-DEED training loss")
                if training:
                    # Correctly flush a final partial accumulation group.
                    group = min(
                        acc_grad_iter,
                        len(loader) - (i // acc_grad_iter) * acc_grad_iter,
                    )
                    scaled = loss / group
                    (scaler.scale(scaled) if scaler is not None else scaled).backward()
                    if (i + 1) % acc_grad_iter == 0 or i + 1 == len(loader):
                        if scaler is None:
                            optimizer.step()
                        else:
                            scaler.step(optimizer)
                            scaler.update()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                total += loss.detach().item()
                if progress_callback is not None and (
                    i == 0 or (i + 1) % 25 == 0 or i + 1 == len(loader)
                ):
                    progress_callback(i + 1, len(loader), total / (i + 1))
        if collect_predictions:
            return (
                total / len(loader),
                torch.cat(collected_labels),
                torch.cat(collected_scores),
            )
        return total / len(loader)

    def predict(self, seq, use_amp=True, augment_inference=False):
        seq = torch.as_tensor(seq, device=self.device).float()
        if seq.ndim == 4:
            seq = seq.unsqueeze(0)
        self._model.eval()
        with (
            torch.no_grad(),
            (
                torch.autocast("cuda")
                if use_amp and self.device.type == "cuda"
                else nullcontext()
            ),
        ):
            output, _ = self._model(
                seq, inference=True, augment_inference=augment_inference
            )
            if isinstance(output, dict):
                logits = output["im_feat"][..., : self._num_classes]
                scores = relocate_scores(logits, output["displ_feat"])
            else:
                scores = relocate_scores(output[..., : self._num_classes])
        return scores.argmax(-1).cpu().numpy(), scores.float().cpu().numpy()


def build_tdeed(config, device):
    from opensportslib.core.config.accessors import (
        get_component_params_by_kind,
        get_data_num_classes,
        get_data_sampling,
    )

    params = dict(get_component_params_by_kind(config, "head"))
    params.update(get_component_params_by_kind(config, "encoder"))
    params.pop("type", None)
    params.setdefault("clip_len", get_data_sampling(config)["clip_len"])
    params.setdefault("num_classes", get_data_num_classes(config))
    return TDEEDModel(params, device)
