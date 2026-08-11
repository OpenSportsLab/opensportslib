"""SpoTTA test-time adaptation for binary E2E action spotting.

This module contains only the effective SpoTTA recipe used by the historical
Header experiment.  It intentionally does not expose inactive options from the
research CLI (frame filtering, action weighting, reset scheduling, or repeated
``steps``).  A :class:`SpoTTA` instance represents one continuous target-set
adaptation session; the E2ESpot API creates a fresh instance for every
``LocalizationModel.infer()`` call.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

try:
    from timm.layers import BatchNormAct2d
except ImportError:  # pragma: no cover - timm is an install dependency
    BatchNormAct2d = ()  # type: ignore[assignment]


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return vars(value)


def _require_recipe_value(
    mapping: dict[str, Any], key: str, expected: Any, path: str
) -> None:
    if key in mapping and mapping[key] != expected:
        raise ValueError(
            f"The 53.33 SpoTTA recipe requires {path}.{key}={expected!r}, "
            f"got {mapping[key]!r}."
        )


@dataclass(frozen=True)
class SpoTTAConfig:
    """Effective hyperparameters for the 53.33-mAP SpoTTA recipe."""

    alpha: float = 0.05
    tether_cap: float = 0.5
    gate_threshold: float = 0.3
    action_class_index: int = 1
    min_action_frames: int = 1
    memory_capacity: int = 8
    update_frequency: int = 2
    lambda_t: float = 1.0
    lambda_u: float = 1.0
    learning_rate: float = 1e-3
    adam_beta: float = 0.9
    ema_nu: float = 1e-3
    max_ema_nu: float = 0.02
    drift_scale: float = 10.0
    drift_threshold: float = 1.0
    drift_gamma: float = 0.2
    augmentation: bool = True

    @classmethod
    def from_mapping(cls, value: Any) -> "SpoTTAConfig":
        root = _mapping(value)
        robust_bn = _mapping(root.get("robust_bn"))
        tether = _mapping(robust_bn.get("tether"))
        gate = _mapping(root.get("confidence_gate"))
        memory = _mapping(root.get("memory"))
        optimizer = _mapping(root.get("optimizer"))
        teacher = _mapping(root.get("teacher"))
        augmentation = _mapping(root.get("augmentation"))

        _require_recipe_value(
            root,
            "prediction_timing",
            "adapt_then_predict",
            "adaptation",
        )
        _require_recipe_value(tether, "mode", "bayesian", "robust_bn.tether")
        _require_recipe_value(
            gate,
            "uncertainty",
            "one_minus_max_probability",
            "confidence_gate",
        )
        _require_recipe_value(
            gate,
            "aggregation",
            "min_over_predicted_action_frames",
            "confidence_gate",
        )
        _require_recipe_value(memory, "class_policy", "header_only", "memory")
        _require_recipe_value(optimizer, "type", "Adam", "optimizer")
        _require_recipe_value(
            optimizer,
            "trainable_parameters",
            "batch_norm_affine_only",
            "optimizer",
        )
        _require_recipe_value(teacher, "type", "ema", "teacher")
        _require_recipe_value(
            teacher, "adaptive_from_bn_drift", True, "teacher"
        )
        _require_recipe_value(
            augmentation, "mode", "framewise_rotta_strong", "augmentation"
        )

        config = cls(
            alpha=float(robust_bn.get("alpha", cls.alpha)),
            tether_cap=float(tether.get("cap", cls.tether_cap)),
            gate_threshold=float(gate.get("threshold", cls.gate_threshold)),
            action_class_index=int(
                gate.get("action_class_index", cls.action_class_index)
            ),
            min_action_frames=int(
                gate.get("min_action_frames", cls.min_action_frames)
            ),
            memory_capacity=int(memory.get("capacity", cls.memory_capacity)),
            update_frequency=int(
                memory.get("update_frequency", cls.update_frequency)
            ),
            lambda_t=float(memory.get("lambda_t", cls.lambda_t)),
            lambda_u=float(memory.get("lambda_u", cls.lambda_u)),
            learning_rate=float(
                optimizer.get("learning_rate", cls.learning_rate)
            ),
            adam_beta=float(optimizer.get("beta", cls.adam_beta)),
            ema_nu=float(teacher.get("base_nu", cls.ema_nu)),
            max_ema_nu=float(teacher.get("max_nu", cls.max_ema_nu)),
            drift_scale=float(teacher.get("drift_scale", cls.drift_scale)),
            drift_threshold=float(
                teacher.get("drift_threshold", cls.drift_threshold)
            ),
            drift_gamma=float(teacher.get("drift_gamma", cls.drift_gamma)),
            augmentation=bool(augmentation.get("enabled", cls.augmentation)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("SpoTTA robust_bn.alpha must be in (0, 1].")
        if not 0 < self.tether_cap <= 1:
            raise ValueError("SpoTTA robust_bn.tether.cap must be in (0, 1].")
        if not 0 <= self.gate_threshold <= 1:
            raise ValueError("SpoTTA confidence_gate.threshold must be in [0, 1].")
        if self.action_class_index != 1:
            raise ValueError(
                "The E2ESpot SpoTTA recipe currently requires Header at class index 1."
            )
        if self.min_action_frames < 1:
            raise ValueError("SpoTTA min_action_frames must be positive.")
        if self.memory_capacity < 1 or self.update_frequency < 1:
            raise ValueError("SpoTTA memory capacity and update frequency must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("SpoTTA optimizer learning_rate must be positive.")
        if not 0 <= self.adam_beta < 1:
            raise ValueError("SpoTTA optimizer beta must be in [0, 1).")
        if not 0 <= self.ema_nu <= self.max_ema_nu <= 1:
            raise ValueError("SpoTTA EMA rates must satisfy 0 <= base_nu <= max_nu <= 1.")
        if not 0 < self.drift_gamma <= 1:
            raise ValueError("SpoTTA teacher drift_gamma must be in (0, 1].")


def _symmetric_kl_diagonal(
    mean: torch.Tensor,
    var: torch.Tensor,
    anchor_mean: torch.Tensor,
    anchor_var: torch.Tensor,
    eps: float,
) -> float:
    """Symmetric KL between two diagonal Gaussian distributions."""

    var = var + eps
    anchor_var = anchor_var + eps
    mean_delta_sq = (mean - anchor_mean) ** 2
    target_source = 0.5 * (
        torch.log(anchor_var / var) + (var + mean_delta_sq) / anchor_var - 1.0
    )
    source_target = 0.5 * (
        torch.log(var / anchor_var) + (anchor_var + mean_delta_sq) / var - 1.0
    )
    return (0.5 * target_source + 0.5 * source_target).sum().item()


class RobustBatchNorm(nn.Module):
    """BatchNorm with conservative target-stat updates and a source tether."""

    def __init__(self, source: nn.modules.batchnorm._BatchNorm, alpha: float):
        super().__init__()
        if not source.track_running_stats:
            raise ValueError("SpoTTA requires BatchNorm layers with tracked running stats.")
        if source.running_mean is None or source.running_var is None:
            raise ValueError("SpoTTA requires initialized BatchNorm running statistics.")

        self.num_features = source.num_features
        self.alpha = alpha
        self.eps = source.eps
        self.weight = deepcopy(source.weight)
        self.bias = deepcopy(source.bias)
        self.register_buffer("source_mean", source.running_mean.detach().clone())
        self.register_buffer("source_var", source.running_var.detach().clone())
        self.register_buffer("anchor_mean", source.running_mean.detach().clone())
        self.register_buffer("anchor_var", source.running_var.detach().clone())
        self.tether_rho = 0.0
        self.anchor_kl = 0.0
        self.last_batch_drift = 0.0

    def _statistics_dims(self, x: torch.Tensor) -> tuple[int, ...]:
        return (0, *range(2, x.ndim))

    def _post_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        view_shape = (1, -1, *([1] * (x.ndim - 2)))
        if self.training:
            batch_var, batch_mean = torch.var_mean(
                x, dim=self._statistics_dims(x), unbiased=False, keepdim=False
            )
            source_var_safe = self.source_var + self.eps
            batch_var_safe = batch_var + self.eps
            drift = (
                torch.log(source_var_safe.sqrt() / batch_var_safe.sqrt())
                + (
                    batch_var_safe
                    + (batch_mean - self.source_mean) ** 2
                )
                / (2 * source_var_safe)
                - 0.5
            )
            self.last_batch_drift = drift.mean().item()

            mean = (1 - self.alpha) * self.source_mean + self.alpha * batch_mean
            var = (1 - self.alpha) * self.source_var + self.alpha * batch_var
            self.anchor_kl = _symmetric_kl_diagonal(
                mean, var, self.anchor_mean, self.anchor_var, self.eps
            )
            mean = (1 - self.tether_rho) * mean + self.tether_rho * self.anchor_mean
            var = (1 - self.tether_rho) * var + self.tether_rho * self.anchor_var
            self.source_mean.copy_(mean.detach())
            self.source_var.copy_(var.detach())
        else:
            mean, var = self.source_mean, self.source_var

        output = (x - mean.view(view_shape)) / torch.sqrt(
            var.view(view_shape) + self.eps
        )
        if self.weight is not None:
            output = output * self.weight.view(view_shape)
        if self.bias is not None:
            output = output + self.bias.view(view_shape)
        return self._post_normalize(output)


class RobustBatchNorm1d(RobustBatchNorm):
    pass


class RobustBatchNorm2d(RobustBatchNorm):
    pass


class RobustBatchNorm3d(RobustBatchNorm):
    pass


class RobustBatchNormAct2d(RobustBatchNorm2d):
    """RobustBN replacement that preserves timm activation and dropout."""

    def __init__(self, source: nn.Module, alpha: float):
        super().__init__(source, alpha)  # type: ignore[arg-type]
        self.drop = deepcopy(getattr(source, "drop", nn.Identity()))
        self.act = deepcopy(getattr(source, "act", nn.Identity()))

    def _post_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.drop(x))


def _set_submodule(model: nn.Module, name: str, value: nn.Module) -> None:
    parent = model
    path = name.split(".")
    for part in path[:-1]:
        parent = getattr(parent, part)
    setattr(parent, path[-1], value)


def replace_batch_norm(
    model: nn.Module, alpha: float
) -> list[RobustBatchNorm]:
    """Replace all supported BatchNorm layers and return the replacements."""

    replacements: list[tuple[str, RobustBatchNorm]] = []
    for name, module in list(model.named_modules()):
        if not name:
            continue
        if BatchNormAct2d and isinstance(module, BatchNormAct2d):
            replacement = RobustBatchNormAct2d(module, alpha)
        elif isinstance(module, nn.BatchNorm1d):
            replacement = RobustBatchNorm1d(module, alpha)
        elif isinstance(module, nn.BatchNorm2d):
            replacement = RobustBatchNorm2d(module, alpha)
        elif isinstance(module, nn.BatchNorm3d):
            replacement = RobustBatchNorm3d(module, alpha)
        else:
            continue
        replacements.append((name, replacement))

    for name, replacement in replacements:
        _set_submodule(model, name, replacement)
    return [replacement for _, replacement in replacements]


@dataclass
class _MemoryItem:
    data: torch.Tensor
    uncertainty: float
    age: int = 0


class HeaderCSTUMemory:
    """Header-only CSTU memory used by the effective SpoTTA recipe."""

    def __init__(self, capacity: int, lambda_t: float, lambda_u: float):
        self.capacity = capacity
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.items: list[_MemoryItem] = []

    def _score(self, age: int, uncertainty: float) -> float:
        timeliness = self.lambda_t / (1 + math.exp(-age / self.capacity))
        uncertainty_score = self.lambda_u * uncertainty / math.log(2)
        return timeliness + uncertainty_score

    def add(self, data: torch.Tensor, uncertainty: float) -> bool:
        new_item = _MemoryItem(data=data.detach().cpu(), uncertainty=uncertainty)
        added = False
        if len(self.items) < self.capacity:
            self.items.append(new_item)
            added = True
        else:
            scores = [self._score(item.age, item.uncertainty) for item in self.items]
            evict_index = max(range(len(scores)), key=lambda index: (scores[index], index))
            if scores[evict_index] > self._score(0, uncertainty):
                self.items.pop(evict_index)
                self.items.append(new_item)
                added = True

        for item in self.items:
            item.age += 1
        return added

    def batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        clips = torch.stack([item.data for item in self.items]).to(device)
        normalized_ages = torch.tensor(
            [item.age / self.capacity for item in self.items],
            dtype=torch.float32,
            device=device,
        )
        return clips, normalized_ages

    def __len__(self) -> int:
        return len(self.items)


class _ColorJitterPro(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms import functional as functional

        ranges = (
            (0.6, 1.4),
            (0.7, 1.3),
            (0.5, 1.5),
            (-0.06, 0.06),
            (0.7, 1.3),
        )
        for operation in torch.randperm(5).tolist():
            factor = torch.empty(1).uniform_(*ranges[operation]).item()
            if operation == 0:
                image = functional.adjust_brightness(image, factor)
            elif operation == 1:
                image = functional.adjust_contrast(image, factor)
            elif operation == 2:
                image = functional.adjust_saturation(image, factor)
            elif operation == 3:
                image = functional.adjust_hue(image, factor)
            else:
                image = functional.adjust_gamma(image.clamp(1e-8, 1.0), factor)
        return image


class FramewiseStrongAugmentation(nn.Module):
    """Tensor implementation of the strong augmentation used by SpoTTA."""

    def __init__(self, image_size: int = 224, gaussian_std: float = 0.005):
        super().__init__()
        self.image_size = image_size
        self.gaussian_std = gaussian_std
        self.color_jitter = _ColorJitterPro()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms import functional as functional

        original_height, original_width = image.shape[-2:]
        image = image.clamp(0.0, 1.0)
        image = self.color_jitter(image)
        padding = self.image_size // 2
        image = functional.pad(image, [padding] * 4, padding_mode="edge")
        padded_size = self.image_size + 2 * padding
        angle = torch.empty(1, device=image.device).uniform_(-15, 15).item()
        translate = [
            torch.empty(1, device=image.device).uniform_(-1 / 16, 1 / 16).item()
            * padded_size,
            torch.empty(1, device=image.device).uniform_(-1 / 16, 1 / 16).item()
            * padded_size,
        ]
        scale = torch.empty(1, device=image.device).uniform_(0.9, 1.1).item()
        image = functional.affine(
            image, angle=angle, translate=translate, scale=scale, shear=0
        )
        sigma = torch.empty(1, device=image.device).uniform_(0.001, 0.5).item()
        image = functional.gaussian_blur(image, kernel_size=5, sigma=[sigma])
        image = functional.center_crop(image, [self.image_size, self.image_size])
        if torch.rand(1, device=image.device).item() < 0.5:
            image = functional.hflip(image)
        image = image + torch.randn_like(image) * self.gaussian_std
        image = image.clamp(0.0, 1.0)
        if image.shape[-2:] != (original_height, original_width):
            image = functional.resize(
                image, [original_height, original_width], antialias=None
            )
        return image


def _logits(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if output.ndim > 3:
        output = output[-1]
    if output.ndim != 3:
        raise ValueError(
            "SpoTTA E2ESpot expects model logits shaped [batch, time, classes], "
            f"but received {tuple(output.shape)}."
        )
    return output


class SpoTTA:
    """Stateful SpoTTA tool for one continuous target-set inference session."""

    def __init__(self, source_model: nn.Module, config: SpoTTAConfig | Any):
        self.config = (
            config if isinstance(config, SpoTTAConfig) else SpoTTAConfig.from_mapping(config)
        )
        try:
            self.device = next(source_model.parameters()).device
        except StopIteration as exc:
            raise ValueError("SpoTTA requires a model with parameters.") from exc

        self.student = deepcopy(source_model).to(self.device)
        for parameter in self.student.parameters():
            parameter.requires_grad_(False)
        self.student_bn = replace_batch_norm(self.student, self.config.alpha)
        if not self.student_bn:
            raise ValueError("SpoTTA requires at least one BatchNorm layer.")
        for layer in self.student_bn:
            layer.requires_grad_(True)

        self.teacher = deepcopy(self.student).to(self.device)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher_bn = [
            module
            for module in self.teacher.modules()
            if isinstance(module, RobustBatchNorm)
        ]
        self.tether_layers = [*self.student_bn, *self.teacher_bn]
        self._tether_rho_ema: torch.Tensor | None = None

        parameters = [
            parameter
            for layer in self.student_bn
            for parameter in layer.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta, 0.999),
            weight_decay=0.0,
        )
        self.memory = HeaderCSTUMemory(
            self.config.memory_capacity, self.config.lambda_t, self.config.lambda_u
        )
        self.augmentation = FramewiseStrongAugmentation()
        self._drift_mean_ema = 0.0
        self._drift_var_ema = 1.0
        self.last_ema_nu = self.config.ema_nu
        self.clips_seen = 0
        self.clips_gated = 0
        self.memory_insertions = 0
        self.update_attempts = 0
        self.updates_completed = 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "clips_seen": self.clips_seen,
            "clips_gated": self.clips_gated,
            "clips_admitted": self.clips_gated,
            "memory_add_attempts": self.clips_gated,
            "memory_insertions": self.memory_insertions,
            "memory_occupancy": len(self.memory),
            "update_attempts": self.update_attempts,
            "updates_completed": self.updates_completed,
            "last_ema_nu": self.last_ema_nu,
        }

    def _update_tether(self) -> None:
        divergences = torch.tensor(
            [layer.anchor_kl for layer in self.tether_layers], dtype=torch.float32
        )
        z_score = (
            (divergences - divergences.mean())
            / (divergences.std(unbiased=False) + 1e-8)
        ).clamp(-1.0, 1.0)
        rho = (z_score + 1.0) / 2.0 * self.config.tether_cap
        if self._tether_rho_ema is None:
            self._tether_rho_ema = rho
        else:
            self._tether_rho_ema = 0.9 * self._tether_rho_ema + 0.1 * rho
        for layer, value in zip(self.tether_layers, self._tether_rho_ema.tolist()):
            layer.tether_rho = float(value)

    def _update_teacher_parameters(self) -> None:
        raw_drift = sum(layer.last_batch_drift for layer in self.student_bn) / len(
            self.student_bn
        )
        gamma = self.config.drift_gamma
        self._drift_mean_ema = (
            (1 - gamma) * self._drift_mean_ema + gamma * raw_drift
        )
        self._drift_var_ema = (
            (1 - gamma) * self._drift_var_ema
            + gamma * (raw_drift - self._drift_mean_ema) ** 2
        )
        z_score = (raw_drift - self._drift_mean_ema) / (
            math.sqrt(self._drift_var_ema) + 1e-8
        )
        sigmoid = 1 / (
            1
            + math.exp(
                -self.config.drift_scale * (z_score - self.config.drift_threshold)
            )
        )
        self.last_ema_nu = self.config.ema_nu + (
            self.config.max_ema_nu - self.config.ema_nu
        ) * sigmoid
        with torch.no_grad():
            for teacher_parameter, student_parameter in zip(
                self.teacher.parameters(), self.student.parameters()
            ):
                teacher_parameter.mul_(1 - self.last_ema_nu).add_(
                    student_parameter, alpha=self.last_ema_nu
                )

    def _update_model(self) -> None:
        self.update_attempts += 1
        if not self.memory.items:
            return
        self._update_tether()
        clips, ages = self.memory.batch(self.device)

        self.teacher.train()
        with torch.no_grad():
            teacher_logits = _logits(self.teacher(clips))

        if self.config.augmentation:
            batch, time, channels, height, width = clips.shape
            augmented = self.augmentation(
                clips.reshape(batch * time, channels, height, width)
            ).reshape(batch, time, channels, height, width)
        else:
            augmented = clips

        self.student.train()
        student_logits = _logits(self.student(augmented))
        batch, time, classes = student_logits.shape
        if classes != 2:
            raise ValueError(
                "The 53.33 SpoTTA recipe is binary and expects Background/Header logits."
            )
        student_flat = student_logits.reshape(-1, classes)
        teacher_flat = teacher_logits.reshape(-1, classes)
        soft_cross_entropy = -(
            teacher_flat.softmax(dim=1) * student_flat.log_softmax(dim=1)
        ).sum(dim=1)
        timeliness = torch.exp(-ages) / (1 + torch.exp(-ages))
        loss = (soft_cross_entropy * timeliness.repeat_interleave(time)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_teacher_parameters()
        self.updates_completed += 1

    def _amp_context(self, use_amp: bool):
        if use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda")
        return nullcontext()

    def predict(
        self, sequence: torch.Tensor | np.ndarray, use_amp: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence, dtype=torch.float32)
        if sequence.ndim == 4:
            sequence = sequence.unsqueeze(0)
        if sequence.ndim != 5:
            raise ValueError(
                "SpoTTA expects [T,C,H,W] or [B,T,C,H,W] input, "
                f"but received {tuple(sequence.shape)}."
            )
        if sequence.shape[2] != 3 and sequence.shape[1] == 3:
            sequence = sequence.transpose(1, 2).contiguous()
        if sequence.shape[2] != 3:
            raise ValueError(f"SpoTTA E2ESpot requires RGB clips, got {tuple(sequence.shape)}.")
        sequence = sequence.to(self.device)

        self.teacher.eval()
        with torch.no_grad():
            gate_probabilities = _logits(self.teacher(sequence)).softmax(dim=-1)
        if gate_probabilities.shape[-1] != 2:
            raise ValueError(
                "The 53.33 SpoTTA recipe requires two output classes: Background and Header."
            )

        batch_size = sequence.shape[0]
        self.clips_seen += batch_size
        predicted_classes = gate_probabilities.argmax(dim=-1)
        frame_uncertainty = 1 - gate_probabilities.max(dim=-1).values
        for index in range(batch_size):
            action_mask = predicted_classes[index] == self.config.action_class_index
            if int(action_mask.sum().item()) < self.config.min_action_frames:
                continue
            clip_uncertainty = float(frame_uncertainty[index][action_mask].min().item())
            if clip_uncertainty > self.config.gate_threshold:
                continue

            self.clips_gated += 1
            if self.memory.add(sequence[index], clip_uncertainty):
                self.memory_insertions += 1
            if self.clips_gated % self.config.update_frequency == 0:
                self._update_model()

        self.teacher.eval()
        with torch.no_grad(), self._amp_context(use_amp):
            probabilities = _logits(self.teacher(sequence)).softmax(dim=-1)
        predicted = probabilities.argmax(dim=-1)
        return predicted.cpu().numpy(), probabilities.cpu().numpy()
