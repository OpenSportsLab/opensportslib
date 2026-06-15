"""Semantic canonical config types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SystemConfig:
    paths: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    gpu: dict[str, Any] = field(default_factory=dict)
    reproducibility: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    common: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    schema_version: int = 3
    task: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    load: dict[str, Any] = field(default_factory=dict)
    components: dict[str, Any] = field(default_factory=dict)
    topology: list[dict[str, Any]] = field(default_factory=list)
    policies: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    trainer: dict[str, Any] = field(default_factory=dict)
    epochs: int | None = None
    criterion: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)


@dataclass
class IOConfig:
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigDocument:
    task: str
    version: int
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    io: IOConfig | None = None
