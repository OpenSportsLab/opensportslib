"""Discoverable, transactional configuration editing before model allocation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, TypedDict

from .loader import _compose_yaml_layers, _normalize_cpu_loader_backend
from .migrate import migrate_config
from .runtime_adapter import namespace_to_plain_dict
from .validate import validate_config


class DataOptions(TypedDict, total=False):
    train_set: str
    valid_set: str
    test_set: str
    data_root: str


class RuntimeOptions(TypedDict, total=False):
    device: str
    seed: int
    output_dir: str


class TrainingOptions(TypedDict, total=False):
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_workers: int


class InferenceOptions(TypedDict, total=False):
    batch_size: int
    num_workers: int
    max_new_tokens: int
    temperature: float
    do_sample: bool
    top_p: float


def _leaves(value, prefix=""):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _leaves(item, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, value


def _get(doc, path, default=None):
    value = doc
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _set(doc, path, value):
    keys = path.split(".")
    parent = doc
    for key in keys[:-1]:
        parent = parent.setdefault(key, {})
    parent[keys[-1]] = deepcopy(value)


def _overlap(a, b):
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def _resolve(doc):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.create(doc), resolve=True, throw_on_missing=True)


@dataclass(frozen=True)
class Option:
    paths: tuple[str, ...]
    description: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple | None = None
    requires_initialization: bool = False


class Config:
    """Editable OSL config. Loading this object never allocates model weights."""

    def __init__(self, document, *, source=None, weights=None, provenance=None):
        self._document = deepcopy(migrate_config(document))
        self.source = str(source) if source is not None else None
        self.weights = weights
        self._provenance = dict(provenance or {})
        self._updates: dict[str, Any] = {}
        self._friendly: dict[str, dict[str, Any]] = {}
        self._has_dotted_overrides = False
        self._validate(self._document)

    @classmethod
    def from_file(cls, path: str | Path):
        from omegaconf import OmegaConf
        from .schemas.schema_canonical import is_canonical_schema

        path = Path(path).expanduser().resolve()
        layers = _compose_yaml_layers(path) or [path]
        document = OmegaConf.create({})
        provenance = {}
        for layer in layers:
            raw = OmegaConf.load(layer)
            for key, _ in _leaves(OmegaConf.to_container(raw, resolve=False)):
                provenance[key] = str(layer)
            document = OmegaConf.merge(document, raw)
        raw = OmegaConf.to_container(document, resolve=False)
        if not is_canonical_schema(raw):
            raw = migrate_config(_resolve(raw))
            provenance = {key: str(path) for key, _ in _leaves(raw)}
        return cls(raw, source=path, provenance=provenance)

    @classmethod
    def from_pretrained(cls, model_id: str):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(model_id)
        try:
            path = hf_hub_download(repo_id=model_id, filename="config.yaml")
        except Exception as exc:
            raise ValueError(
                f"Cannot load OSL config.yaml from {model_id!r}: {exc}. "
                "Check the model ID, network access, authentication for private models, "
                "or provide a config file."
            ) from exc
        config = cls.from_file(path)
        config.weights = model_id
        config._provenance = {key: model_id for key in config._provenance}
        return config

    def get_config(self) -> dict[str, Any]:
        """Return a detached, resolved canonical document."""
        return deepcopy(_resolve(self._document))

    @staticmethod
    def _validate(document):
        resolved = _resolve(document)
        validate_config(resolved)
        for split, settings in _get(resolved, "DATA.common.splits", {}).items():
            overlap = settings.get("overlap_len")
            clip = _get(resolved, "DATA.inputs.video.sampling.clip_len")
            if overlap is not None and clip is not None and not 0 <= overlap < clip:
                raise ValueError(f"DATA.common.splits.{split}.overlap_len must be smaller than clip_len")
        native = _get(resolved, "TRAIN.execution.native_vl", {})
        if "min_pixels" in native and "max_pixels" in native and native["min_pixels"] > native["max_pixels"]:
            raise ValueError("TRAIN.execution.native_vl.min_pixels must not exceed max_pixels")
        return resolved

    def _locked_paths(self, doc):
        locked = set()
        for key, comp in _get(doc, "MODEL.components", {}).items():
            name = _get(comp, "source.name", "") or ""
            if name.startswith("h5_header_"):
                from .rule_variants import HEADER_RULE_VARIANTS, SKELETON_RULE_VARIANTS

                variant = {**HEADER_RULE_VARIANTS, **SKELETON_RULE_VARIANTS}.get(name, {})
                locked.update(f"MODEL.components.{key}.params.{param}" for param in variant)
        return locked

    def _registry(self):
        doc = self.get_config()
        result = {}
        missing = object()

        def add(name, paths, description, minimum=None, maximum=None, choices=None, init=False):
            if isinstance(paths, str):
                paths = (paths,)
            paths = tuple(p for p in paths if _get(doc, p, missing) is not missing)
            if paths:
                result[name] = Option(paths, description, minimum, maximum, choices, init)

        splits = _get(doc, "DATA.common.splits", {})
        for split in ("train", "valid", "test"):
            add(f"data.{split}_set", f"DATA.common.splits.{split}.annotation_path", f"{split.title()} annotation manifest")
        add("data.data_root", "DATA.common.data_root", "Root directory for data and referenced split paths")
        add("runtime.device", "SYSTEM.device", "Execution device", choices=("auto", "cpu", "cuda"), init=True)
        add("runtime.seed", "SYSTEM.reproducibility.seed", "Random seed (enables reproducibility)", 0, 2**32 - 1)
        add("runtime.output_dir", "SYSTEM.paths.save_dir", "Root directory for new run artifacts", init=True)
        task = str(doc.get("TASK", "")).lower()
        trainable = _get(doc, "TRAIN.execution.enabled", True) and _get(doc, "MODEL.metadata.family") != "RuleBased"
        backend = _get(doc, "MODEL.metadata.backend", _get(doc, "TRAIN.execution.backend", "baseline"))
        if trainable:
            for name, path, minimum in (("epochs", "epochs", 1), ("learning_rate", "optimizer.lr", 0), ("weight_decay", "optimizer.weight_decay", 0)):
                add(f"training.{name}", f"TRAIN.{path}", name.replace("_", " ").capitalize(), minimum)
            for field in ("batch_size", "num_workers"):
                add(f"training.{field}", f"DATA.common.splits.train.dataloader.{field}", f"Training {field.replace('_', ' ')}", 1 if field == "batch_size" else 0)
        # VQA engines generate one sample at a time without a DataLoader.
        if task != "vqa" and _get(doc, "MODEL.metadata.family") != "RuleBased":
            for field in ("batch_size", "num_workers"):
                add(f"inference.{field}", tuple(f"DATA.common.splits.{s}.dataloader.{field}" for s in splits if s in {"valid", "test", "valid_data_frames"}), f"Validation and test {field.replace('_', ' ')}", 1 if field == "batch_size" else 0)
        if task == "vqa" and backend in {"qwen_xvars_infer", "qwen_vl_native_infer", "xvars_videochatgpt"}:
            for name in ("max_new_tokens", "temperature", "repetition_penalty", "no_repeat_ngram_size"):
                if backend == "qwen_vl_native_infer" and name in {"repetition_penalty", "no_repeat_ngram_size"}:
                    continue
                add(f"inference.{name}", f"TRAIN.execution.generation.{name}", f"Generation {name.replace('_', ' ')}", 1 if name in {"max_new_tokens", "repetition_penalty"} else 0)
            result["inference.do_sample"] = Option(
                ("TRAIN.execution.generation.do_sample",), "Enable sampling during generation"
            )
            result["inference.top_p"] = Option(
                ("TRAIN.execution.generation.top_p",), "Nucleus sampling probability", 0, 1
            )
        sections = {"sampling": "DATA.inputs.video.sampling", "augmentation": "DATA.inputs.video.augmentations"}
        if trainable:
            sections.update({"scheduler": "TRAIN.scheduler", "training_sampling": "TRAIN.sampling"})
        if task == "vqa":
            sections["prompt"] = "TRAIN.execution.prompt"
            if trainable:
                sections.update({"lora": "TRAIN.execution.lora", "sft": "TRAIN.execution.sft"})
            sections["native_vl"] = "TRAIN.execution.native_vl"
        if _get(doc, "MODEL.policies.test_time_adaptation.enabled", False):
            sections["adaptation"] = "MODEL.policies.test_time_adaptation"
        for key, comp in _get(doc, "MODEL.components", {}).items():
            if comp.get("kind") in {"algorithm", "postprocessor"}:
                sections[key] = f"MODEL.components.{key}.params"
        locked = self._locked_paths(doc)
        for group, root in sections.items():
            for suffix, value in _leaves(_get(doc, root, {})):
                if not suffix or suffix.split(".")[-1] in {"type", "name", "label", "head_name", "trainable_parameters"}:
                    continue
                path = f"{root}.{suffix}"
                if path in locked or isinstance(value, dict):
                    continue
                add(f"{group}.{suffix}", path, f"{group.replace('_', ' ').title()}: {suffix.replace('_', ' ')}", init=group not in {"scheduler", "training_sampling", "sft", "prompt"})
        return result

    def options(self) -> dict[str, dict[str, Any]]:
        doc = self.get_config()
        options = {}
        for name, spec in self._registry().items():
            defaults = {"inference.do_sample": False, "inference.top_p": 1.0}
            values = {path: deepcopy(_get(doc, path, defaults.get(name))) for path in spec.paths}
            current = list(values.values())
            options[name] = {
                "value": current[0] if all(v == current[0] for v in current) else values,
                "type": type(current[0]).__name__, "description": spec.description,
                "minimum": spec.minimum, "maximum": spec.maximum,
                "choices": list(spec.choices) if spec.choices else None,
                "paths": list(spec.paths),
                "sources": {p: self._provenance.get(p, self.source or "runtime") for p in spec.paths},
                "requires_initialization": spec.requires_initialization,
            }
        return options

    @staticmethod
    def _check_value(path, value, old, spec=None):
        if isinstance(old, bool):
            valid = isinstance(value, bool)
        elif isinstance(old, int):
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(old, float):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif old is None:
            valid = value is None or isinstance(value, (str, int, float, bool, list, dict))
        else:
            valid = isinstance(value, type(old))
        if not valid:
            raise ValueError(f"{path}: expected {type(old).__name__}, got {type(value).__name__}")
        if isinstance(value, (float, int)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(f"{path}: value must be finite")
        if spec is not None:
            if spec.choices and value not in spec.choices:
                raise ValueError(f"{path}: choose one of {spec.choices}")
            if (spec.minimum is not None and value < spec.minimum) or (spec.maximum is not None and value > spec.maximum):
                raise ValueError(f"{path}: value outside [{spec.minimum}, {spec.maximum}]")

    def update(self, *, data: DataOptions | None = None, runtime: RuntimeOptions | None = None,
               training: TrainingOptions | None = None, inference: InferenceOptions | None = None,
               overrides: dict[str, Any] | None = None, **groups):
        requested = {**groups, **{k: v for k, v in {"data": data, "runtime": runtime, "training": training, "inference": inference}.items() if v is not None}}
        registry = self._registry()
        doc = self.get_config()
        changes = {}
        friendly = deepcopy(self._friendly)
        for group, settings in requested.items():
            if not isinstance(settings, dict):
                raise ValueError(f"{group}: expected a dictionary of options")
            if not any(name.startswith(group + ".") for name in registry):
                raise ValueError(f"Unsupported option group {group!r}; inspect config.options()")
            for name, value in settings.items():
                key = f"{group}.{name}"
                if key not in registry:
                    raise ValueError(f"Unsupported option {key!r}; inspect config.options()")
                spec = registry[key]
                for path in spec.paths:
                    old = _get(doc, path, {"inference.do_sample": False, "inference.top_p": 1.0}.get(key))
                    self._check_value(key, value, old, spec)
                    if path in changes:
                        raise ValueError(f"Conflicting options for {path}")
                    changes[path] = value
                friendly.setdefault(group, {})[name] = deepcopy(value)
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("overrides: expected a dictionary of canonical dotted paths")
        missing = object()
        for path, value in (overrides or {}).items():
            if not isinstance(path, str) or _get(doc, path, missing) is missing:
                raise ValueError(f"Unknown config path {path!r}; inspect config.get_config()")
            if any(_overlap(path, other) for other in changes):
                raise ValueError(f"Conflicting overrides for {path}")
            self._check_value(path, value, _get(doc, path))
            changes[path] = value
        candidate = deepcopy(self._document)
        for path, value in changes.items():
            _set(candidate, path, value)
        for locked in self._locked_paths(_resolve(candidate)):
            if any(_overlap(path, locked) for path in changes):
                raise ValueError(f"{locked} is controlled by the selected rule variant")
        resolved = _resolve(candidate)
        generation = _get(resolved, "TRAIN.execution.generation", {})
        if generation.get("do_sample") and float(generation.get("temperature", 0)) <= 0:
            raise ValueError("inference.temperature must be positive when inference.do_sample is true")
        for spec in registry.values():
            for path in spec.paths:
                defaults = {
                    "TRAIN.execution.generation.do_sample": False,
                    "TRAIN.execution.generation.top_p": 1.0,
                }
                old = _get(doc, path, defaults.get(path, _get(resolved, path)))
                value = _get(resolved, path, defaults.get(path))
                self._check_value(path, value, old, spec)
        self._synchronize(candidate, changes)
        self._validate(candidate)
        updates = deepcopy(self._updates)
        for path, value in changes.items():
            for old in list(updates):
                if _overlap(old, path):
                    del updates[old]
            updates[path] = deepcopy(value)
        self._document = candidate
        self._updates = updates
        self._friendly = friendly
        self._has_dotted_overrides |= bool(overrides)
        self._provenance.update({path: "user" for path in changes})
        return self

    @staticmethod
    def _synchronize(doc, changes):
        """Keep aliases consumed by existing backends in agreement."""
        task = str(doc.get("TASK", "")).lower()
        for path, value in list(changes.items()):
            if path == "DATA.common.data_root":
                for split, split_cfg in _get(doc, "DATA.common.splits", {}).items():
                    source_path = split_cfg.get("source_path")
                    if source_path is not None and "${" not in str(source_path):
                        _set(doc, f"DATA.common.splits.{split}.source_path", value)
            if path == "SYSTEM.reproducibility.seed":
                _set(doc, "SYSTEM.reproducibility.use_seed", True)
            if path == "SYSTEM.device":
                _set(doc, "MODEL.runtime.device", value)
                if value != "auto":
                    _set(doc, "TRAIN.execution.hf.prefer_cuda", value == "cuda")
            if path.endswith(".dataloader.num_workers") and value == 0:
                root = path.rsplit(".", 1)[0]
                _set(doc, root + ".persistent_workers", False)
                _set(doc, root + ".prefetch_factor", None)
            if task == "vqa" and path.endswith(".dataloader.batch_size"):
                split = path.split(".")[3]
                field = "per_device_train_batch_size" if split == "train" else "per_device_eval_batch_size"
                _set(doc, "TRAIN.execution.sft." + field, value)
            if task == "vqa" and path == "DATA.common.splits.train.dataloader.num_workers":
                _set(doc, "TRAIN.execution.sft.dataloader_num_workers", value)

    def apply_to(self, config):
        """Reapply explicit updates after a runtime/checkpoint config merge."""
        doc = deepcopy(namespace_to_plain_dict(config))
        source_values = self.get_config()
        for path, expression in _leaves(self._document):
            if isinstance(expression, str) and "${" in expression:
                refs = re.findall(r"\$\{([^}]+)\}", expression)
                if _get(doc, path) == _get(source_values, path) or any(any(_overlap(ref, changed) for changed in self._updates) for ref in refs):
                    _set(doc, path, expression)
        for path, value in self._updates.items():
            _set(doc, path, value)
        self._synchronize(doc, self._updates)
        return _normalize_cpu_loader_backend(self._validate(doc))

    def remote_overrides(self):
        if self._has_dotted_overrides:
            raise ValueError("Remote execution does not accept dotted config overrides")
        forbidden = set(self._friendly) - {"data", "inference"}
        if forbidden or "num_workers" in self._friendly.get("inference", {}):
            raise ValueError("Remote tuning accepts advertised inference options only; runtime and workers are server-controlled")
        return deepcopy(self._friendly.get("inference", {}))
