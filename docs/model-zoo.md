# Model Zoo

This page lists OpenSportsLib checkpoints, config-only algorithms, and supporting
model components published on Hugging Face. Task checkpoints can be loaded with
`load_weights(...)`; config-only algorithms and supporting components use the
loading instructions in their sections below.

Scores, class lists, and intended-use statements below come from model cards,
published configs, or the library's component configuration. Scores are not
computed or verified at documentation-build time. Missing training provenance
and scores are marked **Not reported**. Follow each linked repository for its
artifact details; a listing here does not certify end-to-end inference.

## Available Models

| Model | Task | Dataset trained on | Backbone / architecture | Classes / label set | Scores | Hugging Face link | Loading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `OSL-cls-action-mvitv2` | Action / Event Classification | SoccerNet - MVFouls classification subset | MViT v2 | Not reported on model card | Accuracy: 0.57<br>Balanced Accuracy: 0.40<br>Top-2: 0.78 | [OpenSportsLab/OSL-cls-action-mvitv2](https://huggingface.co/OpenSportsLab/OSL-cls-action-mvitv2) | `myModel.load_weights(weights="OpenSportsLab/OSL-cls-action-mvitv2")` |
| `OSL-loc-snbas-2023-e2e` | Action Spotting / Localization | SoccerNet - Ball Action Spotting 2023 | E2E, DALI backend | PASS, DRIVE | tight mAP: 71.48<br>loose mAP: 85.62 | [OpenSportsLab/OSL-loc-snbas-2023-e2e](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2023-e2e) | `myModel.load_weights(weights="OpenSportsLab/OSL-loc-snbas-2023-e2e")` |
| `OSL-loc-snbas-2025-e2e` | Action Spotting / Localization | SoccerNet - Ball Action Spotting 2025 | E2E, DALI backend | PASS, DRIVE, HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL | tight mAP: 47.98<br>loose mAP: 58.35 | [OpenSportsLab/OSL-loc-snbas-2025-e2e](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2025-e2e) | `myModel.load_weights(weights="OpenSportsLab/OSL-loc-snbas-2025-e2e")` |
| `OSL-VQA-XFOUL-XVARS-lora` | Visual Question Answering (VQA) | OSL-XFoul | X-VARS VideoChatGPT + LoRA | Referee-style soccer VQA | Accuracy: 72.24%<br>Balanced Accuracy: 50.00% | [OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora](https://huggingface.co/OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora) | `myModel.load_weights(weights="OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora")` |
| `OSL-VQA-XFOUL-qwen2.5-7B-VL-lora` | Visual Question Answering (VQA) | OSL-XFoul | Qwen2.5-VL-7B-Instruct + LoRA | Referee-style soccer VQA | Accuracy: 65.77%<br>Balanced Accuracy: 36.59% | [OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora](https://huggingface.co/OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora) | `myModel.load_weights(weights="OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora")` |
| `OSL-VQA-XFOUL-qwen3-8B-VL-lora` | Visual Question Answering (VQA) | OSL-XFoul | Qwen3-VL-8B-Instruct + LoRA | Referee-style soccer VQA | Accuracy: 69.96%<br>Balanced Accuracy: 48.33% | [OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora](https://huggingface.co/OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora) | `myModel.load_weights(weights="OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora")` |
| `skeleton-header-max-recall` | Header spotting | Not applicable: rule-based algorithm | Skeleton maximum-recall rules; CPU, HDF5 tracking inputs | `header` | Not reported | [OpenSportsLab/skeleton-header-max-recall](https://huggingface.co/OpenSportsLab/skeleton-header-max-recall) | [Config only; no weights](#skeleton-header-max-recall) |
| `OSL-loc-tennis-public` | Tennis action spotting / localization | Not reported | E2E: `rny002_gsm` + GRU | `no_swing`, `swing` (published config) | Not reported | [OpenSportsLab/OSL-loc-tennis-public](https://huggingface.co/OpenSportsLab/OSL-loc-tennis-public) | [Checkpoint and local inputs](#osl-loc-tennis-public) |

## Supporting Components

These repositories supply components for a configured VQA model. They are not
standalone task models to pass to a task wrapper's `load_weights(...)` method.

| Component | Role | Training source | Scores | Hugging Face link | Configuration |
| --- | --- | --- | --- | --- | --- |
| `base_model_videoChatGPT` | X-VARS decoder model and tokenizer bundle | Not reported | Not reported | [OpenSportsLab/base_model_videoChatGPT](https://huggingface.co/OpenSportsLab/base_model_videoChatGPT) | [Decoder repository](#base_model_videochatgpt) |
| `trained-clip-vit-large-patch14` | Visual encoder checkpoint for the CLIP ViT-L/14 component | Not reported | Not reported | [OpenSportsLab/trained-clip-vit-large-patch14](https://huggingface.co/OpenSportsLab/trained-clip-vit-large-patch14) | [Visual encoder weights](#trained-clip-vit-large-patch14) |

## OSL-cls-action-mvitv2

**Intended use:** video-based soccer action / event classification.

**Dataset/training source:** SoccerNet - MVFouls classification subset, using
video clips.

**Reported metrics:**

| Metric | Score |
| --- | --- |
| Accuracy | 0.57 |
| Balanced Accuracy | 0.40 |
| Top-2 | 0.78 |

**Hugging Face:** [OpenSportsLab/OSL-cls-action-mvitv2](https://huggingface.co/OpenSportsLab/OSL-cls-action-mvitv2)

```python
myModel.load_weights(weights="OpenSportsLab/OSL-cls-action-mvitv2")
```

## OSL-loc-snbas-2023-e2e

**Intended use:** video-based soccer action spotting / localization.

**Dataset/training source:** SoccerNet - Ball Action Spotting 2023, using video
clips at 224p resolution. The model card reports two classes: `PASS` and
`DRIVE`.

**Reported metrics:**

| Metric | Score |
| --- | --- |
| tight mAP | 71.48 |
| loose mAP | 85.62 |

**Hugging Face:** [OpenSportsLab/OSL-loc-snbas-2023-e2e](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2023-e2e)

```python
myModel.load_weights(weights="OpenSportsLab/OSL-loc-snbas-2023-e2e")
```

## OSL-loc-snbas-2025-e2e

**Intended use:** video-based soccer action spotting / localization.

**Dataset/training source:** SoccerNet - Ball Action Spotting 2025, using video
clips at 224p resolution. The model card reports twelve classes: `PASS`,
`DRIVE`, `HEADER`, `HIGH PASS`, `OUT`, `CROSS`, `THROW IN`, `SHOT`,
`BALL PLAYER BLOCK`, `PLAYER SUCCESSFUL TACKLE`, `FREE KICK`, and `GOAL`.

**Reported metrics:**

| Metric | Score |
| --- | --- |
| tight mAP | 47.98 |
| loose mAP | 58.35 |

**Hugging Face:** [OpenSportsLab/OSL-loc-snbas-2025-e2e](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2025-e2e)

```python
myModel.load_weights(weights="OpenSportsLab/OSL-loc-snbas-2025-e2e")
```

## OSL-VQA-XFOUL-XVARS-lora

**Intended use:** soccer visual question answering focused on foul analysis and
referee-style explanations.

**Dataset/training source:** OSL-XFoul, using video clips and VQA supervision.

**Reported metrics:**

| Metric | Score |
| --- | --- |
| Accuracy | 72.24% |
| Balanced Accuracy | 50.00% |

**Hugging Face:** [OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora](https://huggingface.co/OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora)

```python
myModel.load_weights(weights="OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora")
```

Recommended config:

```python
config="opensportslib/configs/vqa/xvars.yaml"
```

## OSL-VQA-XFOUL-qwen2.5-7B-VL-lora

**Intended use:** native end-to-end soccer visual question answering with Qwen
VL.

**Dataset/training source:** OSL-XFoul, using video clips and VQA supervision.

**Reported metrics:**

| Metric | Score |
| --- | --- |
| Accuracy | 65.77% |
| Balanced Accuracy | 36.59% |

**Hugging Face:** [OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora](https://huggingface.co/OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora)

```python
myModel.load_weights(weights="OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora")
```

Recommended config:

```python
config="opensportslib/configs/vqa/qwen3_vl_native.yaml"
```

## OSL-VQA-XFOUL-qwen3-8B-VL-lora

**Intended use:** native end-to-end soccer visual question answering with Qwen
VL.

**Dataset/training source:** OSL-XFoul, using video clips and VQA supervision.

**Reported metrics:**

| Metric | Score |
| --- | --- |
| Accuracy | 69.96% |
| Balanced Accuracy | 48.33% |

**Hugging Face:** [OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora](https://huggingface.co/OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora)

```python
myModel.load_weights(weights="OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora")
```

Recommended config:

```python
config="opensportslib/configs/vqa/qwen3_vl_native.yaml"
```


## skeleton-header-max-recall

**Hugging Face:** [OpenSportsLab/skeleton-header-max-recall](https://huggingface.co/OpenSportsLab/skeleton-header-max-recall)

This inference-only, rule-based algorithm detects headers from player-joint and
ball HDF5 tracking inputs on CPU. The published config selects
`h5_header_skeleton_max_recall` and the label `header`. It contains no learned
checkpoint and does not support training. **Reported scores: Not reported.**

The repository is gated. Request access on its Hugging Face page and, once
approved, authenticate your local Hugging Face client before downloading. Use an
OSL manifest referencing your own player-joint and ball HDF5 files; see the
[HDF5 header spotting guide](headers/README.md) for the input format.

Download only the config and replace its machine-specific data paths:

```python
from huggingface_hub import hf_hub_download
from opensportslib.apis import Config, LocalizationModel

config_path = hf_hub_download(
    repo_id="OpenSportsLab/skeleton-header-max-recall",
    filename="config.yaml",
)
config = Config.from_file(config_path)
config.update(
    data={
        "data_root": "/path/to/tracking",
        "test_set": "/path/to/header_manifest.json",
    },
    runtime={"device": "cpu"},
)
model = LocalizationModel(config=config, weights=None)
predictions = model.infer(use_wandb=False)
model.save_predictions(output_path="header_predictions.json", predictions=predictions)
```

Use `Config.from_file` for this downloaded config: `Config.from_pretrained`
remembers the repository as a weight source, whereas this algorithm has no
weights to load.

**Current server limitation:** Hugging Face registration on the inference server
checks the legacy `TRAIN.runner` field to identify config-only algorithms. This
repository uses `MODEL.metadata.runner`, so registration currently rejects it
for lacking a checkpoint. Use the local Python workflow above until server
registration supports the canonical config. This documentation does not fix
that limitation.

## OSL-loc-tennis-public

**Hugging Face:** [OpenSportsLab/OSL-loc-tennis-public](https://huggingface.co/OpenSportsLab/OSL-loc-tennis-public)

The repository contains `config.yaml` and `model.pt` for E2E tennis spotting.
The published config specifies the `rny002_gsm` backbone, a GRU head, the labels
`no_swing` and `swing`, and a DALI video loader. Its sampling uses an 11-frame
clip at an extraction rate of 2 fps. The current config loader migrates the
published legacy config when loading it.

**Training source: Not reported. Reported scores: Not reported.** The repository
has no model card; the architecture and labels above come from its config.

Set local annotations and the root used to resolve their video paths before
constructing the model. `Config.from_pretrained` downloads the config and
remembers the repository ID; constructing `LocalizationModel` then loads the
checkpoint.

```python
from opensportslib.apis import Config, LocalizationModel

config = Config.from_pretrained("OpenSportsLab/OSL-loc-tennis-public")
config.update(
    data={
        "data_root": "/path/to/tennis/videos",
        "test_set": "/path/to/tennis/test.json",
    },
    runtime={"device": "cpu"},
)
model = LocalizationModel(config=config)
predictions = model.infer(use_wandb=False)
model.save_predictions(output_path="tennis_predictions.json", predictions=predictions)
```

The CPU example uses the library's OpenCV fallback. To use the published DALI
route, install GPU support and DALI with `opensportslib setup --dali`, and select
`device="cuda"`. Provide an [OSL JSON](data/osl-json-format.md) test manifest
whose video paths resolve under your chosen data root.

## base_model_videoChatGPT

**Hugging Face:** [OpenSportsLab/base_model_videoChatGPT](https://huggingface.co/OpenSportsLab/base_model_videoChatGPT)

This is the decoder and tokenizer bundle used by the X-VARS VQA backend. The
repository contains `config.json`, sharded safetensors weights, their index, and
tokenizer files. **Training source: Not reported. Reported scores: Not reported.**

The canonical `opensportslib/configs/vqa/xvars.yaml` already selects this
repository. The relevant component setting is:

```yaml
MODEL:
  components:
    llm_decoder:
      params:
        repo_id: OpenSportsLab/base_model_videoChatGPT
```

This is a fragment to apply to the X-VARS config, not a complete task config.
The decoder works with the configured visual encoder, projector, and task
adapter. See [X-VARS setup and artifact flow](tools/vqa.md) for the workflow and
[OSL-VQA-XFOUL-XVARS-lora](#osl-vqa-xfoul-xvars-lora) for the task adapter.

## trained-clip-vit-large-patch14

**Hugging Face:** [OpenSportsLab/trained-clip-vit-large-patch14](https://huggingface.co/OpenSportsLab/trained-clip-vit-large-patch14)

This repository supplies `model.pth.tar` for the CLIP ViT-L/14 visual encoder
used by the X-VARS configuration. **Training source: Not reported. Reported
scores: Not reported.**

The canonical `opensportslib/configs/vqa/xvars.yaml` selects the checkpoint and
its corresponding architecture through these component settings:

```yaml
MODEL:
  components:
    video_encoder:
      load:
        weights_path: OpenSportsLab/trained-clip-vit-large-patch14
      params:
        vision_tower: openai/clip-vit-large-patch14
```

Apply this fragment to the X-VARS config. Its visual-checkpoint resolver downloads
`model.pth.tar` from the repository; `vision_tower` identifies the architecture
and image processor used with those weights. The encoder supplies video features
to the VQA pipeline. See [X-VARS setup and artifact flow](tools/vqa.md) for feature
extraction and index preparation.
