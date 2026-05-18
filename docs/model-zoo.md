# Model Zoo

This page lists the pretrained OpenSportsLib models published on Hugging Face.
Use the model repository ID with `load_weights(...)` to load a checkpoint into an
OpenSportsLib model.

## Available Models

| Model | Task | Dataset trained on | Backbone / architecture | Classes / label set | Scores | Hugging Face link | Load weights snippet |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `OSL-cls-action-mvitv2` | Action / Event Classification | SoccerNet - MVFouls classification subset | MViT v2 | Not reported on model card | Accuracy: 0.57<br>Balanced Accuracy: 0.40<br>Top-2: 0.78 | [OpenSportsLab/OSL-cls-action-mvitv2](https://huggingface.co/OpenSportsLab/OSL-cls-action-mvitv2) | `myModel.load_weights(weights="OpenSportsLab/OSL-cls-action-mvitv2")` |
| `OSL-loc-snbas-2023-e2e` | Action Spotting / Localization | SoccerNet - Ball Action Spotting 2023 | E2E, DALI backend | PASS, DRIVE | tight mAP: 71.48<br>loose mAP: 85.62 | [OpenSportsLab/OSL-loc-snbas-2023-e2e](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2023-e2e) | `myModel.load_weights(weights="OpenSportsLab/OSL-loc-snbas-2023-e2e")` |
| `OSL-loc-snbas-2025-e2e` | Action Spotting / Localization | SoccerNet - Ball Action Spotting 2025 | E2E, DALI backend | PASS, DRIVE, HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL | tight mAP: 47.98<br>loose mAP: 58.35 | [OpenSportsLab/OSL-loc-snbas-2025-e2e](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2025-e2e) | `myModel.load_weights(weights="OpenSportsLab/OSL-loc-snbas-2025-e2e")` |

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
