# OpenSportsLib APIs

This folder contains the high-level task wrappers used by users of OpenSportsLib.

## Public Entry Points

Use task model classes from `opensportslib.apis`:

- `ClassificationModel(...)`
- `LocalizationModel(...)`
- `VQAModel(...)`

## Shared Base Wrapper

All task wrappers inherit from `BaseTaskModel`, which provides the shared method contract:

- `load_weights(...)`
- `train(...)`
- `infer(...)`
- `evaluate(...)`
- `save_predictions(...)`

## Standard Task Model Methods

Each task model exposes:

- `load_weights(...)`
- `train(...)`
- `infer(...)` (predictions-focused)
- `evaluate(...)` (metrics-focused)

Current behavior:

| Method | Main inputs | Returns | Notes |
| --- | --- | --- | --- |
| `load_weights(weights=...)` | Local checkpoint path or Hugging Face model ID | `None` | Loads weights into the task wrapper. |
| `train(train_set=..., valid_set=...)` | OSL JSON train/validation files | Best checkpoint path or `None` | Split paths can also come from the YAML config. |
| `infer(test_set=...)` | OSL JSON test/inference file | In-memory OSL JSON-style prediction dict | The public API does not require an output path. Use `save_predictions(...)` for explicit persistence. |
| `evaluate(test_set=...)` | OSL JSON test file | Metrics dict | Runs inference first when `predictions` is not provided. |
| `evaluate(test_set=..., predictions=...)` | OSL JSON test file plus prediction dict/path | Metrics dict | Skips inference and evaluates the provided predictions. |
| `save_predictions(output_path=..., predictions=...)` | Prediction dict returned by `infer()` | Saved file path | Explicitly writes an OSL JSON prediction payload to disk. |

Additional weight behavior:

- `ClassificationModel(config=..., weights=...)` uses constructor weights as the default for later `train()` / `infer()` calls.
- `LocalizationModel(config=..., weights=...)` stores constructor weights lazily and loads them on the first `train()` / `infer()` call that needs them.

Annotation and prediction payloads follow the OSL JSON data model. For the full
schema, see the docs page `docs/data/osl-json-format.md`.

## Minimal Usage

```python
from opensportslib.apis import ClassificationModel

m = ClassificationModel(
    config="/path/to/classification.yaml",
    weights=None,  # optional: path or Hugging Face model ID
)

best_ckpt = m.train(
    train_set="/path/to/train.json",
    valid_set="/path/to/valid.json",
)

predictions = m.infer(
    test_set="/path/to/test.json",
)

saved_predictions = m.save_predictions(
    output_path="/path/to/predictions.json",
    predictions=predictions,
)

metrics = m.evaluate(
    test_set="/path/to/test.json",
)
```

## Evaluate Existing Predictions

```python
metrics = m.evaluate(
    test_set="/path/to/test.json",
    predictions="/path/to/predictions.json",
)
```

## Localization Usage

```python
from opensportslib.apis import LocalizationModel

m = LocalizationModel(
    config="/path/to/localization_video_dali.yaml",
    weights=None,  # optional: path or Hugging Face model ID
)

best_ckpt = m.train(
    train_set="/path/to/train.json",
    valid_set="/path/to/valid.json",
)

predictions = m.infer(
    test_set="/path/to/test_annotations.json",
)

saved_predictions = m.save_predictions(
    output_path="/path/to/predictions.json",
    predictions=predictions,
)

metrics = m.evaluate(
    test_set="/path/to/test_annotations.json",
)
```

## VQA Usage

```python
from opensportslib.apis import VQAModel

m = VQAModel(
    config="/path/to/vqa.yaml",
    weights=None,  # optional: path or Hugging Face model ID
)

predictions = m.infer(
    test_set="/path/to/test_annotations.json",
)

single_prediction = m.infer(
    video_path="/path/to/video.mp4",
    question="What card would you give? Why?",
)

metrics = m.evaluate(
    test_set="/path/to/test_annotations.json",
    predictions=predictions,
)
```

### X-VARS Backends

Use `MODEL.metadata.backend: xvars_videochatgpt` with
`TRAIN.execution.training_backend: xvars_videochatgpt_lora` for the
X-VARS-compatible multimodal path. This backend preserves
`video_spatio_temporal_features` during training and injects them into
`<vid_patch>` token positions at inference. In OpenSportsLib, X-VARS parity is
claimed through training, inference, and X-VARS-style prediction export; VQA
evaluation remains OpenSportsLib-native.

For headless parity with the original X-VARS demo, configure the encoder with
`feature_source: indexed_or_raw_clip` and set its `load.weights_path` to
`14_model.pth.tar`. Indexed 300-token features are used when available;
otherwise `infer()` extracts them from the raw video and adds the visual
classifier's action/offence/card prior. Supplying `weights=` optionally applies
a PEFT/LoRA adapter over the configured base VideoChatGPT model.

For upstream-style inference JSON, save VQA predictions with:

```python
m.save_predictions("xvars_predictions.json", predictions, output_format="xvars")
```
