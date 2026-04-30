# OpenSportsLib APIs

This folder contains the high-level task wrappers used by users of OpenSportsLib.

## Public Entry Points

Use task model classes from `opensportslib.apis`:

- `ClassificationModel(...)`
- `LocalizationModel(...)`

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

- `infer()` runs the model on `test_set` and returns predictions directly as an in-memory OSL JSON payload (including confidence scores when provided by the task output format)
- `infer()` does not write predictions to disk
- `evaluate()` runs inference on `test_set` and computes metrics against that same test set ground truth when `predictions` is not provided
- `evaluate(predictions=...)` skips inference and evaluates an in-memory predictions dict or prediction file path directly
- `save_predictions(output_path=..., predictions=...)` saves an OSL JSON predictions payload to a file
- `ClassificationModel(config=..., weights=...)` uses constructor weights as the default for later `train()` / `infer()` calls
- `LocalizationModel(config=..., weights=...)` stores constructor weights lazily and loads them on the first `train()` / `infer()` call that needs them

## Minimal Usage

```python
from opensportslib.apis import ClassificationModel

m = ClassificationModel(
    config="/path/to/classification.yaml",
    weights="/path/to/weights.pt",  # optional
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
    config="/path/to/localization.yaml",
    weights="/path/to/weights.pt",  # optional
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
