# First Workflow

This template uses the public classification wrapper. OpenSportsLib does not ship media or a ready-to-run training dataset: supply valid video files and OSL JSON manifests before training.

## 1. Select a config and prepare splits

Start with `opensportslib/configs/classification/video.yaml`, or copy it outside the installed package for an experiment. Point `DATA.common.splits.<split>` at your manifests and media roots.

Create `train.json`, `valid.json`, and `test.json` following the [OSL JSON format](../data/osl-json-format.md). A classification sample needs an `id`, a video input, and `labels.action.label`; its label must occur in the root label list.

```json
{"version":"2.0","labels":{"action":{"type":"single_label","labels":["pass","shot"]}},"data":[{"id":"clip-001","inputs":[{"type":"video","path":"clips/clip-001.mp4","fps":25.0}],"labels":{"action":{"label":"pass"}}}]}
```

## 2. Train, predict, and evaluate

```python
from opensportslib.apis import ClassificationModel

model = ClassificationModel(config="/path/to/classification.yaml")
best_checkpoint = model.train(train_set="/path/to/train.json", valid_set="/path/to/valid.json")
predictions = model.infer(test_set="/path/to/test.json", weights=best_checkpoint)
metrics = model.evaluate(test_set="/path/to/test.json", predictions=predictions)
saved_path = model.save_predictions(output_path="/path/to/predictions.json", predictions=predictions)
print(metrics, saved_path)
```

`infer()` returns an in-memory OSL JSON-style payload. `save_predictions()` is the explicit disk-write step. Use `evaluate(..., predictions=saved_path)` to score a saved payload without inference.

## 3. Continue with a task

Use `LocalizationModel` with a localization config for spotting, or choose one VQA dependency profile before following the [VQA guide](../tools/vqa.md). See [workflows](../tni/tni.md) for task-specific arguments and [configuration](../config/configuration-guide.md) for all config keys.
