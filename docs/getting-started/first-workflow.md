# First Workflow

This example uses the public classification API. OpenSportsLib does not include videos or a ready-to-run training dataset, so before training you need video files and OSL JSON manifest files. A manifest is a JSON file that lists each sample, its input file, and its label.

## 1. Select a config and prepare splits

Start with `opensportslib/configs/classification/video.yaml`. For your own experiment, copy it outside the installed package and edit that copy. Set `DATA.common.splits.<split>` to the paths for your manifest files and video directories.

Create `train.json`, `valid.json`, and `test.json` in the [OSL JSON format](../data/osl-json-format.md). Each classification sample needs an `id`, a video input, and `labels.action.label`. Every label used by a sample must also appear in the root label list.

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

`infer()` returns predictions in memory as an OSL JSON-style dictionary. It does not write a file. Call `save_predictions()` when you want to save those predictions. To score an already saved prediction file without running inference again, use `evaluate(..., predictions=saved_path)`.

## 3. Continue with a task

For action spotting, use `LocalizationModel` with a localization configuration. For VQA, choose a dependency profile before following the [VQA guide](../tools/vqa.md). The [workflow guide](../tni/tni.md) explains task-specific arguments, and the [configuration guide](../config/configuration-guide.md) lists every configuration key.
