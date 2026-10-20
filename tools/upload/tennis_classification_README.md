---
pretty_name: OSL Tennis Stroke Classification
license: agpl-3.0
language:
  - en
task_categories:
  - video-classification
source_datasets:
  - OpenSportsLab/OSL-loc-tennis-public
size_categories:
  - 10K<n<100K
tags:
  - tennis
  - sports
  - video
  - action-classification
  - opensportslib
  - parquet
  - webdataset
configs:
  - config_name: default
    data_files:
      - split: train
        path: train/metadata.parquet
      - split: validation
        path: valid/metadata.parquet
      - split: test
        path: test/metadata.parquet
---

# OSL Tennis Stroke Classification

**$total short broadcast-video clips**, each labeled with one of four tennis
stroke classes. This is a clip-level classification dataset derived from
the [`swing-data` localization dataset](https://huggingface.co/datasets/$source_repo/tree/$source_commit),
with new train/validation splits that hold out whole source matches.

The release contains RGB videos and classification annotations. Skeletons,
player tracks, ball/racket detections and model weights are not included.
It supports ordinary video classifiers and OpenSportsLib's model-owned frozen
pose/tracking frontend with a trainable graph classifier.

## Splits and labels

| OSL split | Clips | Source matches |
|---|---:|---:|
$split_rows

The classification head is named `action`; labels are single-label strings.
The following class order is used by the supplied OSL configurations:

| Class index | Label | Train | Validation | Test |
|---:|---|---:|---:|---:|
$class_rows

`near_court` and `far_court` describe the player's side in the broadcast view,
not a persistent player identity. `swing` retains the source's coarse category;
this release does not add forehand/backhand or stroke-quality labels.
Bounce annotations are excluded. There is no background/no-action class.

## Clip construction

- Select only the four serve/swing labels above from localization annotations.
- Request $before seconds before and $after seconds after each selected event.
- Trim bounds at the midpoint to neighboring **selected stroke events** and at
  the source video's boundaries. Each clip contains one selected target stroke;
  it may still contain a bounce or other unselected action.
- Cut at decoded frame boundaries and reset the new clip's timestamps to zero.
  Source presentation-time spacing is preserved; no temporal interpolation,
  artificial frame padding or FPS conversion is applied.
- Preserve the decoded source dimensions: **$dimensions pixels**. Original
  annotation headers sometimes report 1920×1080; those are not the media's actual
  dimensions. The converter uses decoded dimensions.
- Encode H.264 MP4 at CRF $crf, without audio. This is a lossy video conversion.

Native source rates are `$rates` frames/s. Measured encoded durations range from
**$shortest to $longest seconds**. Requested windows are at most one second;
frame boundaries and source clock precision can make encoded duration slightly
longer than one second. Exact source frame/PTS bounds and the target's offset
inside the new clip are retained as provenance.

## Match-level split policy

The published localization train and validation sets share source matches.
This release pools those two sets, samples a fraction of whole matches for
validation (`validation_fraction=$fraction`, Python seed `$seed`), and assigns
all clips from each held-out match to validation. Test source matches retain
their published assignment, subject to the invalid-event exclusions below.
No source match appears in more than one split.

Held-out validation matches:

$validation_matches

The full match assignments and class counts are in
[`reports/conversion.json`](reports/conversion.json). These train/validation
splits differ from the published localization benchmark; classification and
spotting scores should not be compared as if they used identical partitions.

## Known invalid source events

**$excluded_count selected events** are excluded because their annotated times
lie outside the decoded source video's duration. No source was repaired or
silently padded. Details and source hashes are preserved in
[`reports/excluded_events.json`](reports/excluded_events.json).

| Source record | Label | Event time (s) | Decoded duration (s) |
|---|---|---:|---:|
$excluded_rows

## Storage format

The dataset uses the existing OpenSportsLib **Parquet + WebDataset** converter
and uploader. MP4 bytes are packed into TAR shards rather than published as
thousands of individual repository files.

```text
README.md
checksums.sha256
reports/
  conversion.json
  excluded_events.json
train/                       # also valid/ and test/
  metadata.parquet
  shard_manifest.parquet
  shards/shard-000000.tar     # additional shards when required by shard size
```

`metadata.parquet` has one row per clip. `sample_id` is the stable ID,
`sample_payload` is the complete OSL sample as a JSON string, `header` holds the
OSL dataset header, and `shard_name`/`sample_index` locate its TAR contents.
`shard_manifest.parquet` maps original relative media paths to `wds_member`
names. For a nine-digit sample key, TAR members are `<key>.json` (annotations)
and `<key>.0.mp4` (the single video input).

Each reconstructed OSL sample has:

```json
{
  "id": "<stable-clip-id>",
  "inputs": [{"type": "video", "path": "clips/train/<stable-clip-id>.mp4", "fps": 29.97}],
  "labels": {"action": {"label": "far_court_serve"}},
  "metadata": {
    "base_video": "<source-match>",
    "source_split": "train",
    "source_record_id": "<source-segment>",
    "source_event_index": 0,
    "source_event": {"head": "tennis_action", "label": "far_court_serve", "position_ms": 4520},
    "conversion": {"event_offset_ms": 480, "source_start_frame": 101}
  }
}
```

This is an abbreviated schema example, not an actual sample. The full conversion
block also includes source video/manifest hashes, time base, original frame/PTS
bounds, requested and actual intervals, dimensions, frame count and duration.

The Hub's default Parquet configuration exposes **metadata**, with splits named
`train`, `validation`, `test`. Hub `validation` corresponds to OSL folder `valid`.
It does not automatically decode the MP4s inside the TAR shards:

```python
import json
from datasets import load_dataset

rows = load_dataset("$repo_id", revision="$revision", split="train")
sample = json.loads(rows[0]["sample_payload"])
print(sample["labels"]["action"]["label"])
```

## Download and train with OpenSportsLib

Run from an installed OpenSportsLib checkout. The OSL downloader reconstructs
ordinary classification JSON and extracts the referenced MP4s:

```python
from pathlib import Path
from opensportslib.tools import download_dataset_split_from_hf
from opensportslib.apis import ClassificationModel, Config

paths = {}
for split in ("train", "valid", "test"):
    result = download_dataset_split_from_hf(
        repo_id="$repo_id",
        revision="$revision",
        split=split,
        output_dir="data/tennis-public",
        download_format="parquet",
    )
    paths[split] = Path(result["json_path"]).resolve()

config = Config.from_file("examples/tennis/classification_skeleton.yaml")
overrides = {}
for split, path in paths.items():
    overrides[f"DATA.common.splits.{split}.annotation_path"] = str(path)
    overrides[f"DATA.common.splits.{split}.source_path"] = str(path.parent)
config.update(training={"epochs": 20, "batch_size": 2, "num_workers": 0},
              overrides=overrides)
model = ClassificationModel(config=config)
best_checkpoint = model.train(use_wandb=False)
```

The skeleton route requires the optional PyG and GPU pose profiles
(`opensportslib setup --pyg` and `opensportslib setup --pose`). It extracts
skeletons online per sampled clip. Detection and pose weights remain frozen;
the graph encoder, temporal neck and classifier are trainable. Clip/view tracking
state resets for each independent training sample. Skeleton caching is optional
and separate from this dataset's media shards.

## Evaluation and limitations

Use the held-out validation set for model selection and the test set for final
evaluation. Report accuracy, balanced accuracy, macro-F1 and per-class results;
swings substantially outnumber serves. This dataset does not establish spotting
performance on continuous broadcasts because it contains selected positive
stroke clips and supplies no background class.

Source resolution is low, especially for far-court joints. Broadcast cuts,
occlusions, camera views and original annotation uncertainty remain present.
No new manual relabeling, pose-quality audit or trained-model accuracy benchmark
is claimed by this release. Ball/racket motion may carry information that a
skeleton-only classifier cannot recover.

## Provenance, validation and license declaration

- Localization source: [`$source_repo`](https://huggingface.co/datasets/$source_repo),
  branch `swing-data`, immutable commit `$source_commit`.
- That dataset identifies
  [`OpenSportsLab/soccernetpro-localization-tennis`](https://huggingface.co/datasets/OpenSportsLab/soccernetpro-localization-tennis)
  as its original videos/annotations source.
- Classification conversion identity: `$identity`.
- Reproduction: `tools/convert/tennis_to_classification.py` in OpenSportsLib,
  with default midpoint trimming, seed $seed, and `--invalid-events skip`.
  Conversion settings, source annotation hashes and release manifest hashes
  are recorded in `reports/conversion.json`.
- Every encoded output was decoded to verify frame count, timestamps, dimensions
  and target inclusion. Publishing verifies byte hashes against the converter's
  receipts, clip metadata, label counts and source-match separation.
  `checksums.sha256` records original relative MP4 paths and their hashes after
  extraction; use the entries belonging to each downloaded split.

The localization dataset's
[source card](https://huggingface.co/datasets/$source_repo/blob/$license_card_commit/README.md)
declares **AGPL-3.0**; this card carries that declaration forward. Original
broadcast footage remains subject to its underlying rights and source terms.
This format conversion does not grant additional rights to the footage.
Credit the original source dataset and its authors when using these derived
clips.
