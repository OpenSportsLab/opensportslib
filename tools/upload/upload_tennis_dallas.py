"""Publish Dallas classification shards and disjoint localization JSON manifests.

Media upload/conversion uses OSL's shared Hugging Face transfer tools. Dataset
cards do not assign a license absent from the original Dallas repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.convert.dallas_to_classification import SOURCE_REPO
from tools.convert.tennis_to_classification import digest, write_json
from tools.train.train_tennis_dallas import validate_data
from opensportslib.tools import create_dataset_repo_on_hf, upload_dataset_as_parquet_to_hf

CLASSIFICATION_REPO = "OpenSportsLab/OSL-cls-tennis-dallas"


def cards(root, localization):
    report, documents = validate_data(root)
    positives = sum(counts["swing"] for counts in report["class_counts"].values())
    dimensions = sorted({(sample["metadata"]["conversion"]["width"], sample["metadata"]["conversion"]["height"])
                         for document in documents.values() for sample in document["data"]})
    dimensions = ", ".join(f"{width}×{height}" for width, height in dimensions)
    rows = "\n".join(f"| {split} | {report['split_counts'][split]} | "
                      f"{report['class_counts'][split]['swing']} | {report['class_counts'][split]['background']} | "
                      f"{', '.join(report['matches'][split])} |" for split in ("train", "valid", "test"))
    shared = f"""
## Source and split protocol

Source: [{SOURCE_REPO}](https://huggingface.co/datasets/{SOURCE_REPO}/tree/{report['source']['revision']}),
original `swing` branch, pinned revision `{report['source']['revision']}`.
Five Dallas 2026 match broadcasts, **{dimensions}**, approximately 25 frames/s.
Original presentation timestamps are retained, including small frame-rate irregularities.
The train/validation/test partition is by whole game, with no game crossing splits.
Validation selects checkpoints; test is reserved for the final report.

| Split | Classification clips | Swing | Background | Source games |
|---|---:|---:|---:|---|
{rows}

`valid` is the validation split name in OSL JSON and remote folders.
The localization manifests contain the same games and **only original swing events**.

## Rights and limitations

The source repository did not declare a dataset license at the pinned revision.
This conversion does not grant additional rights or assign a new license to the broadcasts.
Serve, near/far-court, player identity and stroke type are not annotated.
Only five matches are present; a single held-out match gives a limited estimate of generalization.

## Reproducibility

Generated using OpenSportsLib `tools/convert/dallas_to_classification.py`.
Seed: 42. Conversion identity: `{report['identity']}`.
See `reports/conversion.json` for source revision, settings, game assignments and counts.
Source annotations are preserved; generated clips use H.264, CRF 18, `veryfast`, no audio.
"""
    classification = f"""---
pretty_name: OSL Dallas Tennis Swing Classification
language: [en]
task_categories: [video-classification]
size_categories: [1K<n<10K]
source_datasets: [OpenSportsLab/OSL-loc-tennis-dallas]
tags: [tennis, sports, video, opensportslib, parquet, webdataset, weak-labels]
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

# Dallas tennis: swing vs background

**{report['expected_clips']:,} short video clips**: {positives:,} annotated swings and {positives:,} inferred background clips.
Class indices: `0 = background`, `1 = swing`; OSL classification label head: `action`.

Positive clips use up to 0.5 seconds before and after a swing, trimmed at midpoints
between neighboring swings. Only covering source frames are encoded; output clips
retain native dimensions and relative source presentation times. Target offsets,
frame boundaries, source video hashes and annotation provenance are in each JSON record.

Background clips last approximately one second. Candidate centers are on a 1.25-second
grid. Centers are at least 2.6 seconds from every annotated swing, including half a
second of context and 0.1 seconds of frame-rounding margin. The actual encoded
window is checked to remain at least **two seconds from all annotated swings**.
One background clip is sampled per positive in each game, with seed 42.
**Background labels are weak labels inferred from gaps, not manually verified negatives.**
Replays, transitions, spectators and unannotated strokes may appear in these gaps.
This release measures balanced swing/background recognition, not four-way stroke classification.

## Files and download

`train.json`, `valid.json`, `test.json` are ordinary OSL v2 classification manifests.
Their MP4 paths are logical paths inside each split's WebDataset TAR shards;
they are restored by OSL's Parquet downloader. This avoids thousands of loose Hub files.
Each split has `metadata.parquet`, `shard_manifest.parquet`, `dataset_info.json`, and `shards/*.tar`.
`checksums.sha256` records the original generated MP4 checksums.

```python
from opensportslib.tools import download_dataset_split_from_hf

for split in ("train", "valid", "test"):
    download_dataset_split_from_hf(
        "OpenSportsLab/OSL-cls-tennis-dallas", "main", split,
        "data/tennis-dallas-hf", download_format="parquet")
```
Use the returned local `annotation_path`; downloaded splits need not live directly at the chosen root.
RGB or frozen skeleton extraction + graph classification uses the same video clips.
""" + shared
    localization_card = f"""---
pretty_name: OSL Dallas Tennis Swing Localization
language: [en]
tags: [tennis, video, action-spotting, opensportslib]
size_categories: [1K<n<10K]
---

# Dallas tennis swing localization

Five match broadcasts with **{positives:,} point annotations** of a single class, `swing`.
Label head: `swing`; times are milliseconds relative to the source video.
Original data remains on the **`swing` branch**, including `swing.json` and the five MP4s.
New `train.json`, `valid.json`, and `test.json` manifests on that branch partition the games
three/one/one, preserving every original event. No inferred background events are added.
Counts: train {report['class_counts']['train']['swing']:,} swings,
validation {report['class_counts']['valid']['swing']:,}, test {report['class_counts']['test']['swing']:,}.

Classification companion: [OSL-cls-tennis-dallas](https://huggingface.co/datasets/OpenSportsLab/OSL-cls-tennis-dallas).
That dataset adds weak background clips; its game assignments match these manifests exactly.

```python
from opensportslib.tools import download_dataset_split_from_hf

for split in ("train", "valid", "test"):
    download_dataset_split_from_hf(
        "OpenSportsLab/OSL-loc-tennis-dallas", "swing", split,
        "data/tennis-dallas-localization", download_format="json")
```
""" + shared
    (root / "README.md").write_text(classification)
    (localization / "README.md").write_text(localization_card)
    return report, documents


def publish(root, localization, *, check_only=False):
    from huggingface_hub import HfApi, CommitOperationAdd
    from opensportslib.tools import upload_dataset_inputs_from_json_to_hf

    root, localization = Path(root).resolve(), Path(localization).resolve()
    report, documents = cards(root, localization)
    checksums = []
    for split, document in documents.items():
        for sample in document["data"]:
            path = root / sample["inputs"][0]["path"]
            receipt = json.loads((root / "receipts" / f"{sample['id']}.json").read_text())
            checksum = digest(path)
            if receipt["identity"] != report["identity"] or receipt["output_sha256"] != checksum:
                raise ValueError(f"Clip checksum mismatch: {sample['id']}")
            checksums.append(f"{checksum}  {sample['inputs'][0]['path']}\n")
        local = json.loads((localization / f"{split}.json").read_text())
        if sorted(record["id"] for record in local["data"]) != sorted(report["matches"][split]):
            raise ValueError("Localization games differ from the classification split")
        if sum(len(record["events"]) for record in local["data"]) != report["class_counts"][split]["swing"]:
            raise ValueError("Localization event counts differ from positive clips")
    (root / "checksums.sha256").write_text("".join(sorted(checksums)))
    write_json(root / "reports/conversion.json", report)
    if check_only:
        print("Checks passed; cards prepared; nothing uploaded.", flush=True)
        return
    create_dataset_repo_on_hf(CLASSIFICATION_REPO, progress_cb=print)
    api, published = HfApi(), {}
    for split in ("train", "valid", "test"):
        published[f"classification_{split}"] = upload_dataset_as_parquet_to_hf(
            CLASSIFICATION_REPO, str(root / f"{split}.json"), split=split, revision="main",
            shard_mode="size", shard_size="1GB", progress_cb=print)
        published[f"localization_{split}"] = upload_dataset_inputs_from_json_to_hf(
            SOURCE_REPO, str(localization / f"{split}.json"), split=split, revision="swing", progress_cb=print)
    operations = [CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(root / relative))
                  for relative in ("README.md", "train.json", "valid.json", "test.json", "checksums.sha256", "reports/conversion.json")]
    published["classification_docs"] = str(api.create_commit(repo_id=CLASSIFICATION_REPO, repo_type="dataset", revision="main",
        operations=operations, commit_message="Document balanced Dallas clips and match-disjoint manifests"))
    for revision in ("swing", "main"):
        published[f"localization_docs_{revision}"] = str(api.create_commit(repo_id=SOURCE_REPO, repo_type="dataset", revision=revision,
            operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(localization / "README.md")),
                        CommitOperationAdd(path_in_repo="reports/conversion.json", path_or_fileobj=str(root / "reports/conversion.json"))],
            commit_message="Document Dallas match-disjoint training, validation and test splits"))
    write_json(root / "reports/upload.json", published)
    print(json.dumps(published, indent=2, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tennis-dallas-classification"))
    parser.add_argument("--localization-root", type=Path, default=Path("data/tennis-dallas-localization"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    publish(args.data_root, args.localization_root, check_only=args.check_only)


if __name__ == "__main__":
    main()
