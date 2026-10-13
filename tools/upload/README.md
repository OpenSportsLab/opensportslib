# Model Upload Tools

Scripts to upload OpenSportsLib model artifacts to Hugging Face model repos.

This complements the dataset upload tools in `tools/download/` and supports
both:

- folder uploads with `upload_folder(...)`
- single-file uploads with `upload_file(...)`

## Script

- `upload_model_hf.py`
  - Uploads a model folder or a single checkpoint file to a Hugging Face repo.
  - Automatically creates the target repo if it does not exist.
  - Automatically creates the target branch when `--revision` is not `main`.
  - Optionally uploads extra companion files such as `config.yaml` and
    `README.md`.

## Folder upload

Example: upload a validated Qwen3-VL LoRA adapter and ignore training-only
artifacts.

```bash
python tools/upload/upload_model_hf.py folder \
  --repo-id OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora \
  --folder-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora \
  --commit-message "Upload validated Qwen3 8B VL LoRA adapter" \
  --ignore-pattern "checkpoint-*" \
  --ignore-pattern "*resume*" \
  --ignore-pattern "*sanitized*" \
  --ignore-pattern "native_sft_cache/**" \
  --ignore-pattern "*.pyc" \
  --ignore-pattern "__pycache__/**"
```

Upload the same folder and explicitly attach a config and README:

```bash
python tools/upload/upload_model_hf.py folder \
  --repo-id OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora \
  --folder-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora \
  --config-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora/config.yaml \
  --readme-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora/README.md
```

## File upload

Example: upload a single localization checkpoint file:

```bash
python tools/upload/upload_model_hf.py file \
  --repo-id jeetv/osl-loc-SPL-model \
  --file-path /home/vorajv/opensportslib/checkpoints/train_goal_only/rny002_gsm/2fc8be43/best_checkpoint.pt \
  --path-in-repo model.pt \
  --commit-message "Upload localization checkpoint"
```

Upload the same file and attach a config/model card:

```bash
python tools/upload/upload_model_hf.py file \
  --repo-id jeetv/osl-loc-SPL-model \
  --file-path /home/vorajv/opensportslib/checkpoints/train_goal_only/rny002_gsm/2fc8be43/best_checkpoint.pt \
  --path-in-repo model.pt \
  --config-path /home/vorajv/opensportslib/opensportslib/configs/localization/video_dali.yaml \
  --config-path-in-repo config.yaml \
  --readme-path /home/vorajv/opensportslib/docs/model-zoo.md \
  --readme-path-in-repo README.md
```

## Optional flags

- `--repo-type`
  Defaults to `model`.
- `--revision`
  Upload to a specific branch.
- `--private`
  Create the repo as private if missing.
- `--token`
  Override the local Hugging Face login token.
- `--allow-pattern`
  Repeat to whitelist files during folder upload.
- `--ignore-pattern`
  Repeat to skip files during folder upload.
- `--delete-pattern`
  Repeat to remove matching remote files before folder upload.
- `--extra-file LOCAL=REMOTE`
  Repeat to upload arbitrary companion files after the main upload.

## Notes

- Authenticate first with `huggingface-cli login` unless you pass `--token`.
- For public model repos, prefer excluding training checkpoints, caches, and
  resume artifacts.
- For LoRA repos, upload the adapter folder rather than the entire training run
  directory.

## SoccerNet-GAR full-FPS tracking dataset

`upload_sngar_tracking_full.py` publishes the completed tracking conversion to
`OpenSportsLab/SoccerNet-GAR` on the fixed `tracking-full` branch. It calls the
internal OSL `upload_dataset_as_parquet_to_hf` uploader, retaining sample IDs,
labels, conversion provenance and the original per-clip Parquet bytes.
The input modality remains `tracking_parquet`.

Run from the repository root in the OSL environment:

```bash
# Offline checks only: no Hub access or uploads.
python tools/upload/upload_sngar_tracking_full.py --check-only

# Upload when ready, using your cached Hugging Face login or HF_TOKEN.
python tools/upload/upload_sngar_tracking_full.py --shard-size 1GB
```

The default input is
`/home/giancos/OSLdata/sngar_tracking_classification_full_fps/tracking`;
use `--data-root` to change it and `--repo-id` to select another existing dataset
repository. The script requires a passed `../reports/build.json`, all 87,939
samples in the final `annotations_{train,valid,test}.json` manifests, native-rate
provenance and matching Parquet row counts. Pilot manifests are never uploaded.
Padded missing observations retain their `tracking_valid` flags; observed-only
builds retain their original gaps.

The internal uploader creates or reuses `tracking-full` and produces:

```text
train/metadata.parquet
train/shard_manifest.parquet
train/shards/*.tar
valid/metadata.parquet
valid/shard_manifest.parquet
valid/shards/*.tar
test/metadata.parquet
test/shard_manifest.parquet
test/shards/*.tar
```

Each TAR contains tracking Parquet clips and OSL JSON sidecars. The size target
is approximate. Packaging uses temporary disk space for one whole split at a
time, not only one shard. Set `TMPDIR` to an existing directory with enough free
space for the largest split plus metadata and TAR overhead before running.

Uploads commit one split at a time; a failure can leave earlier splits published.
Retry just the failed/remaining splits with `--splits valid test`. Every run
checks all local splits first. Re-uploading replaces the generated files for the
selected splits and removes stale shards there. The branch helper creates new
branches from the repository's initial commit. The script never targets `videos`.

## SoccerNet-GAR native-FPS videos dataset

`upload_sngar_videos.py` uploads to `OpenSportsLab/SoccerNet-GAR` on the fixed
`videos` branch, using the same internal OSL Parquet/WebDataset uploader:

```bash
# Run after the full video conversion finishes; this command is offline.
python tools/upload/upload_sngar_videos.py --check-only

# Upload when ready, using your cached Hugging Face login or HF_TOKEN.
python tools/upload/upload_sngar_videos.py --shard-size 1GB
```

The default `--data-root` is
`/home/giancos/OSLdata/sngar_classification_rebuilt/video`. The script requires a
passed `../reports/build.json` and all 87,939 samples in the final
`annotations_{train,valid,test}.json` manifests. It checks IDs, native sampling
provenance, temporal alignment results (for temporal builds), and each MP4's
header frame count, fractional frame rate, dimensions and duration. It relies on
the converter's completed decode validation; it does not decode all clips again.
Both temporal and exact conversion modes are supported. Pilot manifests are
never uploaded. Do not modify the conversion outputs during upload.

Each split contains `metadata.parquet`, `shard_manifest.parquet` and `shards/*.tar`.
TARs hold the existing MP4 bytes and OSL JSON sidecars; videos are not re-encoded.
The manifest modality remains `video`, with sample IDs, labels and provenance
preserved. `--repo-id`, `--shard-size` and `--splits valid test` work like the
tracking uploader. All local splits are checked even when retrying selected splits.

Set `TMPDIR` to an existing directory with free space for the largest entire
video split plus TAR/metadata overhead. Packaging is per split, not per shard;
video space requirements can be substantially larger than tracking. Uploads commit
one split at a time, and retries replace that split's generated files and remove
stale shards. A failed run may leave earlier splits published. The branch helper
creates new branches from the repository's initial commit. Only `videos` is targeted.
