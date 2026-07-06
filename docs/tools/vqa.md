# X-VARS Setup and Artifact Flow

Date: July 6, 2026
Canonical dataset: `/home/vorajv/dataset/OSL-XFoul`
Canonical config: `/home/vorajv/opensportslib/opensportslib/configs/vqa/xvars.yaml`
X-VARS reference root: `/home/vorajv/X-VARS/X-VARS`

This page is the source of truth for using the OpenSportsLib `vqa-xvars`
backend. It covers the required weights, how to download them, how to extract
X-VARS CLIP features, how to build the feature and prediction indexes, and how
those artifacts map into the OpenSportsLib VQA config.

## X-VARS setup

The X-VARS-specific part is the setup:

- downloading the X-VARS visual and decoder weights
- optionally pre-extracting CLIP feature pickles
- building `feature_index.json` and `prediction_index.json`

When `feature_source: indexed_or_raw_clip` is used, X-VARS `infer()` prefers
indexed features when available and can fall back to extracting CLIP features
from raw video on the fly. Pre-extracted features remain the preferred path for
speed, parity, and reproducibility.

## Required Artifacts

The X-VARS backend relies on four artifact groups:

- `14_model.pth.tar`
  This is the strict X-VARS visual encoder checkpoint used for feature
  extraction and for the visual encoder weight path in the config.
- `base_model_videoChatGPT`
  This is the decoder backbone used by the X-VARS backend. It must be a
  directory-style Hugging Face model bundle, not a single checkpoint file.
- `feature_index.json`
  Maps OSL sample ids to extracted `PRE_CLIP_feature_clip_{1,2,3}.pkl` files.
- `prediction_index.json`
  Maps OSL sample ids to referee-prior fields derived from OSL labels.

## Download Weights

The local X-VARS repo points to a shared Google Drive folder for the upstream
artifacts:

- Shared drive folder:
  `https://drive.google.com/drive/folders/1UbMAQVFrTB-DtEFUSmv8tBXuurrBfMUJ?usp=sharing`
- X-VARS training reference:
  `/home/vorajv/X-VARS/X-VARS/training.md`
- X-VARS demo reference:
  `/home/vorajv/X-VARS/X-VARS/demo.md`

Download these artifacts from that drive:

- `14_model.pth.tar`
- `base_model_videoChatGPT`

If `base_model_videoChatGPT` is downloaded as a zip archive, extract it so the
final result is a directory containing files such as:

- `config.json`
- `model-00001-of-00003.safetensors`
- `model.safetensors.index.json`
- tokenizer files

## Recommended Local Layout

The OpenSportsLib X-VARS config currently expects the following local layout:

```text
/home/vorajv/X-VARS/weights/
├── 14_model.pth.tar
└── base_model_videoChatGPT/
    ├── config.json
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    ├── model-00003-of-00003.safetensors
    ├── model.safetensors.index.json
    └── tokenizer files...
```

Use your own local paths if needed, but then update the config accordingly.

## How These Paths Map Into `xvars.yaml`

The canonical X-VARS config is:

- `opensportslib/configs/vqa/xvars.yaml`

These fields consume the X-VARS artifacts:

- `MODEL.components.video_encoder.load.weights_path`
  Points to `14_model.pth.tar`
- `MODEL.components.llm_decoder.params.repo_id`
  Points to the `base_model_videoChatGPT` directory
- `DATA.common.feature_index`
  Points to `feature_index.json`
- `DATA.common.prediction_index`
  Points to `prediction_index.json`

In the current local config, those values are:

```yaml
DATA:
  common:
    feature_index: /home/vorajv/dataset/OSL-XFoul/feature_index.json
    prediction_index: /home/vorajv/dataset/OSL-XFoul/prediction_index.json

MODEL:
  components:
    video_encoder:
      load:
        weights_path: /home/vorajv/X-VARS/weights/14_model.pth.tar
    llm_decoder:
      params:
        repo_id: /home/vorajv/X-VARS/weights/base_model_videoChatGPT
```

## Artifact Flow

The artifact pipeline is:

1. Download `14_model.pth.tar` and `base_model_videoChatGPT`
2. Use `14_model.pth.tar` with `tools/convert/extract_xvars_clip_features.py`
   to create `PRE_CLIP_feature_clip_{1,2,3}.pkl`
3. Use `tools/convert/build_xvars_indexes.py` to build:
   - `feature_index.json`
   - `prediction_index.json`
4. Point `opensportslib/configs/vqa/xvars.yaml` at those paths
5. Run `infer()` or `train()` with the X-VARS backend

`base_model_videoChatGPT` is used by the decoder backbone. It is not consumed by
the feature extraction or index builder scripts.

## Step 1: Extract X-VARS CLIP Features

OpenSportsLib provides:

- `tools/convert/extract_xvars_clip_features.py`

This script writes:

- `PRE_CLIP_feature_clip_1.pkl`
- `PRE_CLIP_feature_clip_2.pkl`
- `PRE_CLIP_feature_clip_3.pkl`

under split and sample-id directories.

### CLI arguments

- `--dataset-root`
  Root path containing split JSON files and dataset media
- `--dataset-output-root`
  Root path where extracted feature pickle files are written
- `--splits`
  Splits to process, defaults to `train valid test`
- `--mode`
  `strict_xvars` or `clip_compat`
- `--vision-model`
  Vision backbone, defaults to `openai/clip-vit-large-patch14`
- `--weights-path`
  Required for `strict_xvars`; path to `14_model.pth.tar`
- `--start-frame`
  Strict X-VARS crop start frame
- `--end-frame`
  Strict X-VARS crop end frame
- `--target-fps`
  Strict X-VARS target fps after crop
- `--source-fps`
  Strict X-VARS source fps used for resampling
- `--max-frames`
  Maximum frames for extraction
- `--max-samples`
  Limit samples for smoke testing, `0` means all
- `--overwrite`
  Rewrite feature files if they already exist

### Strict X-VARS example with current local paths

```bash
python tools/convert/extract_xvars_clip_features.py \
  --mode strict_xvars \
  --weights-path /home/vorajv/X-VARS/weights/14_model.pth.tar \
  --dataset-root /home/vorajv/dataset/OSL-XFoul \
  --dataset-output-root /home/vorajv/dataset/OSL-XFoul \
  --splits train valid test \
  --start-frame 63 \
  --end-frame 87 \
  --target-fps 17 \
  --source-fps 25
```

### Generic placeholder example

```bash
python tools/convert/extract_xvars_clip_features.py \
  --mode strict_xvars \
  --weights-path /path/to/14_model.pth.tar \
  --dataset-root /path/to/OSL-XFoul \
  --dataset-output-root /path/to/OSL-XFoul \
  --splits train valid test
```

### Output contract

- `strict_xvars` must produce features with shape `300 x 1024`
- `clip_compat` must produce features with shape `356 x 1024`

## Step 2: Build `feature_index.json` and `prediction_index.json`

OpenSportsLib provides:

- `tools/convert/build_xvars_indexes.py`

This script reads the OSL annotations and extracted feature pickle layout and
writes:

- `feature_index.json`
- `prediction_index.json`

### What each index contains

- `feature_index.json`
  Maps `id` to one or more `feature_paths` pointing at
  `PRE_CLIP_feature_clip_{1,2,3}.pkl`
- `prediction_index.json`
  Maps `id` to referee prior fields like action, offence, and severity derived
  from the OSL labels

### CLI arguments

- `--dataset-root`
  Root path containing train/valid/test JSON files
- `--features-root`
  Root path where the extracted feature pickle files live
- `--output-dir`
  Directory where `feature_index.json` and `prediction_index.json` are written
- `--emit-expected-paths`
  Bootstrap mode that writes expected feature paths even when the `.pkl` files
  do not exist yet

### Example with current local paths

```bash
python tools/convert/build_xvars_indexes.py \
  --dataset-root /home/vorajv/dataset/OSL-XFoul \
  --features-root /home/vorajv/dataset/OSL-XFoul \
  --output-dir /home/vorajv/dataset/OSL-XFoul
```

### Generic placeholder example

```bash
python tools/convert/build_xvars_indexes.py \
  --dataset-root /path/to/OSL-XFoul \
  --features-root /path/to/OSL-XFoul \
  --output-dir /path/to/OSL-XFoul
```

### Bootstrap mode example

```bash
python tools/convert/build_xvars_indexes.py \
  --dataset-root /path/to/OSL-XFoul \
  --features-root /path/to/OSL-XFoul \
  --output-dir /path/to/OSL-XFoul \
  --emit-expected-paths
```

Without `--emit-expected-paths`, the script exits with an error if some samples
have no feature pickle files.

## Inference Behavior

For X-VARS specifically:

- `feature_source: indexed_or_raw_clip` prefers indexed CLIP features when they
  are available through `feature_index.json`
- if indexed features are missing, `infer()` can extract CLIP features from raw
  video on the fly
- pre-extracted indexed features are still recommended for reproducible parity
  and faster runtime

## Upstream-style Prediction Export

For X-VARS-style export rows, use:

```python
predictions = model.infer(test_set="/path/to/test.json")
model.save_predictions("xvars_predictions.json", predictions, output_format="xvars")
```
