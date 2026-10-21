#!/usr/bin/env bash
# Retrain both Dallas classifiers, selecting and stopping on validation loss.
# Run from an activated OSL environment. Reuse the prepared Dallas pilot data,
# VideoMAEv2 snapshot, RGB frame cache, and exact-clip skeleton caches.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${OSL_TRAIN_PYTHON:-python}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-0.0001}"
DATA_ROOT="${DATA_ROOT:-data/tennis-dallas-classification}"
PRETRAINED_PATH="${PRETRAINED_PATH:-data/dallas-models/videomaev2-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dallas-convergence/$(date -u +%Y%m%dT%H%M%SZ)-$$}"

common_args=(--data-root "$DATA_ROOT" --epochs "$MAX_EPOCHS"
             --patience "$PATIENCE" --min-delta "$MIN_DELTA")

printf 'Dallas training outputs: %s\n' "$OUTPUT_ROOT"
# Cached graph training is small and avoids constructing a pose backend on CPU.
# Set SKELETON_DEVICE=cuda to extract cache misses on GPU instead.
"$PYTHON_BIN" tools/train/train_tennis_dallas.py skeleton "${common_args[@]}" \
    --output "$OUTPUT_ROOT/skeleton" --device "${SKELETON_DEVICE:-cpu}" \
    --batch-size "${SKELETON_BATCH_SIZE:-8}" --num-workers "${SKELETON_WORKERS:-4}"

# Run sequentially to avoid contention between ONNX pose inference and VideoMAE.
"$PYTHON_BIN" tools/train/train_tennis_dallas.py video "${common_args[@]}" \
    --output "$OUTPUT_ROOT/video" --device "${VIDEO_DEVICE:-cuda}" \
    --pretrained-path "$PRETRAINED_PATH" \
    --batch-size "${VIDEO_BATCH_SIZE:-4}" --num-workers "${VIDEO_WORKERS:-2}"

printf 'Finished. Metrics: %s/{skeleton,video}/results.json\n' "$OUTPUT_ROOT"
