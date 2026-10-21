#!/usr/bin/env bash
# Train both Dallas action spotters sequentially, selecting on validation loss.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${OSL_TRAIN_PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data/tennis-dallas-localization}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dallas-spotting-convergence/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
common_args=(--data-root "$DATA_ROOT"
             --epochs "${MAX_EPOCHS:-100}" --patience "${PATIENCE:-10}" --min-delta "${MIN_DELTA:-0.0001}"
             --clip-len "${CLIP_LEN:-64}" --epoch-num-frames "${EPOCH_NUM_FRAMES:-64000}")
if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
    common_args+=(--config-only)
fi

printf 'Dallas spotting outputs: %s\n' "$OUTPUT_ROOT"
# Training samples full-resolution 64-frame broadcast windows. Classification
# caches contain independently reset 16-frame clips and cannot substitute here.
"$PYTHON_BIN" tools/train/train_tennis_dallas_spotting.py skeleton "${common_args[@]}" \
    --output "$OUTPUT_ROOT/skeleton" --device "${SKELETON_DEVICE:-cuda}" \
    --batch-size "${SKELETON_BATCH_SIZE:-4}" --num-workers "${SKELETON_WORKERS:-0}" \
    --cache-mode "${SKELETON_CACHE_MODE:-off}" \
    --cache-dir "${SKELETON_CACHE_DIR:-$DATA_ROOT/pose-cache-clips}"

"$PYTHON_BIN" tools/train/train_tennis_dallas_spotting.py video "${common_args[@]}" \
    --output "$OUTPUT_ROOT/video" --device "${VIDEO_DEVICE:-cuda}" \
    --batch-size "${VIDEO_BATCH_SIZE:-2}" --num-workers "${VIDEO_WORKERS:-0}" \
    --video-backbone "${VIDEO_BACKBONE:-rny008_gsm}"

printf 'Outputs: %s/{skeleton,video}\n' "$OUTPUT_ROOT"
