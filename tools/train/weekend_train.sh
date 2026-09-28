#!/usr/bin/env bash
# Run the five full training configs in sequence. Start this inside tmux.
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON_BIN="${PYTHON_BIN:-python}"
export OSL_DATA_ROOT="${OSL_DATA_ROOT:-/home/giancos/OSLdata}"
START_AT="${START_AT:-1}"
mkdir -p weekend_runs "$OSL_DATA_ROOT"

if [[ ! "$START_AT" =~ ^[1-5]$ ]]; then
  echo "START_AT must be an integer from 1 to 5" >&2
  exit 2
fi


# 1. GIN + positional edges on SoccerNet-GAR tracking
echo '=== 1/5: GIN on SoccerNet-GAR tracking ==='
if (( START_AT <= 1 )); then
  "$PYTHON_BIN" -u tools/train/train_config.py classification sngar_tracking_hf.yaml
fi


# 2. MViTv2-S on OSL-XFoul video
echo '=== 2/5: MViTv2-S on OSL-XFoul video ==='
if (( START_AT <= 2 )); then
for split in train valid test; do
  annotation="$OSL_DATA_ROOT/xfoul/224p/$split/$split.json"
  if [[ -f "$annotation" ]]; then
    echo "Reusing $annotation"
    continue
  fi
  "$PYTHON_BIN" tools/download/download_osl_hf.py \
    --repo-id OpenSportsLab/OSL-XFoul \
    --revision 224p \
    --split "$split" \
    --format parquet \
    --output-dir "$OSL_DATA_ROOT/xfoul"
done
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import yaml
from opensportslib.core.config.editable import Config

cfg = Config.from_file("opensportslib/configs/classification/video.yaml").get_config()
root = Path(os.environ["OSL_DATA_ROOT"]) / "xfoul" / "224p"
cfg["DATA"]["common"]["data_root"] = str(root)
for split in ("train", "valid", "test"):
    item = cfg["DATA"]["common"]["splits"][split]
    item["annotation_path"] = str(root / split / f"{split}.json")
    item["source_path"] = str(root / split)
cfg["SYSTEM"]["paths"]["save_dir"] = "./weekend_runs/checkpoints_xfoul"
cfg["SYSTEM"]["paths"]["work_dir"] = "./weekend_runs/checkpoints_xfoul"
Path("weekend_runs/xfoul_mvit.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
"$PYTHON_BIN" -u tools/train/train_config.py classification weekend_runs/xfoul_mvit.yaml
fi


# 3. VideoMAEv2-Base on SoccerNet-GAR frames
echo '=== 3/5: VideoMAEv2-Base on SoccerNet-GAR frames ==='
if (( START_AT <= 3 )); then
for split in train valid test; do
  annotation="$OSL_DATA_ROOT/sngar_frames/frames/$split/$split.json"
  if [[ -f "$annotation" ]]; then
    echo "Reusing $annotation"
    continue
  fi
  "$PYTHON_BIN" tools/download/download_osl_hf.py \
    --repo-id OpenSportsLab/SoccerNet-GAR \
    --revision frames \
    --split "$split" \
    --format parquet \
    --output-dir "$OSL_DATA_ROOT/sngar_frames"
done
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import yaml
from opensportslib.core.config.editable import Config

cfg = Config.from_file("opensportslib/configs/classification/sngar_frames.yaml").get_config()
root = Path(os.environ["OSL_DATA_ROOT"]) / "sngar_frames" / "frames"
cfg["DATA"]["common"]["data_root"] = str(root)
for split in ("train", "valid", "test"):
    item = cfg["DATA"]["common"]["splits"][split]
    item["annotation_path"] = str(root / split / f"{split}.json")
    item["source_path"] = str(root / split)
cfg["SYSTEM"]["paths"]["save_dir"] = "./weekend_runs/checkpoints_frames"
cfg["SYSTEM"]["paths"]["work_dir"] = "./weekend_runs/checkpoints_frames"
Path("weekend_runs/gar_frames_videomae.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
"$PYTHON_BIN" -u tools/train/train_config.py classification weekend_runs/gar_frames_videomae.yaml
fi


# 4. GraphConvSeq + GRU on SN-GAR action spotting tracking
echo '=== 4/5: GraphConvSeq on SN-GAR action spotting tracking ==='
if (( START_AT <= 4 )); then
  "$PYTHON_BIN" -u tools/train/train_config.py localization sngar_spotting_tracking_hf.yaml
fi


# 5. RNY008-GSM + GRU on SN-GAR action spotting video
echo '=== 5/5: RNY008-GSM on SN-GAR action spotting video ==='
if (( START_AT <= 5 )); then
  "$PYTHON_BIN" -u tools/train/train_config.py localization sngar_spotting_video_hf.yaml
fi

echo 'All five training runs finished.'
