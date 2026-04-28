#!/bin/bash
#SBATCH --job-name=osl_dl
#SBATCH --output=ibex_logs/osl_dl_%j.out
#SBATCH --error=ibex_logs/osl_dl_%j.err
#SBATCH --partition=batch
#SBATCH --mem=90G
#SBATCH --time=3:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
##SBATCH --account=conf-neurips-2026.05.15-ghanembs

set -euo pipefail

mkdir -p ibex_logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opensportslib

pwd

# Gated dataset: ensure HF conditions are accepted and auth is available on the node.
python tools/download/download_osl_hf.py \
  --repo-id OpenSportsLab/soccernetpro-classification-GAR \
  --revision frames-parquet \
  --split test \
  --format parquet \
  --output-dir "/ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR-frames"

python tools/download/download_osl_hf.py \
  --repo-id OpenSportsLab/soccernetpro-classification-GAR \
  --revision frames-parquet \
  --split valid \
  --format parquet \
  --output-dir "/ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR-frames"

python tools/download/download_osl_hf.py \
  --repo-id OpenSportsLab/soccernetpro-classification-GAR \
  --revision frames-parquet \
  --split train \
  --format parquet \
  --output-dir "/ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR-frames"
