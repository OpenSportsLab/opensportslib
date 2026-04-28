#!/bin/bash
#SBATCH --job-name=osl_dl
#SBATCH --output=ibex_logs/osl_dl_%j.out
#SBATCH --error=ibex_logs/osl_dl_%j.err
#SBATCH --partition=batch
#SBATCH --mem=90G
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
##SBATCH --account=conf-neurips-2026.05.15-ghanembs

set -euo pipefail

mkdir -p ibex_logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opensportslib


# # OpenSportsLab/OSL-XFoul
# for revision in 224p 720p; do
# for split in test valid train; do
# python tools/download/download_osl_hf.py \
# 	--repo-id OpenSportsLab/OSL-XFoul --revision $revision --split $split --format parquet \
# 	--output-dir /ibex/project/c2134/opensportslab/datasets/OSL-XFoul
# done
# done



# # OpenSportsLab/OSL-SoccerNet
# for revision in ResNET_PCA512 224p 720p; do
# for split in test valid train challenge; do
# python tools/download/download_osl_hf.py \
# 	--repo-id OpenSportsLab/OSL-SoccerNet --revision $revision --split $split --format parquet \
# 	--json-path /ibex/project/c2134/opensportslab/datasets/OSL-SoccerNet
# done
# done



# OpenSportsLab/soccernetpro-classification-GAR
for revision in tracking-parquet frames-parquet; do
for split in test valid train; do
python tools/download/download_osl_hf.py \
        --repo-id OpenSportsLab/soccernetpro-classification-GAR --revision $revision --split $split --format parquet \
        --output-dir /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR
done
done
