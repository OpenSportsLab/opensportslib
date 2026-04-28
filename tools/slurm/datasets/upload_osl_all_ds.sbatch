#!/bin/bash
#SBATCH --job-name=osl_ul
#SBATCH --output=ibex_logs/osl_ul_%j.out
#SBATCH --error=ibex_logs/osl_ul_%j.err
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

# OpenSportsLab/OSL-SoccerNet
for revision in ResNET_PCA512 224p 720p; do
case "$revision" in
ResNET_PCA512) shard_size="1GB" ;;
224p) shard_size="5GB" ;;
720p) shard_size="20GB" ;;
*) shard_size="5GB" ;;
esac

for split in test valid train challenge; do
python tools/download/upload_osl_hf.py \
	--repo-id OpenSportsLab/OSL-SoccerNet --revision $revision --split $split --format parquet --shard-size $shard_size \
	--json-path /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/$revision/$split.json
done
done
