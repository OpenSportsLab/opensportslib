#!/bin/bash
#SBATCH --job-name=osl
#SBATCH --output=ibex_logs/osl_%j.out
#SBATCH --error=ibex_logs/osl_%j.err
#SBATCH --partition=batch
#SBATCH --gpus=v100:1
#SBATCH --mem=90G
#SBATCH --time=47:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
##SBATCH --account=conf-neurips-2026.05.15-ghanembs

mkdir -p ibex_logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opensportslib

# python tools/download/download_osl_hf.py \
# 	--url "https://huggingface.co/datasets/OpenSportsLab/OSL-XFoul/tree/main-parquet/test" \
# 	--output-dir /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet \
# 	--types all

# python tools/download/download_osl_hf.py \
# 	--url "https://huggingface.co/datasets/OpenSportsLab/OSL-XFoul/tree/main-parquet/valid" \
# 	--output-dir /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet \
# 	--types all

# python tools/download/download_osl_hf.py \
# 	--url "https://huggingface.co/datasets/OpenSportsLab/OSL-XFoul/tree/main-parquet/train" \
# 	--output-dir /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet \
# 	--types all

python tools/training/classification.py \
    --config tools/slurm/training/configs/classification.yaml \
    --train-set /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet/train.json \
    --valid-set /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet/valid.json \
    --test-set /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet/test.json
