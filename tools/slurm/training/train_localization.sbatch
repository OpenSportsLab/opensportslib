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

# --- dataset download (edit paths as needed) ---
# python tools/download_dataset.py --dataset soccernet --dest /path/to/data

# --- training ---
python examples/quickstart/basic_localization.py
