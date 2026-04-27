#!/bin/bash
#SBATCH --job-name=osl_env
#SBATCH --output=ibex_logs/osl_env_%j.out
#SBATCH --error=ibex_logs/osl_env_%j.err
#SBATCH --partition=batch
#SBATCH --gpus=v100:1
#SBATCH --mem=90G
#SBATCH --time=3:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
##SBATCH --account=conf-neurips-2026.05.15-ghanembs

mkdir -p ibex_logs

nvidia-smi

module load cuda
module list

nvidia-smi


source ~/miniconda3/etc/profile.d/conda.sh
conda create -n opensportslib -y python=3.12
conda activate opensportslib


pip install -e .


# Install PyTorch (CPU/GPU auto-detected)
opensportslib setup

# Optional: install PyTorch Geometric support
opensportslib setup --pyg

# Optional: install for DALI support
opensportslib setup --dali