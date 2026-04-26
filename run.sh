#!/bin/bash
#SBATCH --job-name=opensportslib_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --partition=batch
#SBATCH --gpus=v100:1
#SBATCH --mem=200G
#SBATCH --time=01:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8


source ~/miniconda3/etc/profile.d/conda.sh
conda activate OpenSportsLib
python test.py
