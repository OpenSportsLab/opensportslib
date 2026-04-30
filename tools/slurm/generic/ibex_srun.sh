#!/usr/bin/env bash

mkdir -p ibex_logs

srun \
  --job-name=osl_run \
  --partition=batch \
  --output=ibex_logs/osl_run_%j.out \
  --error=ibex_logs/osl_run_%j.err \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=6 \
  --gpus=v100:1 \
  --mem=90G \
  --time=3:59:00 \
  bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate opensportslib && nvidia-smi && python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"'
# Optional account flag:
# --account=conf-neurips-2026.05.15-ghanembs
