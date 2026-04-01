#!/bin/bash
#SBATCH --job-name=ssl_mae_200ep
#SBATCH --output=/home/alkhudmk/opensportslib_ssl/opensportslib/slurm_logs/ssl_mae_%j.out
#SBATCH --error=/home/alkhudmk/opensportslib_ssl/opensportslib/slurm_logs/ssl_mae_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

module load python/3.11.0
module load cuda/12.2

export LD_LIBRARY_PATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/alkhudmk/opensportslib_ssl/opensportslib:${PYTHONPATH}"

PROJECT_DIR="/home/alkhudmk/opensportslib_ssl/opensportslib"
VENV_DIR="/ibex/scratch/alkhudmk/opensportslib_venv"
PYTHON="${VENV_DIR}/bin/python"

cd /home/alkhudmk

echo "=========================================="
echo "SSL MAE Pre-training - 200 epochs"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

${PYTHON} -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

from opensportslib import model

api = model.pretraining(
    config='${PROJECT_DIR}/opensportslib/config/pretraining/videomae_sngar.yaml',
    data_dir='/ibex/scratch/alkhudmk/sngar-frames/train',
    save_dir='/ibex/scratch/alkhudmk/ssl_checkpoints'
)

best_ckpt = api.train(
    train_dir='/ibex/scratch/alkhudmk/sngar-frames/train',
    valid_dir='/ibex/scratch/alkhudmk/sngar-frames/valid',
    use_ddp=True,
    use_wandb=False
)

print(f'Best checkpoint: {best_ckpt}')
"

echo ""
echo "End time: $(date)"
echo "DONE"
