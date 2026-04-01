#!/bin/bash
#SBATCH --job-name=ssl_test
#SBATCH --output=/home/alkhudmk/opensportslib_ssl/opensportslib/slurm_logs/ssl_test_%j.out
#SBATCH --error=/home/alkhudmk/opensportslib_ssl/opensportslib/slurm_logs/ssl_test_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

module load python/3.11.0
module load cuda/12.2

export LD_LIBRARY_PATH=/sw/rl9g/python/3.11.0/rl9_gnu12.2.0/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="/home/alkhudmk/opensportslib_ssl/opensportslib"
VENV_DIR="/ibex/scratch/alkhudmk/opensportslib_venv"
PYTHON="${VENV_DIR}/bin/python"

# Ensure our dev branch code takes priority over the stale venv copy
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

cd /home/alkhudmk

echo "=========================================="
echo "SSL Pre-training Pipeline Test"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

${PYTHON} -c "
import sys
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# =========================================
# Test 1: MAE Pre-training (3 epochs)
# =========================================
print()
print('=' * 50)
print('TEST 1: MAE Pre-training')
print('=' * 50)

from opensportslib import model

api = model.pretraining(
    config='${PROJECT_DIR}/opensportslib/config/pretraining/test_mae.yaml',
    data_dir='/ibex/scratch/alkhudmk/ssl_test_data/train',
    save_dir='/ibex/scratch/alkhudmk/ssl_test_checkpoints'
)

best_ckpt = api.train(
    train_dir='/ibex/scratch/alkhudmk/ssl_test_data/train',
    valid_dir='/ibex/scratch/alkhudmk/ssl_test_data/valid',
    use_ddp=False,
    use_wandb=False
)

print(f'Best checkpoint: {best_ckpt}')

# Verify checkpoint can be loaded
import os
assert best_ckpt is not None, 'No checkpoint returned!'
assert os.path.exists(best_ckpt), f'Checkpoint file not found: {best_ckpt}'
ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
print(f'Checkpoint keys: {list(ckpt.keys())}')
print(f'Checkpoint epoch: {ckpt[\"epoch\"]}')
print(f'State dict keys (first 5): {list(ckpt[\"state_dict\"].keys())[:5]}')
print()
print('TEST 1 PASSED: MAE pre-training completed successfully!')

# =========================================
# Test 2: Model builder for all methods
# =========================================
print()
print('=' * 50)
print('TEST 2: Build all SSL models on GPU')
print('=' * 50)

from opensportslib.models.builder import build_model
from types import SimpleNamespace

device = torch.device('cuda:0')

for method_name, ssl_cfg in [
    ('mae', SimpleNamespace(
        method='mae',
        mae=SimpleNamespace(mask_ratio=0.75, mask_type='tube', norm_pix_loss=True)
    )),
    ('dino', SimpleNamespace(
        method='dino',
        dino=SimpleNamespace(
            momentum=0.996, teacher_temp=0.04, student_temp=0.1,
            center_momentum=0.9, head_hidden_dim=512,
            head_bottleneck_dim=128, head_out_dim=1024
        )
    )),
    ('simclr', SimpleNamespace(
        method='simclr',
        simclr=SimpleNamespace(temperature=0.1, proj_hidden_dim=512, proj_out_dim=128)
    )),
]:
    cfg = SimpleNamespace(
        TASK='pretraining',
        SSL=ssl_cfg,
        DATA=SimpleNamespace(frame_size=[224, 224], patch_size=16, num_frames=16, tubelet_size=2),
        MODEL=SimpleNamespace(
            encoder=SimpleNamespace(type='vit_tiny', embed_dim=192, depth=2, num_heads=3),
            decoder=SimpleNamespace(embed_dim=96, depth=1, num_heads=3)
        )
    )
    m = build_model(cfg, device)
    m = m.to(device)

    x = torch.randn(2, 3, 16, 224, 224, device=device)
    if method_name == 'mae':
        out = m(x)
        loss = ((out['pred'] - out['target'])**2).mean(-1)
        loss = (loss * out['mask']).sum() / out['mask'].sum()
    else:
        out = m([x, x])
        loss = out['loss']

    loss.backward()
    print(f'  {method_name}: loss={loss.item():.4f}, backward OK')

    # test encoder extraction
    enc = m.get_encoder()
    enc_out = enc(x)
    print(f'  {method_name}: encoder output shape={enc_out.shape}')

    del m, out, loss, enc, enc_out
    torch.cuda.empty_cache()

print()
print('TEST 2 PASSED: All SSL models built and tested on GPU!')

# =========================================
# Test 3: Verify existing tasks not broken
# =========================================
print()
print('=' * 50)
print('TEST 3: Verify existing imports still work')
print('=' * 50)

from opensportslib.apis import classification, localization, pretraining
from opensportslib.datasets.builder import build_dataset
from opensportslib.models.builder import build_model
from opensportslib.core.loss.builder import build_criterion
from opensportslib.core.scheduler.builder import build_scheduler
from opensportslib.core.optimizer.builder import build_optimizer
print('  All existing imports OK')
print('TEST 3 PASSED: No regression in existing code!')

print()
print('=' * 50)
print('ALL TESTS PASSED')
print('=' * 50)
"

echo ""
echo "End time: $(date)"
echo "DONE"
