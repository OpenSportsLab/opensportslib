# SLURM Guide: salloc and srun

This guide shows practical SLURM patterns to run OpenSportsLib workloads on Ibex using `salloc`, `srun`, and `sbatch`.

---

## Quick mental model

- `salloc`: reserve resources first, then run commands interactively inside the allocation.
- `srun`: launch a command on allocated resources. It can be used either:
  - inside an existing `salloc` session, or
  - directly (one-shot job step).

Use `salloc` when you want interactive debugging and iteration.
Use direct `srun` when you want a clean one-command run.

Reference Ibex-style defaults used below:

- `--partition=batch`
- `--gpus=v100:1`
- `--cpus-per-task=6`
- `--mem=90G`
- `--time=3:59:00`
- `--nodes=1`
- `--ntasks=1`

---

## 1. Interactive workflow with salloc

### Reserve resources

```bash
salloc \
  --job-name=osl \
  --partition=batch \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=6 \
  --gpus=v100:1 \
  --mem=90G \
  --time=3:59:00
```

Once granted, you are inside an allocation.

### Prepare environment on the allocated node

```bash
cd /path/to/opensportslib
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opensportslib
```

### Run your script with srun

```bash
srun --ntasks=1 nvidia-smi && python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

You can repeat `srun ...` multiple times during the same interactive allocation.

---

## 2. One-shot workflow with direct srun

Use this when you do not need an interactive shell.

```bash
srun \
  --partition=batch \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=6 \
  --gpus=v100:1 \
  --mem=90G \
  --time=3:59:00 \
  --job-name=osl \
  --output=ibex_logs/osl_%j.out \
  --error=ibex_logs/osl_%j.err \
  bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate opensportslib && nvidia-smi && python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"'
```

If your cluster requires account charging, append:

```bash
--account=conf-neurips-2026.05.15-ghanembs
```

---

## 3. Batch workflow with sbatch

Use `sbatch` for queued, non-interactive runs.

```bash
#!/bin/bash
#SBATCH --job-name=osl
#SBATCH --output=ibex_logs/osl_%j.out
#SBATCH --error=ibex_logs/osl_%j.err
#SBATCH --partition=batch
#SBATCH --gpus=v100:1
#SBATCH --mem=90G
#SBATCH --time=3:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
##SBATCH --account=conf-neurips-2026.05.15-ghanembs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opensportslib
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Submit it:

```bash
sbatch tools/slurm/ibex_job.sbatch
```

---

## 4. Multi-GPU run (single node)

If your script supports distributed execution, launch one process per GPU.

```bash
srun \
  --partition=batch \
  --nodes=1 \
  --ntasks=4 \
  --ntasks-per-node=4 \
  --cpus-per-task=6 \
  --gpus=v100:1 \
  --mem=90G \
  --time=3:59:00 \
  bash -lc 'cd /path/to/opensportslib && source ~/miniconda3/etc/profile.d/conda.sh && conda activate opensportslib && torchrun --nproc_per_node=4 your_train_script.py'
```

Notes:

- Keep `--ntasks` aligned with `--nproc_per_node`.
- Start with single-GPU first, then scale up.

---

## 5. Helpful options

- `--output=slurm-%j.out`: save logs to a file.
- `--error=slurm-%j.err`: separate stderr log.
- `--chdir=/path/to/opensportslib`: set working directory.
- `--account=<account>`: required on many clusters.
- `--qos=<qos>`: required on some partitions.

Example:

```bash
srun \
  --partition=batch \
  --account=conf-neurips-2026.05.15-ghanembs \
  --gpus=v100:1 \
  --cpus-per-task=6 \
  --mem=90G \
  --time=3:59:00 \
  --job-name=osl \
  --output=ibex_logs/osl_%j.out \
  --error=ibex_logs/osl_%j.err \
  --chdir=/path/to/opensportslib \
  bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate opensportslib && nvidia-smi && python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"'
```

---

## 6. Monitoring and cleanup

- `squeue -u $USER`: list your queued/running jobs.
- `scontrol show job <job_id>`: inspect job details.
- `scancel <job_id>`: cancel a running or queued job.

For interactive sessions, `exit` releases the `salloc` allocation.

---

## 7. Common issues

- `Requested node configuration is not available`:
  - reduce `--gpus`, `--cpus-per-task`, or `--mem`
  - switch to another partition
- Environment not found in `srun`:
  - use `bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ... && ...'`
- CUDA mismatch:
  - verify environment setup and run `opensportslib setup` in your env.

Cluster policies vary. If your site has required flags, treat them as mandatory defaults in all examples above.