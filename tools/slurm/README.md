# Tools for Running Experiments

This folder provides practical helpers to run experiments, especially on SLURM clusters like Ibex.

## SLURM (Ibex) quick start

Scripts are organised under `tools/slurm/`:

```
tools/slurm/
├── install_environment.sh          # create the conda env on a compute node
├── generic/                        # reusable allocation & job templates
│   ├── ibex_salloc.sh              #   start an interactive allocation
│   ├── ibex_srun.sh                #   one-shot srun wrapper
│   └── ibex_job.sbatch             #   generic batch job
├── datasets/                       # dataset download jobs
│   ├── README.md
│   ├── download_osl_xfoul.sbatch
│   ├── download_gar_tracking.sbatch
│   ├── download_gar_frames.sbatch
│   └── download_hf_repo.sbatch
└── training/                       # experiment-specific training jobs
    ├── train_classification.sbatch
    └── train_localization.sbatch
```

Create logs directory before running:

```bash
mkdir -p ibex_logs
```

### 0) Install the environment (once)

```bash
sbatch tools/slurm/install_environment.sh
```

### 1) Interactive allocation (`salloc`)

```bash
bash tools/slurm/generic/ibex_salloc.sh
```

Then once the node is allocated:

```bash
srun --jobid=$SLURM_JOB_ID --pty bash -l
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opensportslib
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 2) One-shot run (`srun`)

```bash
bash tools/slurm/generic/ibex_srun.sh
```

### 3) Generic batch job (`sbatch`)

```bash
sbatch tools/slurm/generic/ibex_job.sbatch
```

### 4) Dataset download jobs (`sbatch`)

```bash
# OSL-XFoul -> /ibex/project/c2134/opensportslab/datasets
sbatch tools/slurm/datasets/download_osl_xfoul.sbatch

# GAR tracking-parquet (gated)
sbatch tools/slurm/datasets/download_gar_tracking.sbatch

# GAR frames-parquet (gated)
sbatch tools/slurm/datasets/download_gar_frames.sbatch


# Generic full repo download
# usage: sbatch download_hf_repo.sbatch <REPO_ID> [REVISION] [OUTPUT_DIR] [HF_TOKEN]
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/OSL-XFoul \
    main-parquet \
    /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet

# SoccerNet localization SNAS (224p)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-localization-snas \
    224p \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/224p

# SoccerNet localization SNAS (ResNET_PCA512)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-localization-snas \
    ResNET_PCA512 \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/ResNET_PCA512

# SoccerNet localization SNBAS (224p-2023)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-localization-snbas \
    224p-2023 \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snbas/224p-2023

# SoccerNet classification VARS (mvfouls)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-classification-vars \
    mvfouls \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-vars/mvfouls

# SoccerNet classification GAR (gated, tracking-parquet)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-classification-GAR \
    tracking-parquet \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR/tracking-parquet 

# SoccerNet classification GAR (gated, frames-parquet)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-classification-GAR \
    frames-parquet \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR/frames-parquet
```

See `tools/slurm/datasets/README.md` for dataset details.

### 5) Training jobs (`sbatch`)

```bash
# classification
sbatch tools/slurm/training/train_classification.sbatch

# localization
sbatch tools/slurm/training/train_localization.sbatch
```

Monitor and manage jobs:

```bash
squeue -u "$USER"
scontrol show job <job_id>
scancel <job_id>
```

## Customize for your experiment

- Edit the `python ...` line in any training script to point to your entrypoint.
- Dataset download scripts write to `/ibex/project/c2134/opensportslab/datasets` by default.
- Adjust `--time`, `--mem`, GPU counts to match your workload.
- Uncomment and set `--account` if your project requires charge codes.
