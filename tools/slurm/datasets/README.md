# Dataset Download Jobs (Ibex)

This folder contains SLURM `sbatch` jobs to download common OpenSportsLab datasets to:

- `/ibex/project/c2134/opensportslab/datasets`

Available jobs:

- `download_osl_xfoul.sbatch`
  - Dataset: `OpenSportsLab/OSL-XFoul`
  - Revision: `main-parquet`
  - Access: public
- `download_gar_tracking.sbatch`
  - Dataset: `OpenSportsLab/soccernetpro-classification-GAR`
  - Revision: `tracking-parquet`
  - Access: gated (HF auth + accepted conditions required)
- `download_gar_frames.sbatch`
  - Dataset: `OpenSportsLab/soccernetpro-classification-GAR`
  - Revision: `frames-parquet`
  - Access: gated (HF auth + accepted conditions required)

- `download_hf_repo.sbatch`
  - Generic: download a **full HuggingFace repo snapshot** (dataset branch/tag/commit)
  - Usage: `sbatch tools/slurm/datasets/download_hf_repo.sbatch <REPO_ID> [REVISION] [OUTPUT_DIR] [HF_TOKEN]`
  - Required: `REPO_ID`
  - Optional: `REVISION` (default `main`), `OUTPUT_DIR`, `HF_TOKEN`

- `upload_hf_repo.sbatch`
  - Generic: upload a local OSL JSON split to a HuggingFace dataset repo
  - Usage: `sbatch tools/slurm/datasets/upload_hf_repo.sbatch <REPO_ID> <REVISION> <JSON_PATH> [FORMAT] [SHARD_SIZE]`
  - Required: `REPO_ID`, `REVISION`, `JSON_PATH`
  - Optional:
    - `FORMAT` (default `json`, choices: `json|parquet`)
    - `SHARD_SIZE` (default `1GB`, used for `parquet`)

Run examples:

```bash
sbatch tools/slurm/datasets/download_osl_xfoul.sbatch
sbatch tools/slurm/datasets/download_gar_tracking.sbatch
sbatch tools/slurm/datasets/download_gar_frames.sbatch

# Generic — any repo/version:
#   sbatch download_hf_repo.sbatch <REPO_ID> [REVISION] [OUTPUT_DIR] [HF_TOKEN]
sbatch tools/slurm/datasets/download_hf_repo.sbatch OpenSportsLab/OSL-XFoul main-parquet

# Gated repo with a token:
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-classification-GAR tracking-parquet \
  /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR/tracking-parquet \
    hf_xxx

# Upload dataset
sbatch tools/slurm/datasets/upload_hf_repo.sbatch \
  OpenSportsLab/OSL-SoccerNet 720p-parquet \
  /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/720p/train.json \
  json 20GB
sbatch tools/slurm/datasets/upload_hf_repo.sbatch \
  OpenSportsLab/OSL-SoccerNet 720p-parquet \
  /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/720p/valid.json \
  json 20GB
sbatch tools/slurm/datasets/upload_hf_repo.sbatch \
  OpenSportsLab/OSL-SoccerNet 720p-parquet \
  /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/720p/test.json \
  json 20GB
sbatch tools/slurm/datasets/upload_hf_repo.sbatch \
  OpenSportsLab/OSL-SoccerNet 720p-parquet \
  /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/720p/challenge.json \
  json 20GB



```

### SoccerNet datasets

```bash
# soccernetpro-localization-snas  (branch: ResNET_PCA512)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-localization-snas \
    ResNET_PCA512 \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/ResNET_PCA512
    
# soccernetpro-localization-snas  (branch: 224p)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-localization-snas \
    224p \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/224p


# soccernetpro-localization-snas  (branch: 720p)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-localization-snas \
    720p \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/720p

# soccernetpro-localization-snbas  (branch: 224p-2023)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-localization-snbas \
    224p-2023 \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snbas/224p-2023

# soccernetpro-classification-vars  (branch: mvfouls)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/soccernetpro-classification-vars \
    mvfouls \
    /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-vars/mvfouls

sbatch tools/slurm/datasets/download_hf_repo.sbatch \
    OpenSportsLab/OSL-XFoul \
    main-parquet \
    /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet

# soccernetpro-classification-GAR  (branch: tracking-parquet)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
  OpenSportsLab/soccernetpro-classification-GAR \
  tracking-parquet \
  /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR/tracking-parquet

# soccernetpro-classification-GAR  (branch: frames-parquet)
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
  OpenSportsLab/soccernetpro-classification-GAR \
  frames-parquet \
  /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR/frames-parquet
```

If the repos are gated, add your token as the 4th argument: `hf_xxx`.

If your cluster requires account charging, uncomment `#SBATCH --account=...` in each script.

