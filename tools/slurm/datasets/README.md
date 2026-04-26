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

Run examples:

```bash
sbatch tools/slurm/datasets/download_osl_xfoul.sbatch
sbatch tools/slurm/datasets/download_gar_tracking.sbatch
sbatch tools/slurm/datasets/download_gar_frames.sbatch
```

If your cluster requires account charging, uncomment `#SBATCH --account=...` in each script.