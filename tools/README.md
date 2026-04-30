# Tools

Collection of utility scripts for dataset conversion, dataset transfer, and SLURM experiment execution.

Detailed usage is documented in each subfolder README.

```
tools/
├── convert/                                  # OSL JSON ↔ Parquet + WebDataset
│   ├── README.md
│   ├── osl_json_to_parquet_webdataset.py
│   └── parquet_webdataset_to_osl_json.py
├── download/                                 # HuggingFace transfer helpers
│   ├── README.md
│   ├── download_osl_hf.py
│   └── upload_osl_hf.py
└── slurm/                                    # SLURM job scripts
    ├── README.md
    ├── install_environment.sh
    ├── generic/
    │   ├── ibex_salloc.sh
    │   ├── ibex_srun.sh
    │   └── ibex_job.sbatch
    ├── datasets/
    │   ├── README.md
    │   ├── download_osl_xfoul.sbatch
    │   ├── download_gar_tracking.sbatch
    │   └── download_gar_frames.sbatch
    └── training/
        ├── train_classification.sbatch
        └── train_localization.sbatch
```

---

## Folder guides

- See [tools/convert/README.md](convert/README.md) for conversion scripts and examples.
- See [tools/download/README.md](download/README.md) for HuggingFace download/upload scripts.
- See [tools/slurm/README.md](slurm/README.md) for Ibex SLURM workflows (`salloc`, `srun`, `sbatch`).
