# Tools

Collection of utility scripts for dataset conversion, Hugging Face transfer,
model upload, and SLURM experiment execution.

Detailed usage is documented in each subfolder README.

```
tools/
├── convert/                                  # dataset builders, OSL JSON ↔ Parquet + WebDataset
│   ├── README.md
│   ├── build_sngar_spotting.py               # raw source → SN-GAR spotting pair
│   ├── sngar_events.py
│   ├── sngar_dataset_card.py
│   ├── verify_sngar_spotting.py
│   ├── osl_json_to_parquet_webdataset.py
│   └── parquet_webdataset_to_osl_json.py
├── download/                                 # HuggingFace dataset transfer helpers
│   ├── README.md
│   ├── download_osl_hf.py
│   ├── push_sngar_spotting.py
│   └── upload_osl_hf.py
├── upload/                                   # Hugging Face model upload helpers
│   ├── README.md
│   └── upload_model_hf.py
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

- See [docs/data/osl-json-format.md](../docs/data/osl-json-format.md) for the OSL JSON schema used by dataset tools.
- See [tools/convert/README.md](convert/README.md) for conversion scripts and examples.
- See [tools/download/README.md](download/README.md) for HuggingFace download/upload scripts.
- See [docs/tools/sngar-spotting.md](../docs/tools/sngar-spotting.md) for the SN-GAR action-spotting dataset build and release.
- See [tools/upload/README.md](upload/README.md) for Hugging Face model upload scripts.
- See [tools/slurm/README.md](slurm/README.md) for Ibex SLURM workflows (`salloc`, `srun`, `sbatch`).
