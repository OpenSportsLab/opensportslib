# Tools

Collection of utility scripts for dataset conversion, Hugging Face transfer,
model upload, and SLURM experiment execution.

Detailed usage is documented in each subfolder README.

The [Dallas tennis pilot](../docs/preprocessing/tennis-dallas.md) documents
`convert/dallas_to_classification.py`, `upload/upload_tennis_dallas.py`, and
`train/train_tennis_dallas.py` for match-disjoint localization/classification data
and paired video/graph experiments. Run
`bash tools/train/train_tennis_dallas_until_convergence.sh` to retrain both with
validation-based early stopping and reuse the pilot caches.
For paired video/graph action spotting from the original broadcast manifests,
run `bash tools/train/train_tennis_dallas_spotting_until_convergence.sh`.
Set `CONFIG_ONLY=1` to validate the data and inspect generated configurations.

```
tools/
├── inference/                                # single-video model diagnostics
│   ├── README.md
│   └── inspect_video.py
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
├── upload/                                   # Hugging Face dataset/model upload helpers
│   ├── README.md
│   └── upload_model_hf.py
└── slurm/                                    # SLURM job scripts
    ├── README.md
    ├── install_environment.sh
    ├── generic/
    │   ├── ibex_salloc_cpu.sh
    │   ├── ibex_salloc_gpu.sh
    │   ├── ibex_srun.sh
    │   └── ibex_job.sbatch
    ├── datasets/
    │   ├── README.md
    │   ├── download_osl_xfoul.sbatch
    │   ├── download_gar_tracking.sbatch
    │   └── download_gar_frames.sbatch
    └── training/
        ├── classification_MVFouls.sbatch
        ├── classification_XFoul.sbatch
        └── localization_SNBAS-2023.sbatch
```

---

## Folder guides

- See [tools/inference/README.md](inference/README.md) for actual pose exports and annotated classification/localization videos.
- See [docs/data/osl-json-format.md](../docs/data/osl-json-format.md) for the OSL JSON schema used by dataset tools.
- See [tools/convert/README.md](convert/README.md) for conversion scripts and examples.
- See [tools/download/README.md](download/README.md) for HuggingFace download/upload scripts.
- See [docs/tools/sngar-spotting.md](../docs/tools/sngar-spotting.md) for the SN-GAR action-spotting dataset build and release.
- See [tools/upload/README.md](upload/README.md) for Hugging Face dataset and model upload scripts.
- See [tools/slurm/README.md](slurm/README.md) for Ibex SLURM workflows (`salloc`, `srun`, `sbatch`).
