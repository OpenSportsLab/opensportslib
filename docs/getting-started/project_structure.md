# Project Structure

This page describes the core layout of the OpenSportsLib repository and where key components live.

---

## Repository Layout

```text
opensportslib/
├── MANIFEST.in
├── README.md
├── pyproject.toml
└── opensportslib/
    ├── __init__.py

    ├── apis/                   # High-level user APIs
    │   ├── classification.py
    │   └── localization.py

    ├── config/                 # Default configuration files
    │   ├── default.yaml
    │   ├── classification/
    │   │   ├── default.yaml
    │   │   ├── video.yaml
    │   │   ├── sngar_tracking.yaml
    │   │   └── sngar_frames.yaml
    │   └── localization/
    │       ├── default.yaml
    │       ├── video_ocv.yaml
    │       ├── video_dali.yaml
    │       ├── calf_resnetpca512.yaml
    │       └── netvladpp_resnetpca512.yaml

    ├── core/                   # Training engine & utilities
    │   ├── loss/
    │   ├── optimizer/
    │   ├── scheduler/
    │   ├── sampler/
    │   ├── trainer/
    │   └── utils/

    ├── datasets/               # Dataset loaders and builders
    │   ├── builder.py
    │   ├── classification_dataset.py
    │   ├── localization_dataset.py
    │   └── utils/

    ├── metrics/                # Evaluation metrics
    │   ├── classification_metric.py
    │   └── localization_metric.py

    └── models/                 # Model architectures
        ├── backbones/
        ├── heads/
        ├── neck/
        ├── base/
        ├── utils/
        └── builder.py
```
