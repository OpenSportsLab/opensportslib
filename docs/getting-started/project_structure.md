# Project Structure

This page describes the core layout of the SoccerNetPro repository and where key components live.

---

## Repository Layout

```text
soccernetpro/
├── MANIFEST.in
├── README.md
├── pyproject.toml
└── soccernetpro/
    ├── __init__.py

    ├── apis/                   # High-level user APIs
    │   ├── classification.py
    │   └── localization.py

    ├── config/                 # Default configuration files
    │   ├── classification.yaml
    │   ├── classification_tracking.yaml
    │   ├── localization.yaml
    │   └── graph_tracking_classification/

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