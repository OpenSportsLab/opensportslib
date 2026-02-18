# Folder Structure

The following is the detailed folder structure for the `SoccerNetPro` repository, describing the organization of core modules, configurations, and documentation.

**Root Directory**
`README.md`: Project overview, feature list, installation instructions, and quick-start examples.

`pyproject.toml`: Project metadata, build system requirements, and comprehensive dependency lists.

`MANIFEST.in`: Ensures non-Python files, such as YAML configurations, are included in package distributions.

`soccernetpro/` (Source Code)
The main package containing the framework's logic.

``apis/``
High-level entry points for training and inference.

```classification.py```: API for video-based action recognition and foul classification tasks.

```localization.py```: API for temporal action spotting tasks.

``core/``
The internal engine of the framework.

```trainer/```: Implementation of task-specific training and inference loops.

***loss/***: Custom loss functions (e.g., Cross-Entropy and specialized localization losses).

***optimizer/ & scheduler/***: Centralized builders for optimization strategies and learning rate schedules.

***sampler/:*** Specialized data samplers, such as the weighted sampler for handling imbalanced datasets.

***utils/:*** Core utilities for DDP setup, configuration resolution, checkpointing, and video processing.

**models/**
Modular architecture components for flexible model building.

***backbones/***, ***necks/***, ***heads/***: Specific layers and builders for constructing neural network architectures.

***base/***: Base classes and modality-specific implementations (e.g., video-based, tracking-based, or end-to-end models).

utils/: Shared layers and specialized modules such as TSM, GSM, and ASFormer implementations.

builder.py: The centralized logic used to assemble full models from configuration files.

datasets/
Data ingestion and preprocessing logic for various soccer-related tasks.

classification_dataset.py: Loaders for standard action recognition and foul classification.

localization_dataset.py: High-performance loaders integrated with NVIDIA DALI for temporal action spotting.

builder.py: Orchestrates dataset instantiation based on YAML settings.

config/
Templates and configuration files defining task parameters.

classification.yaml: Configuration template for foul classification tasks.

localization.yaml: Configuration template for action spotting tasks.

graph_tracking_classification/: Specialized configurations for graph-based tracking models (e.g., GraphConv, GIN).

metrics/
Evaluation logic and performance measurement tools.

classification_metric.py: Logic for calculating accuracy and F1 scores.

localization_metric.py: Specialized tools for calculating Average Precision (mAP) for spotting tasks.