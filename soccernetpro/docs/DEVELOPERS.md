## 📁 Folder Structure

The following is the detailed folder structure for the `SoccerNetPro` repository, describing the organization of core modules, configurations, and documentation.

---

### Root Directory

- **`README.md`**  
  Project overview, feature list, installation instructions, and quick-start examples.

- **`pyproject.toml`**  
  Project metadata, build system requirements, and dependency lists.

- **`MANIFEST.in`**  
  Ensures non-Python files (such as YAML configurations) are included in package distributions.

---

## `soccernetpro/` (Source Code)

The main package containing the framework’s core logic.

---

### `apis/`
High-level entry points for training and inference.

- **`classification.py`**  
  API for video-based action recognition and foul classification tasks.

- **`localization.py`**  
  API for temporal action spotting tasks.

---

### `core/`
The internal engine of the framework.

- **`trainer/`**  
  Implementation of task-specific training and inference loops.

- **`loss/`**  
  Custom loss functions (e.g., Cross-Entropy and specialized localization losses).

- **`optimizer/` & `scheduler/`**  
  Centralized builders for optimization strategies and learning rate schedules.

- **`sampler/`**  
  Specialized data samplers, such as weighted samplers for handling imbalanced datasets.

- **`utils/`**  
  Core utilities for:
  - DDP setup  
  - Configuration resolution  
  - Checkpointing  
  - Video processing  

---

### `models/`
Modular architecture components for flexible model building.

- **`backbones/`, `necks/`, `heads/`**  
  Layers and builders used to construct neural network architectures.

- **`base/`**  
  Base classes and modality-specific implementations  
  (video-based, tracking-based, or end-to-end models).

- **`utils/`**  
  Shared layers and specialized modules such as:
  - TSM  
  - GSM  
  - ASFormer  

- **`builder.py`**  
  Centralized logic used to assemble full models from configuration files.

---

### `datasets/`
Data ingestion and preprocessing logic for various soccer-related tasks.

- **`classification_dataset.py`**  
  Loaders for action recognition and foul classification.

- **`localization_dataset.py`**  
  High-performance loaders integrated with NVIDIA DALI for temporal action spotting.

- **`builder.py`**  
  Orchestrates dataset instantiation based on YAML settings.

---

### `config/`
Templates and configuration files defining task parameters.

- **`classification.yaml`**  
  Configuration template for foul classification tasks.

- **`localization.yaml`**  
  Configuration template for action spotting tasks.

- **`graph_tracking_classification/`**  
  Specialized configurations for graph-based tracking models  
  (e.g., GraphConv, GIN).

---

### `metrics/`
Evaluation logic and performance measurement tools.

- **`classification_metric.py`**  
  Calculates accuracy, precision, recall, and F1 scores.

- **`localization_metric.py`**  
  Computes Average Precision (mAP) for temporal spotting tasks.


### 👨‍💻 Where to Start as a New Developer

Start from the public APIs:
```bash
soccernetpro/apis/classification.py
soccernetpro/apis/localization.py
```
These handle: config loading ,trainer creation, training execution, inference entry points  
Most workflows begin here.

### 📦 Add a New Dataset

Work in:
```bash
soccernetpro/datasets/
soccernetpro/datasets/builder.py
```
Steps:

1. Create a dataset class.

2. Register it in `datasets/builder.py`


### 🧠 Add a New Model

Work in:
```bash
soccernetpro/models/
soccernetpro/models/builder.py
```
Steps:

1. Implement your model inside one of:
   - `backbones/`
   - `necks/`
   - `heads/`
   - `base/`

2. Register the model inside `models/builder.py`


### ⚙️ Modify Configurations
All experiment configs live in:
```bash
soccernetpro/config/*.yaml
```
Typical edits:
- TYPE
- DATA
- MODEL
- TRAIN
- SYSTEM

### 🔁 Modify the Training Loop

To change training behavior, edit:
```bash
soccernetpro/core/trainer/
```
This is where you can modify:

- Forward pass  
- Loss computation  
- Logging  
- Validation   
- Checkpointing  

### High-Level Workflow

YAML Config
   ↓
APIs (apis/)
   ↓
Datasets (datasets/)
   ↓
Models (models/)
   ↓
Trainer (core/trainer/)
   ↓
Metrics (metrics/)