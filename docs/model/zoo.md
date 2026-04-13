## 🏟️ Model Zoo

Explore pretrained models built using **OpenSportsLib** for sports video understanding.

---

### 🎯 Classification Models

| Model | Dataset | Accuracy | Balanced Acc | Top-2 | Link |
|------|--------|---------|--------------|------|------|
| MViT V2 Classification | SoccerNet MVFouls | 0.57 | 0.40 | 0.78 | https://huggingface.co/OpenSportsLab/oslib-MViTv2-classification |

---

### 📍 Localization Models (Action Spotting)

| Model | Dataset | Classes | tight mAP | loose mAP | Link |
|------|--------|--------|----------|----------|------|
| E2E Localization (SNBAS 2025) | SoccerNet 2025 | 12 | 47.98 | 58.35 | https://huggingface.co/OpenSportsLab/oslib-e2e-localization-snbas-2025 |
| E2E Localization (SNBAS 2023) | SoccerNet 2023 | 2 | 71.48 | 85.62 | https://huggingface.co/OpenSportsLab/oslib-e2e-localization-snbas-2023 |

---


## 🚀 Quick Usage

All models can be used via **OpenSportsLib**:

```python
from opensportslib import model

# Classification
cls_model = model.classification(config="config.yaml")
cls_model.infer(
    test_set="test.json",
    pretrained="OpenSportsLab/oslib-MViTv2-classification"
)

# Localization
loc_model = model.localization(config="config.yaml")
loc_model.infer(
    test_set="test.json",
    pretrained="OpenSportsLab/oslib-e2e-localization-snbas-2025"
)
```