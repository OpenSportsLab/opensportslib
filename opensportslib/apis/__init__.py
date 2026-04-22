# opensportslib/apis/__init__.py

# Import task APIs
from opensportslib.apis.base_task_model import BaseTaskModel
from opensportslib.apis.classification import ClassificationModel
from opensportslib.apis.localization import LocalizationModel
import warnings
warnings.filterwarnings("ignore")

# Expose only these
__all__ = [
    "BaseTaskModel",
    "ClassificationModel",
    "LocalizationModel",
]
