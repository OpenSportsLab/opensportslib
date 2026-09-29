from __future__ import annotations

from services.classification_service import ClassificationService
from services.localization_service import LocalizationService
from services.vqa_service import VQAService


SERVICE_BY_TASK = {
    "vqa": VQAService,
    "classification": ClassificationService,
    "localization": LocalizationService,
}
