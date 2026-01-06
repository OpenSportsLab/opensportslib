# soccernetpro/datasets/builder.py
from soccernetpro.datasets.classification_dataset import ClassificationDataset
from soccernetpro.datasets.localization_dataset import LocalizationDataset
# from .spotting_dataset import SpottingDataset

def build_dataset(config, annotation_file=None, processor=None, split="train"):
    """Return a dataset instance based on model type"""
    task = config.TASK.lower()
    if "classification" in task:
        return ClassificationDataset(config, annotation_file, processor, split=split)
    elif "localization" in task:
        return LocalizationDataset(config, annotation_file, processor, split=split)
    else:
        raise ValueError(f"No dataset")
