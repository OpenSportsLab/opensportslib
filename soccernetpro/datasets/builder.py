# soccernetpro/datasets/builder.py
# from .spotting_dataset import SpottingDataset

def build_dataset(config, annotation_file=None, processor=None, split="train"):
    """Return a dataset instance based on model type"""
    task = config.TASK.lower()

    if "classification" in task:
        from soccernetpro.datasets import classification_dataset
        return classification_dataset.build(config, annotation_file, processor, split)
    
    elif "localization" in task:
        from soccernetpro.datasets.localization_dataset import LocalizationDataset
        return LocalizationDataset(config, annotation_file, processor, split=split)
    
    else:
        raise ValueError(f"No dataset found for task: {task}")
