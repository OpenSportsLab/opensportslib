# opensportslib/datasets/builder.py
# from .spotting_dataset import SpottingDataset

def build_dataset(config, annotation_file=None, processor=None, split="train"):
    """Return a dataset instance based on model type"""
    task = config.TASK.lower()

    if "classification" in task:
        from opensportslib.datasets import classification_dataset
        return classification_dataset.build(config, annotation_file, processor, split)
    
    elif "localization" in task:
        from opensportslib.datasets.localization_dataset import LocalizationDataset
        return LocalizationDataset(config, annotation_file, processor, split=split)

    elif "pretraining" in task:
        from opensportslib.datasets import pretraining_dataset
        return pretraining_dataset.build(config, data_dir=annotation_file, split=split)

    else:
        raise ValueError(f"No dataset found for task: {task}")


##### --------- ####
# def build_dataset(config, annotation_file=None, processor=None, split="train"):
#     """Return a dataset instance based on task and modality"""
#     task = config.TASK.lower()

#     if "classification" in task:
#         modality = config.DATA.data_modality.lower()
        
#         if modality == "tracking_parquet":
#             from opensportslib.datasets.classification_dataset import TrackingDataset
#             return TrackingDataset(config, annotation_file, split)
#         elif modality == "video":
#             from opensportslib.datasets.classification_dataset import VideoDataset
#             return VideoDataset(config, annotation_file, processor, split)
#         else:
#             raise ValueError(f"Unknown data_modality: {modality}")
    
#     elif "localization" in task:
#         from opensportslib.datasets.localization_dataset import LocalizationDataset
#         return LocalizationDataset(config, annotation_file, processor, split=split)
    
#     else:
#         raise ValueError(f"No dataset found for task: {task}")
