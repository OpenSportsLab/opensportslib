from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Trainer as HFTrainer
import torch

class WeightedTrainer(HFTrainer):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
    # --- Weighted training loader ---
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        weights = self.train_dataset.get_sample_weights()
        #print(f"Sample weights: {max(weights)} {min(weights)}")
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,  # per-device batch size
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers if hasattr(self.args, "dataloader_num_workers") else 0,
            pin_memory=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,  # validation batch size
            shuffle=False,  # do not shuffle evaluation
            num_workers=self.args.dataloader_num_workers if hasattr(self.args, "dataloader_num_workers") else 0,
            pin_memory=True,
        )

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model

        optimizer_grouped_parameters = []
        print("Weighted trainer ", self.config.TRAIN.head_lr, self.config.TRAIN.backbone_lr)
        # ---- Classifier head ----
        if self.config.MODEL.unfreeze_head:
            optimizer_grouped_parameters.append({
                "params": model.classifier.parameters(),
                "lr": self.config.TRAIN.head_lr,
            })

        # ---- Backbone (last N layers) ----
        n = self.config.MODEL.unfreeze_last_n_layers
        if n > 0:
            optimizer_grouped_parameters.append({
                "params": model.videomae.encoder.layer[-n:].parameters(),
                "lr": self.config.TRAIN.backbone_lr,
            })

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            weight_decay=self.config.TRAIN.weight_decay,
        )

        return self.optimizer

from transformers.optimization import get_scheduler
class VideoMAETrainer(HFTrainer):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config   

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model

        optimizer_grouped_parameters = []
        print(type(self.config.TRAIN.head_lr), type(self.config.TRAIN.backbone_lr))
        # ---- Classifier head ----
        if self.config.MODEL.unfreeze_head:
            optimizer_grouped_parameters.append({
                "params": model.classifier.parameters(),
                "lr": self.config.TRAIN.head_lr,
            })

        # ---- Backbone (last N layers) ----
        n = self.config.MODEL.unfreeze_last_n_layers
        if n > 0:
            optimizer_grouped_parameters.append({
                "params": model.videomae.encoder.layer[-n:].parameters(),
                "lr": self.config.TRAIN.backbone_lr,
            })

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            weight_decay=self.config.TRAIN.weight_decay,
        )

        return self.optimizer

