# soccernetpro/core/trainer.py

import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from transformers import Trainer as HFTrainer, TrainingArguments
from soccernetpro.core.utils.checkpoint import *



class Trainer_Classification:
    """
    Unified Trainer that can either use native PyTorch loop or HuggingFace Trainer.
    """
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU")
        self.model = None
        self.hf_trainer = None

    def compute_metrics(self, pred):
        from soccernetpro.metrics.classification_metric import compute_classification_metrics
        return compute_classification_metrics(pred, top_k=5)

    def train(self, model, train_dataset, val_dataset=None):
        """
        Use HuggingFace Trainer for VideoMAE training
        """
        from soccernetpro.core.sampler.weighted_sampler import WeightedTrainer, VideoMAETrainer
        from soccernetpro.core.utils.data import balanced_subset

        # run_name = (
        #     f"_freeze={self.config.MODEL.freeze_backbone}"
        #     f"_lr={self.config.TRAIN.learning_rate}"
        #     f"_bs={self.config.DATA.train_batch_size}"
        # )

        # wandb.init(
        #     project="videomae-finetune",
        #     name=run_name,
        # )

        if self.config.MODEL.type == "huggingface":
            self.model = model
    
            args = TrainingArguments(
                label_names=["labels"],
                output_dir=self.config.TRAIN.save_dir,
                per_device_train_batch_size=self.config.DATA.train_batch_size,
                per_device_eval_batch_size=self.config.DATA.valid_batch_size,
                num_train_epochs=self.config.TRAIN.epochs,
                #learning_rate=self.config.TRAIN.learning_rate,
                eval_strategy="epoch" if val_dataset else "no",
                save_strategy="epoch",
                logging_strategy="steps",
                logging_steps=5,
                save_total_limit=10,
                load_best_model_at_end=True,
                fp16=True,
                warmup_ratio=0.1,
                #max_steps=(len(train_dataset) // self.config.DATA.train_batch_size) * self.config.TRAIN.epochs
            )

            if self.config.TRAIN.use_weighted_sampler:
                print("Using Weighted Trainer with WeightedRandomSampler")
                self.hf_trainer = WeightedTrainer(
                    model=self.model,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=self.compute_metrics,
                    config=self.config
                )
            else:
                #########
                #train_subset = balanced_subset(train_dataset, 10)
                #########
                self.hf_trainer = VideoMAETrainer(
                    model=self.model,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=self.compute_metrics,
                    config=self.config
                )
    
            self.hf_trainer.train()
            #############
            train_metrics = self.hf_trainer.evaluate(train_dataset, metric_key_prefix="train")
            print("TRAIN METRICS:", train_metrics)
            #############
        # wandb.finish()

    def infer(self, test_dataset):
        if self.config.MODEL.type == "huggingface":

            args = TrainingArguments(
            output_dir=self.config.TRAIN.save_dir,  # any directory, not used here
            per_device_eval_batch_size=1#self.config.DATA.valid_batch_size,  # or whatever batch size you want
            )

            self.hf_trainer = HFTrainer(
                model=self.model,
                args=args,
                compute_metrics=self.compute_metrics  # optional, can compute later manually
            )

            preds_output = self.hf_trainer.predict(test_dataset)
            logits = preds_output.predictions
            # if isinstance(logits, tuple):
            #     logits = logits[0]

            # predictions = np.argmax(logits, axis=-1)
            labels = preds_output.label_ids
            metrics = self.compute_metrics((logits, labels))

        return logits, metrics


    def demo(self, model, video_paths):
        pass

    def save(self, model, path, processor=None, tokenizer=None, optimizer=None, epoch=None):
        """
        Save model checkpoint
        """
        save_checkpoint(model, path, processor, tokenizer, optimizer, epoch)
        print(f"Model saved at {path}")

    def load(self, path, optimizer=None):
        """
        Load model checkpoint. Returns loaded model, optimizer, epoch
        """
        if self.config.MODEL.type == "huggingface":
            epoch = None
            self.model, processor = load_huggingface_checkpoint(self.config, path=path, device=self.device)
            print(f"Model loaded from {path}")
            return self.model, processor, epoch
        else:
            self.model, optimizer, epoch = load_checkpoint(
                self.model, path, optimizer, device=self.device
            )
            self.optimizer = optimizer
            print(f"Model loaded from {path}, epoch: {epoch}")
            return self.model, self.optimizer, epoch
