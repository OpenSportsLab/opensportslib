# soccernetpro/core/trainer/classification_trainer.py

import os
import gc
import json
import time
import logging

import torch
import tqdm
import wandb
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Trainer as HFTrainer, TrainingArguments

from soccernetpro.metrics.classification_metric import compute_classification_metrics, process_preds_labels
from soccernetpro.core.utils.wandb import log_confusion_matrix_wandb
from soccernetpro.core.utils.checkpoint import *
from soccernetpro.core.utils.config import select_device
from soccernetpro.core.utils.seed import seed_worker

# ============================================================
# Base Trainer
# ============================================================

class BaseTrainerClassification:
    """
    Base trainer with common functionality for classification tasks.
    Subclasses implement _forward_batch() for modality-specific logic.
    """
    
    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        model,
        optimizer,
        scheduler,
        criterion,
        class_weights,
        class_names,
        save_dir,
        model_name,
        max_epochs=1000,
        device="cuda",
        top_k=2,
        wandb_project="classification",
        wandb_run_name=None,
        wandb_config=None,
        patience=10,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.class_weights = class_weights
        self.class_names = class_names

        self.model_name = model_name
        self.max_epochs = max_epochs
        self.device = device
        self.top_k = top_k
        self.patience = patience

        # best model tracking
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.best_model_state = None
        self.best_epoch = 0

        # W&B init
        self.wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config,
            reinit=True
        )

        run_id = wandb.run.id if wandb.run else time.strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join(save_dir, model_name, run_id)
        os.makedirs(self.save_dir, exist_ok=True)

        try:
            wandb.watch(self.model, log="gradients", log_freq=100)
        except Exception:
            pass

    def _forward_batch(self, batch):
        """
        Modality-specific forward pass. Override in subclass.
        Returns: logits, labels
        """
        raise NotImplementedError

    def train(self, epoch_start=0, save_every=3):
        logging.info("Starting training")

        for epoch in range(epoch_start, self.max_epochs):
            print(f"\nEpoch {epoch+1}/{self.max_epochs}")

            # Train
            pbar = tqdm.tqdm(total=len(self.train_loader), desc="Training", position=0, leave=True)
            _, _, train_loss, train_metrics = self._run_epoch(
                self.train_loader, 
                epoch + 1, 
                train=True, 
                set_name="train", 
                pbar=pbar
            )
            pbar.close()

            # Validation
            pbar = tqdm.tqdm(total=len(self.val_loader), desc="Valid", position=1, leave=True)
            _, _, val_loss, val_metrics = self._run_epoch(
                self.val_loader, 
                epoch + 1, 
                train=False, 
                set_name="valid", 
                pbar=pbar
            )
            pbar.close()

            # Get current LR before scheduler step
            prev_lr = self.optimizer.param_groups[0]["lr"] # NOTE: this is for ReduceLROnPlateau scheduler we discussed with Silvio

            # Scheduler step (ReduceLROnPlateau needs metric)
            val_metric = val_metrics.get("balanced_accuracy", val_metrics.get("accuracy", 0))
            train_metric = train_metrics.get("balanced_accuracy", train_metrics.get("accuracy", 0))

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Model reversion on LR change
            # NOTE: this is for ReduceLROnPlateau scheduler we discussed with Silvio
            if current_lr != prev_lr and self.best_model_state is not None:
                print(f"LR changed from {prev_lr:.2e} to {current_lr:.2e}, reverting to best model")
                self.model.load_state_dict(self.best_model_state)

            # W&B logging
            wandb.log({
                "epoch": epoch + 1,
                "lr": current_lr,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"valid/{k}": v for k, v in val_metrics.items()},
            })

            print(f"Train Loss: {train_loss:.4f} | Train Bal Acc: {train_metric:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Bal Acc: {val_metric:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:  
                self.best_val_loss = val_loss
                self.best_val_metric = val_metric
                self.best_epoch = epoch + 1
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint(epoch + 1, is_best=True)
                print(f"New best model at epoch {epoch + 1}")

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch + 1, is_best=False)

            # Early stopping on min LR
            min_lr = getattr(self.scheduler, 'min_lrs', [1e-8])[0] if hasattr(self.scheduler, 'min_lrs') else 1e-8
            if current_lr <= 2 * min_lr:
                print("Early stopping: learning rate too low")
                break

        # Save validation metrics report using best model
        if self.best_model_state is not None:
            print(f"\nGenerating validation report from best model (epoch {self.best_epoch})")
            self.model.load_state_dict(self.best_model_state)
            pbar = tqdm.tqdm(total=len(self.val_loader), desc="Best model validation", position=0, leave=True)
            all_logits, all_labels, val_loss, val_metrics = self._run_epoch(
                self.val_loader,
                self.best_epoch,
                train=False,
                set_name="valid",
                pbar=pbar
            )
            pbar.close()
            self._save_metrics_report(all_logits, all_labels, val_loss, val_metrics, "validation", self.best_epoch)

        print(f"Training finished. Best epoch: {self.best_epoch}")

    def test(self, epoch=None):
        """
        Run test set evaluation.
        If epoch is provided, logs under that epoch number.
        """
        print("\nRunning TEST evaluation")
        pbar = tqdm.tqdm(total=len(self.test_loader), desc="Test", position=0, leave=True)
        all_logits, all_labels, test_loss, test_metrics = self._run_epoch(
            self.test_loader,
            epoch if epoch is not None else "final",
            train=False,
            set_name="test",
            pbar=pbar
        )
        pbar.close()

        wandb.log({
            "test/loss": test_loss,
            **{f"test/{k}": v for k, v in test_metrics.items()},
        })

        # Save test metrics report
        self._save_metrics_report(all_logits, all_labels, test_loss, test_metrics, "test", epoch)

        print("TEST METRICS:", test_metrics)
        return test_loss, test_metrics

    def _run_epoch(self, dataloader, epoch, train=False, set_name="train", pbar=None):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_batches = 0

        all_logits = []
        all_labels = []

        for batch in dataloader:
            if pbar:
                pbar.update()

            with torch.set_grad_enabled(train):
                logits, labels = self._forward_batch(batch)

                if self.class_weights is not None:
                    loss = self.criterion(output=logits, labels=labels, weight=self.class_weights.to(self.device))
                else:
                    loss = self.criterion(output=logits, labels=labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        # Compute metrics
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        metrics = compute_classification_metrics((all_logits, all_labels), top_k=self.top_k)

        # Confusion matrix for valid/test
        if set_name in ["valid", "test"]:
            preds_all, labels_all, _ = process_preds_labels((all_logits, all_labels), top_k=None)
            class_names = [self.class_names[i] for i in sorted(self.class_names.keys())]

            log_confusion_matrix_wandb(
                y_true=labels_all.tolist(),
                y_pred=preds_all.tolist(),
                class_names=class_names,
                split_name=set_name
            )

        gc.collect()
        torch.cuda.empty_cache()

        return all_logits, all_labels, total_loss / max(1, total_batches), metrics

    def _save_metrics_report(self, all_logits, all_labels, loss, metrics, set_name, epoch=None):
        """
        Save detailed metrics report as txt file and confusion matrix as png.
        """
        from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
        
        preds = np.argmax(all_logits, axis=-1)
        
        # sort classes alphabetically for consistent reporting
        class_names_sorted = sorted(self.class_names.keys(), key=lambda k: self.class_names[k])
        sorted_class_names = sorted(class_names_sorted)
        class_to_sorted_idx = {self.class_names[name]: i for i, name in enumerate(sorted_class_names)}
        
        sorted_labels = np.array([class_to_sorted_idx[l] for l in all_labels])
        sorted_preds = np.array([class_to_sorted_idx[p] for p in preds])
        
        cm = confusion_matrix(sorted_labels, sorted_preds)
        per_class_accuracy = np.diag(cm) / np.maximum(cm.sum(axis=1), 1) * 100
        balanced_acc = balanced_accuracy_score(sorted_labels, sorted_preds) * 100
        
        # save confusion matrix plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=sorted_class_names, yticklabels=sorted_class_names)
            plt.title(f'Confusion Matrix ({set_name})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plots_dir = os.path.join(self.save_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{set_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: could not save confusion matrix plot: {e}")
        
        # save text report
        results_dir = os.path.join(self.save_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        report_path = os.path.join(results_dir, f'{set_name}_metrics.txt')
        with open(report_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"{set_name.upper()} METRICS REPORT\n")
            if epoch is not None:
                f.write(f"Epoch: {epoch}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Loss: {loss:.4f}\n")
            f.write(f"Balanced Accuracy: {balanced_acc:.2f}%\n\n")
            
            # overall metrics
            f.write(f"{'Metric':<30} {'Value':>10}\n")
            f.write(f"{'-'*40}\n")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    f.write(f"{k:<30} {v:>10.4f}\n")
                else:
                    f.write(f"{k:<30} {str(v):>10}\n")
            f.write(f"{'-'*40}\n\n")
            
            # per-class accuracy
            f.write(f"{'Class':<30} {'Accuracy':>10} {'Samples':>10}\n")
            f.write(f"{'-'*60}\n")
            for i, class_name in enumerate(sorted_class_names):
                num_samples = int(cm[i].sum())
                f.write(f"{class_name:<30} {per_class_accuracy[i]:>9.2f}% {num_samples:>10}\n")
            f.write(f"{'-'*60}\n\n")
            
            # sklearn classification report
            f.write("Classification Report:\n\n")
            f.write(classification_report(
                sorted_labels, sorted_preds,
                target_names=sorted_class_names,
                zero_division=0
            ))
            f.write(f"\n{'-'*60}\n\n")
            
            # confusion matrix as text
            f.write("Confusion Matrix:\n\n")
            f.write(f"{cm}\n")
        
        print(f"Saved {set_name} metrics report to {report_path}")

    def _save_checkpoint(self, epoch, is_best=False):
        os.makedirs(os.path.join(self.save_dir, "models"), exist_ok=True)
        
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_metric": self.best_val_metric,
            "best_val_loss": self.best_val_loss,
        }

        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = os.path.join(self.save_dir, "models", filename)
        torch.save(state, path)
        print(f"Saved: {path}")
        return path


# ============================================================
# MV Trainer (Video)
# ============================================================

class MVTrainerClassification(BaseTrainerClassification):
    """Trainer for multi-view video classification."""

    def _forward_batch(self, batch):
        mvclips = batch["pixel_values"].to(self.device).float()
        labels = batch["labels"].to(self.device)

        outputs = self.model(mvclips)

        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        return logits, labels


# ============================================================
# Tracking Trainer
# ============================================================

class TrackingTrainerClassification(BaseTrainerClassification):
    """Trainer for tracking-based classification."""

    def _forward_batch(self, batch):
        # Move all tensors to device at once
        tracking_batch = {
            "x": batch["x"].to(self.device),
            "edge_index": batch["edge_index"].to(self.device),
            "batch": batch["batch"].to(self.device),
            "batch_size": batch["batch_size"],
            "seq_len": batch["seq_len"],
        }
        labels = batch["labels"].to(self.device)
        
        logits = self.model(tracking_batch)
        
        return logits, labels


# ============================================================
# Unified Trainer Entry Point
# ============================================================

class Trainer_Classification:
    """
    Unified trainer that dispatches to appropriate trainer based on config.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = select_device(self.config.SYSTEM)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.trainer = None

    def compute_metrics(self, pred):
        return compute_classification_metrics(pred, top_k=2)

    def train(self, model, train_dataset, val_dataset=None):
        from soccernetpro.core.loss.builder import build_criterion
        from soccernetpro.core.optimizer.builder import build_optimizer
        from soccernetpro.core.scheduler.builder import build_scheduler
        from soccernetpro.core.utils.data import tracking_collate_fn

        modality = getattr(self.config.DATA, 'data_modality', 'video')
        seed = getattr(self.config.SYSTEM, 'seed', 42)
        g = torch.Generator()
        g.manual_seed(seed)

        # HuggingFace models (VideoMAE)
        if self.config.MODEL.type == "huggingface":
            self._train_huggingface(model, train_dataset, val_dataset)
            return

        # Custom models (MV or Tracking)
        self.model = model.to(self.device)

        # Build components
        optimizer = build_optimizer(self.model.parameters(), cfg=self.config.TRAIN.optimizer)
        scheduler = build_scheduler(optimizer, cfg=self.config.TRAIN.scheduler)
        criterion = build_criterion(self.config.TRAIN.criterion)

        # Class weights
        if self.config.TRAIN.use_weighted_loss:
            class_weights = train_dataset.get_class_weights(
                num_classes=train_dataset.num_classes(),
                sqrt=True
            ).to(self.device)
        else:
            class_weights = None

        # Collate function
        collate_fn = tracking_collate_fn if modality == "tracking_parquet" else None

        # Sampler
        if self.config.TRAIN.use_weighted_sampler:
            sample_weights = train_dataset.get_sample_weights()

            samples_per_class = getattr(self.config.TRAIN, 'samples_per_class', None)
            if samples_per_class:
                num_classes = train_dataset.num_classes()
                num_samples = samples_per_class * num_classes
            else:
                num_samples = len(sample_weights)

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=num_samples,
                replacement=True,
                generator=g
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.DATA.train_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.DATA.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.DATA.valid_batch_size,
            shuffle=False,
            num_workers=self.config.DATA.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g
        )

        # Select trainer class
        if modality == "tracking_parquet":
            TrainerClass = TrackingTrainerClassification
        else:
            TrainerClass = MVTrainerClassification

        # Create trainer
        self.trainer = TrainerClass(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            class_weights=class_weights,
            class_names=train_dataset.label_map,
            save_dir=self.config.TRAIN.save_dir,
            model_name=self.config.MODEL.backbone.type,
            max_epochs=self.config.TRAIN.epochs,
            device=self.device,
            top_k=2,
            wandb_project=self.config.TASK,
            wandb_run_name=f"{self.config.MODEL.backbone.type}_{modality}",
            wandb_config={
                "modality": modality,
                "lr": self.config.TRAIN.optimizer.lr,
                "batch_size": self.config.DATA.train_batch_size,
            },
            patience=self.config.TRAIN.patience,
        )

        self.trainer.train(epoch_start=self.epoch, save_every=self.config.TRAIN.save_every)

    def _train_huggingface(self, model, train_dataset, val_dataset):
        """Handle HuggingFace Trainer for VideoMAE."""
        from soccernetpro.core.sampler.weighted_sampler import WeightedTrainer, VideoMAETrainer

        self.model = model

        args = TrainingArguments(
            label_names=["labels"],
            output_dir=self.config.TRAIN.save_dir,
            per_device_train_batch_size=self.config.DATA.train_batch_size,
            per_device_eval_batch_size=self.config.DATA.valid_batch_size,
            num_train_epochs=self.config.TRAIN.epochs,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=5,
            save_total_limit=10,
            load_best_model_at_end=True,
            fp16=True,
            warmup_ratio=0.1,
        )

        if self.config.TRAIN.use_weighted_sampler:
            self.trainer = WeightedTrainer(
                model=self.model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
                config=self.config
            )
        else:
            self.trainer = VideoMAETrainer(
                model=self.model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics,
                config=self.config
            )

        self.trainer.train()

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
        
        else:
            print("Using Custom Trainer class for evaluation of non-HuggingFace model")
            from soccernetpro.core.loss.builder import build_criterion
            from soccernetpro.core.optimizer.builder import build_optimizer
            from soccernetpro.core.scheduler.builder import build_scheduler
            from soccernetpro.core.utils.data import tracking_collate_fn

            modality = getattr(self.config.DATA, 'data_modality', 'video')
            collate_fn = tracking_collate_fn if modality == "tracking_parquet" else None

            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.config.DATA.valid_batch_size, 
                shuffle=False, 
                num_workers=self.config.DATA.num_workers, 
                pin_memory=True,
                collate_fn=collate_fn
            )

            optimizer = self.optimizer if self.optimizer is not None else build_optimizer(self.model.parameters(), cfg=self.config.TRAIN.optimizer)
            scheduler = self.scheduler if self.scheduler is not None else build_scheduler(optimizer, cfg=self.config.TRAIN.scheduler)
            criterion = build_criterion(self.config.TRAIN.criterion)

            # Select trainer class based on modality
            if modality == "tracking_parquet":
                TrainerClass = TrackingTrainerClassification
            else:
                TrainerClass = MVTrainerClassification

            self.test_trainer = TrainerClass(
                train_loader=None,
                val_loader=None,
                test_loader=test_loader,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                class_weights=None,
                class_names=test_dataset.label_map,
                save_dir=self.config.TRAIN.save_dir,
                model_name=self.config.MODEL.backbone.type,
                max_epochs=self.config.TRAIN.epochs,
                device=self.device,
                top_k=2,
                wandb_project=self.config.TASK,
                wandb_run_name=f"{self.config.MODEL.backbone.type}_{modality}_test",
                wandb_config={
                    "modality": modality,
                    "lr": self.config.TRAIN.optimizer.lr,
                    "batch_size": self.config.DATA.train_batch_size,
                },
            )
            loss, metrics = self.test_trainer.test()
            
        return metrics

    def demo(self, model, video_paths):
        pass

    def save(self, model, path, processor=None, tokenizer=None, optimizer=None, epoch=None):
        """
        Save model checkpoint
        """
        save_checkpoint(model, path, processor, tokenizer, optimizer, epoch)
        print(f"Model saved at {path}")

    def load(self, path, optimizer=None, scheduler=None):
        """
        Load model checkpoint. Returns loaded model, optimizer, epoch
        """
        if self.config.MODEL.type == "huggingface":
            epoch = None
            self.model, processor = load_huggingface_checkpoint(self.config, path=path, device=self.device)
            print(f"Model loaded from {path}")
            return self.model, processor, scheduler, epoch
        else:
            from soccernetpro.models.builder import build_model
            if self.model is None:
                self.model, _ = build_model(self.config, self.device)
            self.model, optimizer, scheduler, epoch = load_checkpoint(
                self.model, path, optimizer, scheduler, device=self.device
            )
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.epoch = epoch
            print(f"Model loaded from {path}, epoch: {epoch}")
            return self.model, self.optimizer, self.scheduler, self.epoch

