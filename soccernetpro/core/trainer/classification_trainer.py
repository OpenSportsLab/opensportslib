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
from soccernetpro.core.utils.ddp import DistributedWeightedSampler
from soccernetpro.metrics.classification_metric import compute_classification_metrics, process_preds_labels
from soccernetpro.core.utils.wandb import log_confusion_matrix_wandb
from soccernetpro.core.utils.checkpoint import *
from soccernetpro.core.utils.config import select_device
import torch.distributed as dist
from datetime import datetime
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
        monitor="balanced_accuracy",
        mode="max"
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.model = model#.to(device)
        #self.model = DDP(self.model, device_ids=[device])
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

        self.monitor = monitor
        self.mode = mode

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.rank == 0:
            # W&B init
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config,
                reinit=True
            )
            run_id = wandb.run.id if wandb.run else time.strftime("%Y%m%d-%H%M%S")
        else:
            self.wandb_run = None
            run_id = time.strftime("%Y%m%d-%H%M%S")

        #run_id = wandb.run.id if wandb.run else time.strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join(save_dir, model_name, run_id)
        os.makedirs(self.save_dir, exist_ok=True)

        try:
            if self.rank == 0:
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
        monitor = self.monitor
        mode = self.mode
        best_metric = -float("inf") if mode == "max" else float("inf")
        best_path = None
        for epoch in range(epoch_start, self.max_epochs):
            print(f"\nEpoch {epoch+1}/{self.max_epochs}")

            # Train
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            disable = self.rank != 0
            pbar = tqdm.tqdm(total=len(self.train_loader), desc="Training", position=0, leave=True, disable=disable)
            _, _, train_loss, train_metrics = self._run_epoch(
                self.train_loader, 
                epoch + 1, 
                train=True, 
                set_name="train", 
                pbar=pbar
            )
            pbar.close()

            # Validation
            pbar = tqdm.tqdm(total=len(self.val_loader), desc="Valid", position=1, leave=True, disable=disable)
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

            if self.rank == 0:
                # ---------------- W&B LOG ----------------
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

            # ---------------- CHECKPOINT ----------------
            current = val_loss if monitor == "loss" else val_metrics.get(monitor, 0)

            is_better = current > best_metric if mode == "max" else current < best_metric

            if is_better and self.rank == 0:
                best_metric = current
                self.best_metric = best_metric
                best_path = self._save_checkpoint("best", epoch + 1, tag="best")

                artifact = wandb.Artifact("model-checkpoint", type="model")
                artifact.add_file(best_path)
                wandb.log_artifact(artifact)
            
        print("Training finished.")

    # =========================================================
    # TEST = Separate Call
    # =========================================================
    def test(self, epoch=None, detailed_results=False):
        """
        Run test set evaluation.
        If epoch is provided, logs under that epoch number.
        """
        print("\nRunning TEST evaluation")
        pbar = tqdm.tqdm(total=len(self.test_loader), desc="Test", position=0, leave=True, disable = self.rank != 0)
        all_logits, all_labels, test_loss, test_metrics = self._run_epoch(
            self.test_loader,
            epoch if epoch is not None else "final",
            train=False,
            set_name="test",
            pbar=pbar
        )
        pbar.close()

        if self.rank==0:
            wandb.log({
                "test/loss": test_loss,
                **{f"test/{k}": v for k, v in test_metrics.items()},
            })

            print("TEST METRICS:", test_metrics)

            if detailed_results:
                from soccernetpro.metrics.classification_metric import compute_detailed_classification_metrics
                compute_detailed_classification_metrics(
                    all_logits=all_logits,
                    all_labels=all_labels,
                    class_names=self.class_names,
                    save_dir=self.save_dir,
                    set_name="test",
                )

        return test_loss, test_metrics

    def _run_epoch(self, dataloader, epoch, train=False, set_name="train", pbar=None):
        import torch.distributed as dist

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_batches = 0

        all_logits = []
        all_labels = []
        results = []

        # -------- Create epoch folder --------
        epoch_dir = os.path.join(self.save_dir, str(epoch))
        os.makedirs(epoch_dir, exist_ok=True)
        save_path = os.path.join(epoch_dir, f"predictions_{set_name}_epoch_{epoch}.json")

        # =========================================================
        # LOOP
        # =========================================================
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            logits_cpu = logits.detach().cpu()
            labels_cpu = labels.detach().cpu()

            all_logits.append(logits_cpu)
            all_labels.append(labels_cpu)

            # -------- JSON preds --------
            probs = torch.softmax(logits_cpu, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = probs.max(dim=1).values
            ids = batch["id"]

            for i in range(len(preds)):
                results.append({
                    "id": ids[i],
                    "pred_label": self.class_names[preds[i].item()],
                    "confidence": float(confs[i].item()),
                    "pred_class_idx": preds[i].item(),
                })

        # =========================================================
        # CONCAT LOCAL
        # =========================================================
        if len(all_logits) > 0:
            all_logits = torch.cat(all_logits).numpy()
            all_labels = torch.cat(all_labels).numpy()
        else:
            all_logits = np.zeros((0, 1))
            all_labels = np.zeros((0,))

        # =========================================================
        # DDP GATHER (SAFE — handles uneven sizes)
        # =========================================================
        if dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, (all_logits, all_labels, results))

            if self.rank == 0:
                all_logits = np.concatenate([g[0] for g in gathered])
                all_labels = np.concatenate([g[1] for g in gathered])
                results = [r for g in gathered for r in g[2]]
            else:
                # non-rank0 returns early
                return None, None, 0.0, {}

        # =========================================================
        # METRICS (rank 0 only)
        # =========================================================
        metrics = compute_classification_metrics(
            (all_logits, all_labels),
            top_k=self.top_k
        )

        # =========================================================
        # CONFUSION MATRIX (rank 0 only)
        # =========================================================
        if self.rank == 0 and set_name in ["valid", "test"]:
            preds_all, labels_all, _ = process_preds_labels((all_logits, all_labels))
            class_names = [self.class_names[i] for i in sorted(self.class_names.keys())]

            log_confusion_matrix_wandb(
                y_true=labels_all.tolist(),
                y_pred=preds_all.tolist(),
                class_names=class_names,
                split_name=set_name,
            )

        # =========================================================
        # SAVE JSON (rank 0 only)
        # =========================================================
        if self.rank == 0:
            submission = {
                "version": "2.0",
                "task": "action_classification",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "metadata": {"type": "predictions"},
                "data": [],
            }

            for r in results:
                submission["data"].append({
                    "id": r["id"],
                    "labels": {
                        "action": {
                            "label": r["pred_label"],
                            "confidence": r["confidence"],
                        }
                    },
                })

            print("RESULTS Length:", len(results))

            with open(save_path, "w") as f:
                json.dump(submission, f, indent=2)

        # =========================================================
        return all_logits, all_labels, total_loss / max(1, total_batches), metrics



    # =========================================================
    # CHECKPOINT
    # =========================================================
    def _save_checkpoint(self, filename, epoch, tag=None):
        epoch_dir = os.path.join(self.save_dir, str(filename))
        os.makedirs(epoch_dir, exist_ok=True)

        state = {
            "epoch": epoch,
            "state_dict": self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "monitor": self.monitor,
            "mode": self.mode,
            "best_metric": self.best_metric
        }

        name = f"epoch_{epoch}.pt"
        if tag:
            name = f"{tag}_epoch_{epoch}.pt"

        path_aux = os.path.join(epoch_dir, name)
        torch.save(state, path_aux)
        print(f"Saved checkpoint: {path_aux}")
        return path_aux


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

    def compute_metrics(self, pred, mode="logits"):
        return compute_classification_metrics(pred, top_k=2, mode=mode)

    def train(self, model, train_dataset, val_dataset=None, rank=0, world_size=1):
        from soccernetpro.core.loss.builder import build_criterion
        from soccernetpro.core.optimizer.builder import build_optimizer
        from soccernetpro.core.scheduler.builder import build_scheduler
        from soccernetpro.core.utils.data import tracking_collate_fn
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler

        is_ddp = world_size > 1

        modality = getattr(self.config.DATA, 'data_modality', 'video')
        seed = getattr(self.config.SYSTEM, 'seed', 42)
        g = torch.Generator()
        g.manual_seed(seed)

        # HuggingFace models (VideoMAE)
        if self.config.MODEL.type == "huggingface":
            self._train_huggingface(model, train_dataset, val_dataset)
            return

        if is_ddp:
                torch.cuda.set_device(rank)
                self.device = torch.device(f"cuda:{rank}")
        else:
                self.device = select_device(self.config.SYSTEM)
        # Custom models (MV or Tracking)
        self.model = model.to(self.device)

        if is_ddp:
            self.model = DDP(self.model, device_ids=[rank])

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
                # -------------------------
        # TRAIN SAMPLER
        # -------------------------
        if self.config.TRAIN.use_weighted_sampler:
            sample_weights = train_dataset.get_sample_weights()

            samples_per_class = getattr(self.config.TRAIN, 'samples_per_class', None)
            if samples_per_class:
                num_classes = train_dataset.num_classes()
                num_samples = samples_per_class * num_classes
            else:
                num_samples = len(sample_weights)

            if is_ddp:
                train_sampler = DistributedWeightedSampler(
                    weights=sample_weights,
                    num_replicas=world_size,
                    rank=rank,
                    replacement=True,
                    num_samples=num_samples,
                    seed=self.config.TRAIN.seed
                )
            else:
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=num_samples,
                    replacement=True,
                    generator=g
                )

            shuffle = False

        else:
            if is_ddp:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True
                )
            else:
                train_sampler = None

            shuffle = not is_ddp


        # -------------------------
        # VAL SAMPLER (SEPARATE!)
        # -------------------------
        if is_ddp:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False
            )
        else:
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.DATA.train.dataloader.batch_size,
            shuffle=(train_sampler is None and shuffle),
            sampler=train_sampler,
            num_workers=self.config.DATA.train.dataloader.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.DATA.valid.dataloader.batch_size,
            shuffle=False,
            sampler=val_sampler,   # ← IMPORTANT
            num_workers=self.config.DATA.valid.dataloader.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
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
                "backbone": self.config.MODEL.backbone.type,
                "aggregation": self.config.MODEL.neck.agr_type,
                "lr": self.config.TRAIN.optimizer.lr,
                "batch_size": self.config.DATA.train.dataloader.batch_size,
                #"num_classes": self.config.DATA.num_classes
            },
            patience=getattr(self.config.TRAIN, "patience", 0),
            monitor=getattr(self.config.TRAIN, "monitor", "balanced_accuracy"),
            mode=getattr(self.config.TRAIN, "mode", "max"),
        )

        self.trainer.train(epoch_start=self.epoch, save_every=self.config.TRAIN.save_every)

    def _train_huggingface(self, model, train_dataset, val_dataset):
        """Handle HuggingFace Trainer for VideoMAE."""
        from soccernetpro.core.sampler.weighted_sampler import WeightedTrainer, VideoMAETrainer

        self.model = model

        args = TrainingArguments(
            label_names=["labels"],
            output_dir=self.config.TRAIN.save_dir,
            per_device_train_batch_size=self.config.DATA.train.dataloader.batch_size,
            per_device_eval_batch_size=self.config.DATA.valid.dataloader.batch_size,
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
        #############
        train_metrics = self.hf_trainer.evaluate(train_dataset, metric_key_prefix="train")
        print("TRAIN METRICS:", train_metrics)
        #############

    def infer(self, test_dataset, rank=0, world_size=1):
        if self.config.MODEL.type == "huggingface":

            args = TrainingArguments(
            output_dir=self.config.TRAIN.save_dir,  # any directory, not used here
            per_device_eval_batch_size=1#self.config.DATA.valid.dataloader.batch_size,  # or whatever batch size you want
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
            from torch.nn.parallel import DistributedDataParallel as DDP
            from torch.utils.data.distributed import DistributedSampler
            
            is_ddp = world_size > 1

            if is_ddp:
                torch.cuda.set_device(rank)
                self.device = torch.device(f"cuda:{rank}")
            else:
                self.device = select_device(self.config.SYSTEM)

            # model
            self.model = self.model.to(self.device)
            if is_ddp:
                self.model = DDP(self.model, device_ids=[rank]) 
                test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)
            else:
                test_sampler = None

            modality = getattr(self.config.DATA, 'data_modality', 'video')
            collate_fn = tracking_collate_fn if modality == "tracking_parquet" else None

            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.config.DATA.test.dataloader.batch_size, 
                shuffle=False, 
                sampler=test_sampler,
                num_workers=self.config.DATA.test.dataloader.num_workers, 
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
                    "backbone": self.config.MODEL.backbone.type,
                    "aggregation": self.config.MODEL.neck.agr_type,
                    "lr": self.config.TRAIN.optimizer.lr,
                    "batch_size": self.config.DATA.train.dataloader.batch_size,
                    #"num_classes": self.config.DATA.num_classes
                },
                monitor=getattr(self.config.TRAIN, "monitor", "balanced_accuracy"),
                mode=getattr(self.config.TRAIN, "mode", "max"),
            )
            loss, metrics = self.test_trainer.test(
                detailed_results=getattr(self.config.TRAIN, 'detailed_results', False)
            )
            
        return metrics

    def evaluate(self, pred_path, gt_path, class_names, exclude_labels=[]):

        label_to_idx = {v: k for k, v in class_names.items()}

        with open(pred_path) as f:
            pred_data = json.load(f)

        with open(gt_path) as f:
            gt_data = json.load(f)

        gt_dict = {}
        for item in gt_data["data"]:
            sid = item["id"]
            gt_label = item["labels"]["action"]["label"]
            if gt_label not in exclude_labels:
                gt_dict[sid] = label_to_idx[gt_label]

        preds = []
        labels = []

        for item in pred_data["data"]:
            sid = item["id"]
            if sid not in gt_dict:
                continue

            pred_label = item["labels"]["action"]["label"]

            preds.append(label_to_idx[pred_label])
            labels.append(gt_dict[sid])

        metrics = self.compute_metrics(
            (preds, labels),
            mode="labels"
        )

        print(metrics)
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

