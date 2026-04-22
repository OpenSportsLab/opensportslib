# opensportslib/core/trainer/classification_trainer.py

"""classification trainers for video and tracking modalities.

provides a base trainer with modality-agnostic training, validation,
and test loops, plus two modality-specific subclasses that implement
the forward pass. Trainer_Classification is the top-level dispatcher 
consumed by the API layer.

"""

import os
import gc
import json
import time
import logging

import torch
import tqdm
import wandb
import numpy as np

from torch.utils.data import (
    DataLoader, 
    WeightedRandomSampler,
)

from transformers import Trainer as HFTrainer, TrainingArguments
from opensportslib.core.utils.ddp import DistributedWeightedSampler

from opensportslib.core.utils.wandb import log_confusion_matrix_wandb
from opensportslib.core.utils.checkpoint import *

from opensportslib.core.utils.config import select_device
from opensportslib.core.utils.data import mixup_data
import torch.distributed as dist
from datetime import datetime
from opensportslib.core.utils.seed import seed_worker
from opensportslib.metrics.classification_metric import (
    compute_classification_metrics,
    process_preds_labels
)

# -------------------------------------------------------------------
# base classification trainer
# -------------------------------------------------------------------

class BaseTrainerClassification:
    """modality-agnostic training loop for classification.

    handles epoch iteration, gradient updates, DDP gather, 
    metric computation, W&B logging, checkpoint saving, and JSON
    prediction export. subclasses only need to override _forward_batch()
    with modality-specific tensor preparation.

    Args:
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        test_loader: DataLoader for test set (may be None during training).
        model: the classification model (already on device).
        optimizer: PyTorch optimizer.
        scheduler: learning-rate scheduler.
        criterion: loss function callable.
        class_weights: optional per-class weight tensor for the loss.
        class_names: dict mapping class indices to names.
        save_dir: root directory for checkpoint and prediction output.
        model_name: name used for the checkpoint sub-directory.
        max_epochs: maximum number of training epochs.
        device: torch.device or device string.
        top_k: k value for top-k accuracy computation.
        wandb_project: W&B project name.
        wandb_run_name: W&B run display name.
        wandb_config: dict of hyperparameters logged to W&B.
        patience: early-stopping patience (0=disabled).
        monitor: metric name to monitor for checkpointing.
        mode: "max" or "min" depending on the monitored metric.
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
        patience=10,
        monitor="balanced_accuracy",
        mode="max",
        revert_on_lr_reduction=False,
        config=None,
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
        self.config = config

        self.best_checkpoint_path = None
        self.best_metric = None
        self.revert_on_lr_reduction = revert_on_lr_reduction
        self._best_model_state = None
        self.predictions_payload = None
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        try:
            if self.rank == 0:
                wandb.watch(self.model, log="gradients", log_freq=100)
        except Exception:
            pass

    # -- abstract forward pass --------------------------------------
    
    def _forward_batch(self, batch):
        """run the modality-specific forward pass.

        must be overridden by every subclass.

        Args:
            batch: a dict produced by the DataLoader.

        Returns:
            a tuple (logits, labels) where both are tensors on
            self.device.
        """
        raise NotImplementedError

    # -- process batch ----------------------------------------------

    def _process_batch(self, batch, train):
        """run forward pass, compute loss, and optionally update weights.

        the default implementation calls _forward_batch() for the
        modality-specific forward pass, then computes the loss and
        runs the backward step.  subclasses may override this entirely
        to inject AMP, mixup, or other training-time modifications
        without touching the base training loop.

        Args:
            batch: a dict produced by the DataLoader.
            train: if True, compute gradients and update weights.

        Returns:
            a tuple (logits, labels, loss).
        """
        has_labels = "labels" in batch or "label" in batch
        with torch.set_grad_enabled(train):
            logits, labels = self._forward_batch(batch)
            if labels is None:
                    has_labels = False
            loss = None
            if has_labels:
                if self.class_weights is not None:
                    loss = self.criterion(
                        output=logits, labels=labels,
                        weight=self.class_weights.to(self.device)
                    )
                else:
                    loss = self.criterion(output=logits, labels=labels)

            if train and loss is not None:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        return logits, labels, loss, has_labels

    # -- training loop ----------------------------------------------

    def train(self, epoch_start=0, save_every=3):
        """run the full training loop with validation after each epoch.

        Args:
            epoch_start: the epoch number to start from (0-based).
            save_every: currently unused; reserved for periodic 
            checkpoint saving.
        """
        logging.info("Starting training")
        monitor = self.monitor
        mode = self.mode
        best_metric = -float("inf") if mode == "max" else float("inf")
        best_path = None

        for epoch in range(epoch_start, self.max_epochs):
            logging.info(f"\nEpoch {epoch+1}/{self.max_epochs}")

            # --- train ---
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            
            disable = self.rank != 0
            
            pbar = tqdm.tqdm(
                total=len(self.train_loader), desc="Training", 
                position=0, leave=True, disable=disable
            )
            _, _, train_loss, train_metrics = self._run_epoch(
                self.train_loader, epoch + 1, 
                train=True, set_name="train", pbar=pbar
            )
            pbar.close()

            # --- validation ---
            pbar = tqdm.tqdm(
                total=len(self.val_loader), desc="Valid", 
                position=1, leave=True, disable=disable
            )
            _, _, val_loss, val_metrics = self._run_epoch(
                self.val_loader, epoch + 1, 
                train=False, set_name="valid", pbar=pbar
            )
            pbar.close()

            prev_lr = self.optimizer.param_groups[0]["lr"]

            # capture LR before the scheduler step so we can detect
            # plateau-triggered reductions.
            val_metric = val_metrics.get(
                "balanced_accuracy", val_metrics.get("accuracy", 0)
            )
            train_metric = train_metrics.get(
                "balanced_accuracy", train_metrics.get("accuracy", 0)
            )

            # ReduceLROnPlateau needs the monitored metric
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # early stopping: mirror pixels_vs_positions behavior
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                min_lr = self.scheduler.min_lrs[0]
                if current_lr <= 2 * min_lr:
                    if self.rank == 0:
                        logging.info(
                            f"Early stopping at epoch {epoch+1}: "
                            f"lr {current_lr:.2e} <= 2 * min_lr {min_lr:.2e}"
                        )
                    break

            # When ReduceLROnPlateau drops the LR, revert weights to
            # the best checkpoint so training continues from the
            # strongest point rather than from a potentially overfit
            # state. This mirrors the pixels_vs_positions recipe.
            if (
                self.revert_on_lr_reduction
                and current_lr != prev_lr
                and self._best_model_state is not None
            ):
                self.model.load_state_dict(self._best_model_state)
                print(
                    f"LR reduced from {prev_lr:.2e} to {current_lr:.2e} "
                    f"-- reverted to best model"
                )

            if self.rank == 0:
                # ---------------- W&B LOG ----------------
                if wandb.run is not None:
                    wandb.log({
                        "epoch": epoch + 1,
                        "lr": current_lr,
                        "train/loss": train_loss,
                        "valid/loss": val_loss,
                        **{f"train/{k}": v for k, v in train_metrics.items()},
                        **{f"valid/{k}": v for k, v in val_metrics.items()},
                    })

                logging.info(f"Train Loss: {train_loss:.4f} | Train Bal Acc: {train_metric:.4f}")
                logging.info(f"Val Loss: {val_loss:.4f} | Val Bal Acc: {val_metric:.4f}")

            # ---------------- CHECKPOINT ----------------
            current = val_loss if monitor == "loss" else val_metrics.get(monitor, 0)

            is_better = current > best_metric if mode == "max" else current < best_metric

            if is_better and self.rank == 0:
                best_metric = current
                self.best_metric = best_metric

                if self.revert_on_lr_reduction:
                    self._best_model_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }

                best_path = self._save_checkpoint("best", epoch + 1, tag="best")
                self.best_checkpoint_path = best_path

                if wandb.run is not None:
                    artifact = wandb.Artifact("model-checkpoint", type="model")
                    artifact.add_file(best_path)
                    wandb.log_artifact(artifact)
            
        if self.rank == 0:
            logging.info(f"Best checkpoint : {self.best_checkpoint_path}")
            logging.info("Training finished.")

    
    # -- TEST evaluation ------------------------------------------
    
    def test(self, epoch=None, detailed_results=False):
        """run the test set evaluation.

        Args:
            epoch: the epoch number to evaluate (if None, uses "final").
            detailed_results: whether to compute detailed classification metrics.

        Returns:
            a tuple (test_loss, test_metrics).
        """
        logging.info("\nRunning TEST evaluation")
        pbar = tqdm.tqdm(
            total=len(self.test_loader), desc="Test", position=0, 
            leave=True, disable = self.rank != 0
        )
        all_logits, all_labels, test_loss, test_metrics = self._run_epoch(
            self.test_loader,
            epoch if epoch is not None else "final",
            train=False, set_name="test", pbar=pbar
        )
        pbar.close()

        if self.rank==0:
            if wandb.run is not None:
                wandb.log({
                    "test/loss": test_loss,
                    **{f"test/{k}": v for k, v in test_metrics.items()},
                })

            if detailed_results:
                from opensportslib.metrics.classification_metric import (
                    compute_detailed_classification_metrics
                )
                compute_detailed_classification_metrics(
                    all_logits=all_logits, all_labels=all_labels,
                    class_names=self.class_names, save_dir=self.save_dir,
                    set_name="test"
                )

        logging.info(f"TEST METRICS : {test_metrics}")
        return test_loss, test_metrics

    # -- single epoch logic -------------------------------------

    def _run_epoch(self, dataloader, epoch, train=False, set_name="train", pbar=None):
        """execute one pass over a dataloader.

        handles forward/backward, per-batch bookkeeping, DDP gather, metric
        computation, confusion-matrix logging, and JSON prediction export.

        Args:
            dataloader: the DataLoader to iterate over.
            epoch: the epoch number (for checkpointing and folder naming).
            train: if True, compute gradients and update weights.
            set_name: "train", "valid", or "test" (for logging and JSON).
            pbar: optional tqdm progress bar.

        Returns:
            a tuple (all_logits, all_labels, avg_loss, metrics).
            on non-rank-0 DDP workers the first two are None and metrics
            is an empty dict.
        """
        
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
        save_path = os.path.join(
            epoch_dir, f"predictions_{set_name}_epoch_{epoch}.json"
        )

        # --- batch loop ---
        for batch in dataloader:
            if pbar:
                pbar.update()

            logits, labels, loss, has_labels = self._process_batch(batch, train)

            if loss is not None:
                total_loss += loss.item()
                total_batches += 1

            logits_cpu = logits.detach().cpu()
            all_logits.append(logits_cpu)

            if has_labels:
                labels_cpu = labels.detach().cpu()
                all_labels.append(labels_cpu)

            # per-sample predictions for JSON export.
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

        # --- concatenate local predictions ---
        if len(all_logits) > 0:
            all_logits = torch.cat(all_logits).numpy()
        else:
            all_logits = np.zeros((0, 1))

        if len(all_labels) > 0:
            all_labels = torch.cat(all_labels).numpy()
        else:
            all_labels = np.zeros((0,))

        # --- DDP gather (handles uneven shard sizes) ---
        if dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, (all_logits, all_labels, results))

            if self.rank == 0:
                all_logits = np.concatenate([g[0] for g in gathered])
                all_labels = np.concatenate([g[1] for g in gathered])
                results = [r for g in gathered for r in g[2]]
            else:
                self.predictions_payload = None
                return None, None, 0.0, {}

        # --- metrics (rank-0 only in DDP) ---
        if len(all_labels) > 0:
            metrics = compute_classification_metrics(
                (all_logits, all_labels), top_k=self.top_k,
            )
        else:
            metrics = {}

        # --- confusion matrix (validation and test only) ---
        if self.rank == 0 and set_name in ["valid", "test"] and len(all_labels) > 0:
            preds_all, labels_all, _ = process_preds_labels(
                (all_logits, all_labels)
            )
            class_names = [
                self.class_names[i] for i in sorted(self.class_names.keys())
            ]

            log_confusion_matrix_wandb(
                y_true=labels_all.tolist(),
                y_pred=preds_all.tolist(),
                class_names=class_names,
                split_name=set_name,
            )

        # --- save JSON (rank-0 only) ---
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

            logging.info(f"RESULTS Length: {len(results)}")
            logging.info(f"Predicitions are stored at : {save_path}")
            with open(save_path, "w") as f:
                json.dump(submission, f, indent=2)
            self.predictions_payload = submission

        return all_logits, all_labels, total_loss / max(1, total_batches), metrics


    # -- checkpoint saving ---------------------------------------
    
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
            "best_metric": self.best_metric,
        }

        if hasattr(self, "scaler"):
            state["scaler"] = self.scaler.state_dict()

        name = f"epoch_{epoch}.pt"
        if tag:
            name = f"{tag}_epoch_{epoch}.pt"

        path_aux = os.path.join(epoch_dir, name)
        torch.save(state, path_aux)
        logging.info(f"Saved checkpoint: {path_aux}")
        return path_aux


# --------------------------------------------------------------
# modality-specific trainers 
# --------------------------------------------------------------

class MVTrainerClassification(BaseTrainerClassification):
    """forward pass for multi-view video classification.
    
    expects batches with pixel_values of shape
    (B, V, C, T, H, W) and integer labels of shape (B,).
    """

    def _forward_batch(self, batch):
        """move video clips to device and run the model.

        Args:
            batch: dict with keys "pixel_values" and "labels".

        Returns:
            a tuple (logits, labels) on self.device.
        """
        mvclips = batch["pixel_values"].to(self.device).float()
        labels = batch.get("labels", None)
        if labels is not None:
            labels = labels.to(self.device)

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
    """forward pass for tracking-based classification.
    
    expects batches with x of shape (B, N, 2), edge_index of shape (2, E),
    batch of shape (B,), batch_size, seq_len, and integer labels of shape (B,).
    """

    def _forward_batch(self, batch):
        """move tracking data to device and run the model.

        Args:
            batch: dict with keys "x", "edge_index", "batch", "batch_size",
            "seq_len", and "labels".

        Returns:
            a tuple (logits, labels) on self.device.
        """
        tracking_batch = {
            "x": batch["x"].to(self.device),
            "edge_index": batch["edge_index"].to(self.device),
            "batch": batch["batch"].to(self.device),
            "batch_size": batch["batch_size"],
            "seq_len": batch["seq_len"],
        }
        labels = batch.get("labels", None)
        if labels is not None:
            labels = labels.to(self.device)
        
        logits = self.model(tracking_batch)
        
        return logits, labels

class FramesTrainerClassification(BaseTrainerClassification):
    """forward pass for frames_npy video classification.

    supports optional mixed-precision training (AMP) and mixup
    augmentation, controlled via config.TRAIN.use_amp and
    config.TRAIN.mixup_alpha respectively.

    expects batches with pixel_values of shape (B, T, H, W, C)
    and integer labels of shape (B,).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        self.use_amp = getattr(cfg.TRAIN, "use_amp", False) if cfg else False
        self.mixup_alpha = getattr(cfg.TRAIN, "mixup_alpha", 0.0) if cfg else 0.0
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _forward_batch(self, batch):
        pixel_values = batch["pixel_values"].to(self.device).float()
        labels = batch["labels"].to(self.device)
        logits = self.model({"pixel_values": pixel_values})
        return logits, labels

    def _process_batch(self, batch, train):
        pixel_values = batch["pixel_values"].to(self.device).float()
        labels = batch["labels"].to(self.device)

        with torch.set_grad_enabled(train):
            use_mixup = (
                train
                and self.mixup_alpha > 0
                and np.random.random() > 0.5
            )

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if use_mixup:
                    pixel_values, labels_a, labels_b, lam = mixup_data(
                        pixel_values, labels, self.mixup_alpha
                    )
                    logits = self.model({"pixel_values": pixel_values})
                    loss = (
                        lam * self.criterion(output=logits, labels=labels_a)
                        + (1 - lam) * self.criterion(output=logits, labels=labels_b)
                    )
                    labels = labels_a
                else:
                    logits = self.model({"pixel_values": pixel_values})
                    if self.class_weights is not None:
                        loss = self.criterion(
                            output=logits, labels=labels,
                            weight=self.class_weights.to(self.device),
                        )
                    else:
                        loss = self.criterion(output=logits, labels=labels)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return logits, labels, loss, True

# --------------------------------------------------------------
# unified trainer dispatcher 
# --------------------------------------------------------------

class Trainer_Classification:
    """high-level trainer that dispatches to the right modality trainer.

    consumed by ClassificationModel. Responsible for building data
    loaders, optimizers, schedulers, and samplers, then delegating the
    actual loop to MVTrainerClassification or TrackingTrainerClassification.

    Args:
        config: the configuration object.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = select_device(self.config.SYSTEM)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.trainer = None
        self.predictions_payload = None

    def compute_metrics(self, pred, mode="logits"):
        """thin wrapper around the metric module.

        Args:
            pred: a tuple (logits, labels).
            mode: "logits" or "labels" (default: "logits").

        Returns:
            a dictionary of classification metrics.
        """
        return compute_classification_metrics(
            pred, top_k=2, mode=mode
        )

    # -- training -----------------------------------------------

    def train(self, model, train_dataset, val_dataset=None, rank=0, world_size=1):
        """build all training components and run the loop.

        detects the model type (HuggingFace vs. custom) and the data 
        modality (video vs. tracking) to select the right trainer class,
        sampler, and collate function.

        Args:
            model: the classification model.
            train_dataset: training ClassificationDataset.
            val_dataset: validation ClassificationDataset (optional).
            rank: GPU rank (0-indexed).
            world_size: total number of GPUs.
        """
        from opensportslib.core.loss.builder import build_criterion
        from opensportslib.core.optimizer.builder import build_optimizer
        from opensportslib.core.scheduler.builder import build_scheduler
        from opensportslib.core.utils.data import tracking_collate_fn
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler

        is_ddp = world_size > 1
        modality = getattr(self.config.DATA, 'data_modality', 'video')
        seed = self.config.SYSTEM.seed

        g = torch.Generator()
        g.manual_seed(seed)

        # HuggingFace models (e.g. VideoMAE) use the HF Trainer.
        if self.config.MODEL.type == "huggingface":
            self._train_huggingface(model, train_dataset, val_dataset)
            return

        if is_ddp:
                torch.cuda.set_device(rank)
                self.device = torch.device(f"cuda:{rank}")
        else:
                self.device = select_device(self.config.SYSTEM)
        
        self.model = model.to(self.device)

        if is_ddp:
            self.model = DDP(self.model, device_ids=[rank])

        # Build components
        optimizer = build_optimizer(
            self.model.parameters(), cfg=self.config.TRAIN.optimizer
        )
        scheduler = build_scheduler(
            optimizer, cfg=self.config.TRAIN.scheduler
        )
        criterion = build_criterion(self.config.TRAIN.criterion)

        # --- class weights for the loss ---
        if self.config.TRAIN.use_weighted_loss:
            class_weights = train_dataset.get_class_weights(
                num_classes=train_dataset.num_classes(), sqrt=True
            ).to(self.device)
        else:
            class_weights = None

        # tracking modality needs a customm collate that merges PyG
        # Data objects into a single batched graph per timestamp.
        collate_fn = tracking_collate_fn if modality == "tracking_parquet" else None

        # --- train sampler ---
        if self.config.TRAIN.use_weighted_sampler:
            sample_weights = train_dataset.get_sample_weights()

            samples_per_class = getattr(
                self.config.TRAIN, 'samples_per_class', None
            )
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
                    seed=self.config.SYSTEM.seed
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


        # --- validation sampler ---
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

        num_train_workers = self.config.DATA.train.dataloader.num_workers
        num_val_workers = self.config.DATA.valid.dataloader.num_workers

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.DATA.train.dataloader.batch_size,
            shuffle=(train_sampler is None and shuffle),
            sampler=train_sampler,
            num_workers=num_train_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True,
            persistent_workers=num_train_workers > 0,
            prefetch_factor=4 if num_train_workers > 0 else None,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.DATA.valid.dataloader.batch_size,
            shuffle=False,
            sampler=val_sampler,   
            num_workers=num_val_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=num_val_workers > 0,
            prefetch_factor=4 if num_val_workers > 0 else None,
        )

        # select the modality-specific trainer.
        if modality == "tracking_parquet":
            TrainerClass = TrackingTrainerClassification
        elif modality == "frames_npy":
            TrainerClass = FramesTrainerClassification
        else:
            TrainerClass = MVTrainerClassification

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
            save_dir=self.config.SYSTEM.save_dir,
            model_name=self.config.MODEL.backbone.type,
            max_epochs=self.config.TRAIN.epochs,
            device=self.device,
            top_k=2,
            patience=getattr(self.config.TRAIN, "patience", 0),
            monitor=getattr(self.config.TRAIN, "monitor", "balanced_accuracy"),
            mode=getattr(self.config.TRAIN, "mode", "max"),
            revert_on_lr_reduction=(modality in ("tracking_parquet", "frames_npy")),
            config=self.config,
        )

        self.trainer.train(epoch_start=self.epoch, save_every=self.config.TRAIN.save_every)
        return getattr(self.trainer, "best_checkpoint_path", None)

    def _train_huggingface(self, model, train_dataset, val_dataset):
        """Handle HuggingFace Trainer for VideoMAE."""
        from opensportslib.core.sampler.weighted_sampler import WeightedTrainer, VideoMAETrainer

        self.model = model

        args = TrainingArguments(
            label_names=["labels"],
            output_dir=self.config.SYSTEM.save_dir,
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
        logging.info(f"TRAIN METRICS: {train_metrics}")
        #############

    def infer(self, test_dataset, rank=0, world_size=1):
        if self.config.MODEL.type == "huggingface":

            args = TrainingArguments(
            output_dir=self.config.SYSTEM.save_dir,  # any directory, not used here
            per_device_eval_batch_size=1#self.config.DATA.valid.dataloader.batch_size,  # or whatever batch size you want
            )

            self.hf_trainer = HFTrainer(
                model=self.model,
                args=args,
                compute_metrics=self.compute_metrics  # optional, can compute later manually
            )

            preds_output = self.hf_trainer.predict(test_dataset)
            logits = preds_output.predictions
            if isinstance(logits, tuple):
                logits = logits[0]

            probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
            pred_idx = np.argmax(probs, axis=-1)
            confs = probs.max(axis=-1)

            class_names = test_dataset.label_map
            sample_ids = [item.get("id", str(i)) for i, item in enumerate(test_dataset.samples)]

            submission = {
                "version": "2.0",
                "task": "action_classification",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "metadata": {"type": "predictions"},
                "data": [],
            }

            for sid, label_idx, conf in zip(sample_ids, pred_idx, confs):
                submission["data"].append({
                    "id": sid,
                    "labels": {
                        "action": {
                            "label": class_names[int(label_idx)],
                            "confidence": float(conf),
                        }
                    },
                })

            out_dir = os.path.join(self.config.SYSTEM.save_dir, "final")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "predictions_test_epoch_final.json")
            with open(out_path, "w") as f:
                json.dump(submission, f, indent=2)
            self.predictions_payload = submission
            return submission
        
        else:
            from opensportslib.core.loss.builder import build_criterion
            from opensportslib.core.optimizer.builder import build_optimizer
            from opensportslib.core.scheduler.builder import build_scheduler
            from opensportslib.core.utils.data import tracking_collate_fn
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
            elif modality == "frames_npy":
                TrainerClass = FramesTrainerClassification
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
                save_dir=self.config.SYSTEM.save_dir,
                model_name=self.config.MODEL.backbone.type,
                max_epochs=self.config.TRAIN.epochs,
                device=self.device,
                top_k=2,
                monitor=getattr(self.config.TRAIN, "monitor", "balanced_accuracy"),
                mode=getattr(self.config.TRAIN, "mode", "max"),
                revert_on_lr_reduction=(modality in ("tracking_parquet", "frames_npy")),
                config=self.config,
            )
            self.test_trainer.test(
                detailed_results=getattr(self.config.TRAIN, 'detailed_results', False)
            )
            self.predictions_payload = getattr(
                self.test_trainer, "predictions_payload", None
            )
            return self.predictions_payload

    def evaluate(self, pred_path, gt_path, class_names, exclude_labels=[]):

        label_to_idx = {v: k for k, v in class_names.items()}

        if isinstance(pred_path, dict):
            pred_data = pred_path
        elif isinstance(pred_path, str):
            with open(pred_path) as f:
                pred_data = json.load(f)
        else:
            raise TypeError(
                f"Unsupported predictions type: {type(pred_path).__name__}. Expected dict or str."
            )

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
        return metrics


    def demo(self, model, video_paths):
        pass

    def save(self, model, path, processor=None, tokenizer=None, optimizer=None, epoch=None):
        """
        Save model checkpoint
        """
        save_checkpoint(model, path, processor, tokenizer, optimizer, epoch)
        logging.info(f"Model saved at {path}")

    def load(self, path, optimizer=None, scheduler=None):
        """
        Load model checkpoint. Returns loaded model, optimizer, epoch
        """
        if self.config.MODEL.type == "huggingface":
            epoch = None
            self.model, processor = load_huggingface_checkpoint(self.config, path=path, device=self.device)
            logging.info(f"Model loaded from {path}")
            return self.model, processor, scheduler, epoch
        else:
            from opensportslib.models.builder import build_model
            if self.model is None:
                self.model, _ = build_model(self.config, self.device)
            self.model, optimizer, scheduler, epoch = load_checkpoint(
                self.model, path, optimizer, scheduler, device=self.device
            )
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.epoch = epoch
            logging.info(f"Model loaded from {path}, epoch: {epoch}")
            return self.model, self.optimizer, self.scheduler, self.epoch
