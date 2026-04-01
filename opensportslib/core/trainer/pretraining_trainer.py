# opensportslib/core/trainer/pretraining_trainer.py

"""SSL pre-training trainers for MAE, DINO, and SimCLR.

Provides a base trainer with a method-agnostic training loop, plus
three method-specific subclasses that implement the forward pass and
loss computation. Trainer_Pretraining is the top-level dispatcher
consumed by the API layer.
"""

import os
import logging
import time

import torch
import tqdm
import wandb
import numpy as np

from torch.utils.data import DataLoader
from opensportslib.core.utils.checkpoint import save_checkpoint, load_checkpoint
from opensportslib.core.utils.config import select_device
from opensportslib.core.utils.seed import seed_worker
import torch.distributed as dist


# -------------------------------------------------------------------
# base pretraining trainer
# -------------------------------------------------------------------

class BaseTrainerPretraining:
    """Method-agnostic training loop for self-supervised pre-training.

    Handles epoch iteration, gradient updates, DDP synchronisation,
    W&B logging, and checkpoint saving. Subclasses override
    _forward_batch() to implement method-specific logic.

    Args:
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set (may be None).
        model: the SSL model (already on device).
        optimizer: PyTorch optimizer.
        scheduler: learning-rate scheduler.
        save_dir: root directory for checkpoint output.
        model_name: name used for the checkpoint sub-directory.
        max_epochs: maximum number of training epochs.
        device: torch.device or device string.
        warmup_epochs: number of linear warmup epochs.
        save_backbone_only: if True, only save the encoder backbone.
        config: full configuration namespace.
    """

    def __init__(
        self,
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        save_dir,
        model_name,
        max_epochs=200,
        device="cuda",
        warmup_epochs=10,
        save_backbone_only=True,
        config=None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.save_backbone_only = save_backbone_only
        self.config = config

        self.best_checkpoint_path = None
        self.best_loss = float("inf")

        self.rank = dist.get_rank() if dist.is_initialized() else 0

        os.makedirs(self.save_dir, exist_ok=True)

        try:
            if self.rank == 0 and wandb.run is not None:
                wandb.watch(self.model, log="gradients", log_freq=100)
        except Exception:
            pass

    # -- abstract forward pass --------------------------------------

    def _forward_batch(self, batch):
        """Run the method-specific forward pass.

        Must be overridden by every subclass.

        Args:
            batch: a dict produced by the DataLoader.

        Returns:
            a scalar loss tensor.
        """
        raise NotImplementedError

    # -- training loop ----------------------------------------------

    def train(self, epoch_start=0, save_every=10):
        """Run the full pre-training loop.

        Args:
            epoch_start: the epoch number to start from.
            save_every: save a checkpoint every N epochs.
        """
        logging.info("Starting SSL pre-training")

        for epoch in range(epoch_start, self.max_epochs):
            logging.info(f"\nEpoch {epoch + 1}/{self.max_epochs}")

            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            disable = self.rank != 0
            pbar = tqdm.tqdm(
                total=len(self.train_loader), desc="Training",
                position=0, leave=True, disable=disable
            )
            train_loss = self._run_epoch(
                self.train_loader, epoch + 1, train=True, pbar=pbar
            )
            pbar.close()

            # validation
            val_loss = None
            if self.val_loader is not None:
                pbar = tqdm.tqdm(
                    total=len(self.val_loader), desc="Valid",
                    position=1, leave=True, disable=disable
                )
                val_loss = self._run_epoch(
                    self.val_loader, epoch + 1, train=False, pbar=pbar
                )
                pbar.close()

            # scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.rank == 0:
                if wandb.run is not None:
                    log_dict = {
                        "epoch": epoch + 1,
                        "lr": current_lr,
                        "train/loss": train_loss,
                    }
                    if val_loss is not None:
                        log_dict["valid/loss"] = val_loss
                    wandb.log(log_dict)

                logging.info(
                    f"Train Loss: {train_loss:.6f}"
                    + (f" | Val Loss: {val_loss:.6f}" if val_loss is not None else "")
                    + f" | LR: {current_lr:.2e}"
                )

            # checkpoint
            monitor_loss = val_loss if val_loss is not None else train_loss
            if self.rank == 0:
                if monitor_loss < self.best_loss:
                    self.best_loss = monitor_loss
                    best_path = self._save_checkpoint("best", epoch + 1, tag="best")
                    self.best_checkpoint_path = best_path

                    if wandb.run is not None:
                        artifact = wandb.Artifact("ssl-checkpoint", type="model")
                        artifact.add_file(best_path)
                        wandb.log_artifact(artifact)

                if (epoch + 1) % save_every == 0:
                    self._save_checkpoint(str(epoch + 1), epoch + 1)

        if self.rank == 0:
            logging.info(f"Best checkpoint: {self.best_checkpoint_path}")
            logging.info("Pre-training finished.")

    # -- single epoch logic -----------------------------------------

    def _run_epoch(self, dataloader, epoch, train=False, pbar=None):
        """Execute one pass over a dataloader.

        Args:
            dataloader: the DataLoader to iterate over.
            epoch: current epoch number.
            train: if True, compute gradients and update weights.
            pbar: optional tqdm progress bar.

        Returns:
            average loss for the epoch.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_batches = 0

        for batch in dataloader:
            if pbar:
                pbar.update()

            with torch.set_grad_enabled(train):
                loss = self._forward_batch(batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)

        # DDP average
        if dist.is_initialized():
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        return avg_loss

    # -- checkpoint saving -------------------------------------------

    def _save_checkpoint(self, filename, epoch, tag=None):
        """Save a training checkpoint.

        Args:
            filename: sub-directory name under save_dir.
            epoch: current epoch number.
            tag: optional prefix for the checkpoint file name.

        Returns:
            path to the saved checkpoint file.
        """
        epoch_dir = os.path.join(self.save_dir, str(filename))
        os.makedirs(epoch_dir, exist_ok=True)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        if self.save_backbone_only and hasattr(model_to_save, "get_encoder"):
            state_dict = model_to_save.get_encoder().state_dict()
        else:
            state_dict = model_to_save.state_dict()

        state = {
            "epoch": epoch,
            "state_dict": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()

        name = f"epoch_{epoch}.pt"
        if tag:
            name = f"{tag}_epoch_{epoch}.pt"

        path = os.path.join(epoch_dir, name)
        torch.save(state, path)
        logging.info(f"Saved checkpoint: {path}")
        return path


# -------------------------------------------------------------------
# method-specific trainers
# -------------------------------------------------------------------

class MAETrainerPretraining(BaseTrainerPretraining):
    """Forward pass for Masked Autoencoder pre-training.

    Expects batches with pixel_values of shape (B, C, T, H, W).
    Loss = MSE on masked patches only.
    """

    def _forward_batch(self, batch):
        clips = batch["pixel_values"].to(self.device).float()
        output = self.model(clips)

        pred = output["pred"]
        target = output["target"]
        mask = output["mask"]

        # MSE loss on masked patches only
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # per-patch mean
        loss = (loss * mask).sum() / mask.sum()
        return loss


class DINOTrainerPretraining(BaseTrainerPretraining):
    """Forward pass for DINO self-distillation pre-training.

    Expects batches with pixel_values as a list of 2 views.
    The model computes and returns the loss internally.
    """

    def _forward_batch(self, batch):
        views = [v.to(self.device).float() for v in batch["pixel_values"]]
        output = self.model(views)

        # update teacher EMA after each step
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.update_teacher()

        return output["loss"]


class ContrastiveTrainerPretraining(BaseTrainerPretraining):
    """Forward pass for SimCLR contrastive pre-training.

    Expects batches with pixel_values as a list of 2 views.
    The model computes and returns the NT-Xent loss internally.
    """

    def _forward_batch(self, batch):
        views = [v.to(self.device).float() for v in batch["pixel_values"]]
        output = self.model(views)
        return output["loss"]


# -------------------------------------------------------------------
# unified trainer dispatcher
# -------------------------------------------------------------------

class Trainer_Pretraining:
    """High-level trainer that dispatches to the right SSL method trainer.

    Consumed by PretrainingAPI. Responsible for building data loaders,
    optimizers, schedulers, then delegating to the method-specific
    trainer subclass.

    Args:
        config: the configuration object.
    """

    def __init__(self, config):
        self.config = config
        self.device = select_device(self.config.SYSTEM)
        self.model = None
        self.trainer = None

    def train(self, model, train_dataset, val_dataset=None, rank=0, world_size=1):
        """Build all training components and run the pre-training loop.

        Args:
            model: the SSL model.
            train_dataset: training PretrainingDataset.
            val_dataset: validation PretrainingDataset (optional).
            rank: GPU rank.
            world_size: total number of GPUs.
        """
        from opensportslib.core.optimizer.builder import build_optimizer
        from opensportslib.core.scheduler.builder import build_scheduler
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler

        is_ddp = world_size > 1
        seed = self.config.SYSTEM.seed
        ssl_method = self.config.SSL.method.lower()

        g = torch.Generator()
        g.manual_seed(seed)

        if is_ddp:
            torch.cuda.set_device(rank)
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = select_device(self.config.SYSTEM)

        self.model = model.to(self.device)

        if is_ddp:
            self.model = DDP(self.model, device_ids=[rank])

        # build optimizer and scheduler
        optimizer = build_optimizer(
            self.model.parameters(), cfg=self.config.TRAIN.optimizer
        )
        scheduler = build_scheduler(
            optimizer, cfg=self.config.TRAIN.scheduler,
            default_args={"len_train_loader": len(train_dataset)}
        )

        # samplers
        if is_ddp:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size,
                rank=rank, shuffle=True, drop_last=True
            )
        else:
            train_sampler = None

        val_sampler = None
        if val_dataset is not None and is_ddp:
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size,
                rank=rank, shuffle=False, drop_last=False
            )

        # collate function for multi-view methods
        collate_fn = _multi_view_collate if ssl_method in ("dino", "simclr") else None

        num_workers = getattr(self.config.DATA, "num_workers", 4)
        batch_size = self.config.DATA.batch_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                worker_init_fn=seed_worker,
                generator=g,
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None,
            )

        # select method-specific trainer
        if ssl_method == "mae":
            TrainerClass = MAETrainerPretraining
        elif ssl_method == "dino":
            TrainerClass = DINOTrainerPretraining
        elif ssl_method in ("simclr", "contrastive"):
            TrainerClass = ContrastiveTrainerPretraining
        else:
            raise ValueError(f"Unsupported SSL method: {ssl_method}")

        warmup_epochs = getattr(self.config.TRAIN, "warmup_epochs", 10)
        save_backbone_only = getattr(self.config.TRAIN, "save_backbone_only", True)
        save_every = getattr(self.config.TRAIN, "save_every", 10)

        self.trainer = TrainerClass(
            train_loader=train_loader,
            val_loader=val_loader,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=self.config.SYSTEM.save_dir,
            model_name=self.config.MODEL.encoder.type,
            max_epochs=self.config.TRAIN.epochs,
            device=self.device,
            warmup_epochs=warmup_epochs,
            save_backbone_only=save_backbone_only,
            config=self.config,
        )

        self.trainer.train(epoch_start=0, save_every=save_every)
        return getattr(self.trainer, "best_checkpoint_path", None)

    def load(self, path):
        """Load a checkpoint.

        Args:
            path: path to checkpoint file.

        Returns:
            tuple (model, optimizer, scheduler, epoch).
        """
        from opensportslib.models.builder import build_model
        if self.model is None:
            self.model = build_model(self.config, self.device)
        self.model, optimizer, scheduler, epoch = load_checkpoint(
            self.model, path, device=self.device
        )
        logging.info(f"Loaded checkpoint from {path}, epoch: {epoch}")
        return self.model, optimizer, scheduler, epoch


def _multi_view_collate(batch):
    """Custom collate for multi-view SSL methods (DINO, SimCLR).

    Stacks each view separately so the output pixel_values is a
    list of tensors rather than a single tensor.
    """
    view1 = torch.stack([item["pixel_values"][0] for item in batch])
    view2 = torch.stack([item["pixel_values"][1] for item in batch])
    ids = [item["id"] for item in batch]
    return {"pixel_values": [view1, view2], "id": ids}
