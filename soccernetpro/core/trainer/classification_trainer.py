# soccernetpro/core/trainer.py

import torch
import numpy as np
import wandb
import json
import gc
import logging
import tqdm
import time
from torch.utils.data import DataLoader
from transformers import Trainer as HFTrainer, TrainingArguments
from soccernetpro.metrics.classification_metric import compute_classification_metrics, process_preds_labels
from soccernetpro.core.utils.wandb import log_attention_wandb, log_confusion_matrix_wandb, log_sample_videos_wandb 
from soccernetpro.core.utils.checkpoint import *
from soccernetpro.core.utils.config import select_device

class MVTrainerClassification:
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
        log_attention=False,
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
        self.log_attention = log_attention

        # ---------------- W&B INIT ----------------
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

    # =========================================================
    # TRAIN = Train + Validation ONLY
    # =========================================================
    def train(self, epoch_start=0, save_every=3):
        logging.info("start training")
        counter = 0

        for epoch in range(epoch_start, self.max_epochs):
            print(f"\nEpoch {epoch+1}/{self.max_epochs}")

            # ---------------- TRAIN ----------------
            pbar = tqdm.tqdm(total=len(self.train_loader), desc="Training", position=0, leave=True)
            _, train_loss, train_metrics = self._run_epoch(
                self.train_loader,
                epoch + 1,
                train=True,
                set_name="train",
                pbar=pbar
            )
            pbar.close()

            # ---------------- VALID ----------------
            pbar = tqdm.tqdm(total=len(self.val_loader), desc="Valid", position=1, leave=True)
            _, val_loss, val_metrics = self._run_epoch(
                self.val_loader,
                epoch + 1,
                train=False,
                set_name="valid",
                pbar=pbar
            )
            pbar.close()

            # ---------------- SCHEDULER ----------------
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            # ---------------- W&B LOG ----------------
            wandb.log({
                "epoch": epoch + 1,
                "lr": lr,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"valid/{k}": v for k, v in val_metrics.items()},
            })

            print("TRAIN METRICS:", train_metrics)
            print("VAL METRICS:", val_metrics)

            # ---------------- CHECKPOINT ----------------
            counter += 1
            if counter >= save_every:
                path = self._save_checkpoint(epoch + 1)

                artifact = wandb.Artifact("model-checkpoint", type="model")
                artifact.add_file(path)
                wandb.log_artifact(artifact)

                counter = 0

        print("Training finished.")

    # =========================================================
    # TEST = Separate Call
    # =========================================================
    def test(self, epoch=None):
        """
        Run test set evaluation.
        If epoch is provided, logs under that epoch number.
        """
        print("\nRunning TEST evaluation")
        pbar = tqdm.tqdm(total=len(self.test_loader), desc="Test", position=0, leave=True)
        _, test_loss, test_metrics = self._run_epoch(
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

        print("TEST METRICS:", test_metrics)
        return test_loss, test_metrics

    # =========================================================
    # CORE LOOP
    # =========================================================
    def _run_epoch(self, dataloader, epoch, train=False, set_name="train", pbar=None):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_batches = 0

        all_logits = []
        all_labels = []

        logged_samples = False

        # -------- Create epoch folder --------
        epoch_dir = os.path.join(self.save_dir, str(epoch))
        os.makedirs(epoch_dir, exist_ok=True)

        prediction_file = f"predictions_{set_name}_epoch_{epoch}.json"
        save_path = os.path.join(epoch_dir, prediction_file)

        data = {"Set": set_name}
        actions = {}

        for batch_idx, batch in enumerate(dataloader):
            mvclips, targets = batch["pixel_values"], batch["labels"]
            mvclips = mvclips.to(self.device).float()
            targets = targets.to(self.device)

            if pbar is not None:
                pbar.update()

            with torch.set_grad_enabled(train):
                outputs = self.model(mvclips)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    attention = outputs[1] if len(outputs) > 1 else None
                else:
                    logits = outputs
                    attention = None

                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)

                if self.class_weights is not None:
                    print("Using class weights for loss computation")
                    loss = self.criterion(output=logits, labels=targets, weight=self.class_weights.to(self.device))
                else:
                    loss = self.criterion(output=logits, labels=targets)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            all_logits.append(logits.detach().cpu())
            all_labels.append(targets.detach().cpu())

            # -------- JSON LOG --------
            preds = torch.argmax(logits.detach().cpu(), dim=1)
            for i in range(len(preds)):
                key = f"{batch_idx}_{i}"
                actions[key] = {
                    "Action class": int(preds[i].item())
                }

            # -------- Sample Video Log (once/epoch) --------
            # if not logged_samples and set_name in ["train", "valid"]:
            #     log_sample_videos_wandb(
            #         mvclips=mvclips,
            #         preds=preds.numpy(),
            #         labels=targets.detach().cpu().numpy(),
            #         split_name=set_name,
            #         max_samples=1
            #     )
            #     logged_samples = True

            # -------- Attention Log --------
            # if attention is not None and set_name in ["valid", "test"]:
            #     log_attention_wandb(attention, set_name)

        # -------- METRICS --------
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        metrics = compute_classification_metrics(
            (all_logits, all_labels),
            top_k=self.top_k
        )

        # -------- CONFUSION MATRIX --------
        preds_all, labels_all, _ = process_preds_labels(
            (all_logits, all_labels),
            top_k=None
        )

        if set_name in ["valid", "test"]:
            class_names = [self.class_names[i] for i in sorted(self.class_names.keys())] or [str(i) for i in range(all_logits.shape[1])]

            log_confusion_matrix_wandb(
                y_true=labels_all.tolist(),
                y_pred=preds_all.tolist(),
                class_names=class_names,
                split_name=set_name
            )

        gc.collect()
        torch.cuda.empty_cache()

        # -------- SAVE JSON --------
        data["Actions"] = actions
        with open(save_path, "w") as f:
            json.dump(data, f)

        return (
            save_path,
            total_loss / max(1, total_batches),
            metrics
        )

    # =========================================================
    # CHECKPOINT
    # =========================================================
    def _save_checkpoint(self, epoch):
        epoch_dir = os.path.join(self.save_dir, str(epoch))
        os.makedirs(epoch_dir, exist_ok=True)

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }

        path_aux = os.path.join(epoch_dir, f"{epoch}_model.pth.tar")
        torch.save(state, path_aux)
        print(f"Saved checkpoint: {path_aux}")
        return path_aux


class Trainer_Classification:
    """
    Unified Trainer that can either use native PyTorch loop or HuggingFace Trainer.
    """
    def __init__(self, config):
        self.config = config
        self.device = select_device(self.config.SYSTEM)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.hf_trainer = None

    def compute_metrics(self, pred):
        return compute_classification_metrics(pred, top_k=2)

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
        else:
            print("Using Custom Trainer for non-HuggingFace model")
            from soccernetpro.core.loss.builder import build_criterion
            from soccernetpro.core.optimizer.builder import build_optimizer
            from soccernetpro.core.scheduler.builder import build_scheduler
            self.model = model.to(self.device)
            train_loader = DataLoader(train_dataset, batch_size=self.config.DATA.train_batch_size, 
                                      shuffle=True, num_workers=self.config.DATA.num_workers, pin_memory=True
                            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.DATA.valid_batch_size, 
                                    shuffle=False, num_workers=self.config.DATA.num_workers, pin_memory=True
                            )

            optimizer = self.optimizer if self.optimizer is not None else build_optimizer(self.model.parameters(), cfg=self.config.TRAIN.optimizer)
            scheduler = self.scheduler if self.scheduler is not None else build_scheduler(optimizer, cfg=self.config.TRAIN.scheduler)
            criterion = build_criterion(self.config.TRAIN.criterion)

            if self.config.TRAIN.use_weighted_loss:
                self.class_weights = train_dataset.get_class_weights(
                    num_classes=self.config.MODEL.num_classes,
                    sqrt=True
                ).to(self.device)
            else:
                self.class_weights = None

            self.hf_trainer = MVTrainerClassification(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=None,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    class_weights=self.class_weights,
                    class_names=train_dataset.label_map,
                    save_dir=self.config.TRAIN.save_dir,
                    model_name=self.config.MODEL.backbone.type,
                    max_epochs=self.config.TRAIN.epochs,
                    device=self.device,
                    top_k=2,

                    # W&B
                    wandb_project=self.config.TASK,
                    wandb_run_name=f"{self.config.MODEL.backbone.type}_{self.config.MODEL.neck.type}_{self.config.MODEL.neck.agr_type}_{self.config.MODEL.head.type}_cls",
                    wandb_config={
                        "backbone": self.config.MODEL.backbone.type,
                        "aggregation": self.config.MODEL.neck.agr_type,
                        "lr": self.config.TRAIN.optimizer.lr,
                        "batch_size": self.config.DATA.train_batch_size,
                        "num_classes": self.config.MODEL.num_classes
                    },
                    log_attention=True
                )
            self.hf_trainer.train(epoch_start=self.epoch, save_every=self.config.TRAIN.save_every)



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

            test_loader = DataLoader(test_dataset, batch_size=self.config.DATA.valid_batch_size, 
                                    shuffle=False, num_workers=self.config.DATA.num_workers, pin_memory=True
                            )

            optimizer = self.optimizer if self.optimizer is not None else build_optimizer(self.model.parameters(), cfg=self.config.TRAIN.optimizer)
            scheduler = self.scheduler if self.scheduler is not None else build_scheduler(optimizer, cfg=self.config.TRAIN.scheduler)
            criterion = build_criterion(self.config.TRAIN.criterion)

            self.hf_trainer = MVTrainerClassification(
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

                    # W&B
                    wandb_project=self.config.TASK,
                    wandb_run_name=f"{self.config.MODEL.backbone.type}_{self.config.MODEL.neck.type}_{self.config.MODEL.neck.agr_type}_{self.config.MODEL.head.type}_cls",
                    wandb_config={
                        "backbone": self.config.MODEL.backbone.type,
                        "aggregation": self.config.MODEL.neck.agr_type,
                        "lr": self.config.TRAIN.optimizer.lr,
                        "batch_size": self.config.DATA.train_batch_size,
                        "num_classes": self.config.MODEL.num_classes
                    },
                    log_attention=True
                )
            loss, metrics = self.hf_trainer.test()
            
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



