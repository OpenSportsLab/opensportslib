"""
Copyright 2022 James Hong, Haotian Zhang, Matthew Fisher, Michael Gharbi,
Kayvon Fatahalian

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from soccernetpro.metrics.localization_metric import infer_and_process_predictions_e2e
from soccernetpro.core.optimizer.builder import build_optimizer
from soccernetpro.core.optimizer.builder import build_optimizer
from soccernetpro.core.scheduler.builder import build_scheduler
from soccernetpro.core.utils.config import store_json
from soccernetpro.datasets.builder import build_dataset
import os
import torch
import wandb
import time
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

def build_trainer(cfg, model=None, default_args=None, resume_from=None):
    """Build a trainer from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        model : The model that is used to train. Needed only if E2E method because training do not rely on pytorch lightning.
            Default: None.
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        evaluator: The constructed trainer.
    """
    if cfg.TRAIN.type == "trainer_e2e":
        print(cfg.MODEL.work_dir)
        checkpoint_dir = default_args["work_dir"]
        start_epoch = 0
        logging.info(f"Checkpoint directory: {checkpoint_dir}")
        
        # Handle checkpoint loading
        if resume_from is not None:
            if not os.path.isfile(resume_from):
                raise ValueError(f"Checkpoint file not found: {resume_from}")
                
            logging.info(f"Loading checkpoint from: {resume_from}")
            checkpoint = torch.load(resume_from)
            
            # Load model state
            model.load(checkpoint['model_state_dict'])
            logging.info("Model state loaded successfully")
            
            # Get current training progress
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resuming from epoch {start_epoch}")
            
            # Check if we've already reached target epochs
            if start_epoch >= cfg.TRAIN.num_epochs:
                logging.error(f"Model already trained for {start_epoch} epochs")
                logging.error(f"Target epochs in config: {cfg.TRAIN.num_epochs}")
                logging.error("Please increase num_epochs in config to continue training")
                raise ValueError("Need to increase num_epochs to continue training")
            
            logging.info(f"Will continue training from epoch {start_epoch} to {cfg.TRAIN.num_epochs}")
        
        logging.info("Building optimizer...")
        optimizer, scaler = build_optimizer(model._get_params(), cfg.TRAIN.optimizer)
        
        # Load optimizer state if available in checkpoint
        if resume_from is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logging.info("Optimizer and scaler states loaded")
            except Exception as e:
                logging.warning(f"Could not load optimizer state: {e}")
                logging.warning("Will start with fresh optimizer state")
            
        logging.info("Building scheduler...")
        lr_scheduler = build_scheduler(optimizer, cfg.TRAIN.scheduler, default_args)
        
        # Load scheduler state if available
        if resume_from is not None and 'lr_state_dict' in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_state_dict'])
                logging.info("Scheduler state loaded")
            except Exception as e:
                logging.warning(f"Could not load scheduler state: {e}")
                logging.warning("Will start with fresh scheduler state")

        trainer = Trainer_e2e(
            cfg,
            model,
            optimizer,
            scaler,
            lr_scheduler,
            default_args["work_dir"],
            default_args["dali"],
            default_args["repartitions"],
            default_args["cfg_test"],
            #default_args["cfg_challenge"],
            default_args["cfg_valid_data_frames"],
            start_epoch=start_epoch
        )
        
        # Load training history if resuming
        if resume_from is not None:
            trainer.best_epoch = checkpoint.get('best_epoch', 0)
            trainer.best_criterion_valid = checkpoint.get('best_criterion_valid', 
                0 if cfg.TRAIN.criterion_valid == "map" else float("inf"))
            logging.info(f"Restored best epoch: {trainer.best_epoch}")
            

    return trainer

class Trainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass



class Trainer_e2e(Trainer):
    """Trainer class used for the e2e model.

    Args:
        args (dict): Dict of config.
        model.
        optimizer (torch.optim.Optimizer): The optimizer to update model parameters. Set to None if validation epoch.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
        lr_scheduler : The learning rate scheduler.
        work_dir (string): The folder in which the different files will be saved.
        dali (bool): Whether videos are processed with dali or opencv.
        repartitions (List[int]): List of gpus used data processing.
            Default: None.
        cfg_test (dict): Dict of config for the inference (testing purpose) and evaluation of the test split. Occurs once training is done.
            Default: None.
        cfg_challenge (dict): Dict of config for the inference (testing purpose) of the challenge split. Occurs once training is done.
            Default: None.
        cfg_valid_data_frames (dict): Dict of config for the inference (testing purpose) and evaluation of the valid split. Occurs through the epochs after a certain number of epochs only if the criterion for the valid split is 'map'.
            Default: None.
    """

    def __init__(
        self,
        args,
        model,
        optimizer,
        scaler,
        lr_scheduler,
        work_dir,
        dali,
        repartitions=None,
        cfg_test=None,
        #cfg_challenge=None,
        cfg_valid_data_frames=None,
        start_epoch=0
    ):
        self.config = args
        self.losses = []
        self.best_epoch = 0
        self.best_criterion_valid = 0 if args.TRAIN.criterion_valid == "map" else float("inf")

        self.num_epochs = args.TRAIN.num_epochs
        self.epoch = start_epoch
        self.model = model

        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler

        self.acc_grad_iter = args.TRAIN.acc_grad_iter

        self.start_valid_epoch = args.TRAIN.start_valid_epoch
        self.criterion_valid = args.TRAIN.criterion_valid
        self.valid_map_every = args.TRAIN.valid_map_every
        #self.save_dir = work_dir
        self.dali = dali

        self.repartitions = repartitions
        self.cfg_test = cfg_test
        #self.cfg_challenge = cfg_challenge
        self.cfg_valid_data_frames = cfg_valid_data_frames

        run_name = f"{args.TASK}_{args.MODEL.type}_{args.MODEL.backbone.type}_{args.MODEL.head.type}"

        self.wandb_run = wandb.init(
            project=args.TASK,
            name=run_name,
            config={
                "backbone": args.MODEL.backbone.type,
                "head": args.MODEL.head.type,
                "lr": args.TRAIN.optimizer.lr,
                "batch_size": args.DATA.train.dataloader.batch_size,
                "num_classes": len(args.DATA.classes) if hasattr(args.DATA, "classes") else None,
                "epochs": args.TRAIN.num_epochs,
                "resume": start_epoch > 0,
            },
            reinit=True
        )

        run_id = wandb.run.id if wandb.run else time.strftime("%Y%m%d-%H%M%S")
        self.save_dir = os.path.join(work_dir, run_name, run_id)
        os.makedirs(self.save_dir, exist_ok=True)
        try:
            wandb.watch(self.model, log="gradients", log_freq=100)
        except Exception:
            pass

    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint with training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'lr_state_dict': self.lr_scheduler.state_dict(),
            'best_epoch': self.best_epoch,
            'best_criterion_valid': self.best_criterion_valid
        }

        os.makedirs(self.save_dir, exist_ok=True)
        # Save latest checkpoint
        # latest_path = os.path.join(self.save_dir, f"latest_checkpoint_{epoch:03d}.pt")
        # torch.save(checkpoint, latest_path)
        # logging.info(f"Latest checkpoint saved: {latest_path}")

        # # Remove previous latest checkpoint
        # for f in os.listdir(self.save_dir):
        #     if f.startswith("latest_checkpoint_") and f != os.path.basename(latest_path):
        #         os.remove(os.path.join(self.save_dir, f))

        latest_path = os.path.join(self.save_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, latest_path)
        logging.info(f"Latest checkpoint saved: {latest_path}")
        
        # Save best checkpoint if needed
        # if is_best:
        #     # best_path = os.path.join(self.save_dir, f"best_checkpoint_{epoch:03d}.pt")
        #     best_path = os.path.join(self.save_dir, f"best_checkpoint_{epoch:03d}.pt")
        #     torch.save(checkpoint, best_path)
        #     logging.info(f"Best checkpoint saved: {best_path}")
            
        #     # Remove previous best checkpoint
        #     for f in os.listdir(self.save_dir):
        #         if f.startswith("best_checkpoint_") and f != os.path.basename(best_path):
        #             os.remove(os.path.join(self.save_dir, f))
                    
        if is_best:
            best_path = os.path.join(self.save_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_path)
            logging.info(f"Best checkpoint saved: {best_path}")
            
    def train(self, train_loader, valid_loader, classes):
        """Training loop with checkpoint management."""
        if self.criterion_valid == "map":
            data_obj_valid = build_dataset(self.config, split="valid_data_frames")
            dataset_Valid_Frames =  data_obj_valid.building_dataset(
                data_obj_valid.cfg,
                None,
                {"repartitions": self.repartitions, "classes": classes},
            )

        for epoch in range(self.epoch, self.num_epochs):
            train_loss = self.model.epoch(
                train_loader,
                self.dali,
                self.optimizer,
                self.scaler,
                lr_scheduler=self.lr_scheduler,
                acc_grad_iter=self.acc_grad_iter,
            )

            valid_loss = self.model.epoch(
                valid_loader, self.dali, acc_grad_iter=self.acc_grad_iter
            )
            print(
                f"[Epoch {epoch+1}/{self.num_epochs}] Train loss: {train_loss:.5f} Valid loss: {valid_loss:.5f}"
            )
            logging.info(
                f"[Epoch {epoch+1}/{self.num_epochs}] Train loss: {train_loss:.5f} Valid loss: {valid_loss:.5f}"
            )

            valid_mAP = 0
            is_best = False

            if self.criterion_valid == "loss":
                if valid_loss < self.best_criterion_valid:
                    self.best_criterion_valid = valid_loss
                    self.best_epoch = epoch
                    is_best = True
                    print("New best epoch!")
            elif self.criterion_valid == "map":
                if epoch >= self.start_valid_epoch and epoch % self.valid_map_every == 0:
                    pred_file = None
                    if self.save_dir is not None:
                        pred_file = os.path.join(
                            self.save_dir, f"pred-valid_{epoch:03d}"
                        )
                        os.makedirs(self.save_dir, exist_ok=True)
                    valid_mAP = infer_and_process_predictions_e2e(
                        self.model,
                        self.dali,
                        dataset_Valid_Frames,
                        "VALID",
                        classes,
                        pred_file,
                        dataloader_params=self.cfg_valid_data_frames.dataloader,
                    )
                    if valid_mAP > self.best_criterion_valid:
                        self.best_criterion_valid = valid_mAP
                        self.best_epoch = epoch
                        is_best = True
                        print("New best epoch!")
            else:
                print("Unknown criterion:", self.criterion_valid)

            self.losses.append(
                {
                    "epoch": epoch,
                    "train": train_loss,
                    "valid": valid_loss,
                    "valid_mAP": valid_mAP,
                }
            )

            # ---------------- W&B LOG ----------------
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "valid/loss": valid_loss,
                "valid/mAP": valid_mAP,
                "lr": self.optimizer.param_groups[0]["lr"],
                "best/mAP": self.best_criterion_valid if self.criterion_valid == "map" else None,
                "best/loss": self.best_criterion_valid if self.criterion_valid == "loss" else None,
            })

            if self.save_dir is not None:
                os.makedirs(self.save_dir, exist_ok=True)
                store_json(
                    os.path.join(self.save_dir, "loss.json"),
                    self.losses,
                    pretty=True
                )
                self.save_checkpoint(epoch, is_best)

        logging.info(f"Training completed. Best epoch: {self.best_epoch}")

        if self.dali:
            train_loader.delete()
            valid_loader.delete()
            if self.criterion_valid == "map":
                dataset_Valid_Frames.delete()

        if self.save_dir is not None:
            self._run_final_evaluation(classes)

    def _run_final_evaluation(self, classes):
        """Run final evaluation using best model."""
        # Load best model for evaluation
        best_checkpoint_path = os.path.join(
            self.save_dir, f"best_checkpoint.pt"
        )
        checkpoint = torch.load(best_checkpoint_path)
        self.model.load(checkpoint['model_state_dict'])
        logging.info(f"Loaded best model from epoch {self.best_epoch}")

        # Evaluate on valid if not already done
        eval_splits = ["valid"] if self.criterion_valid != "map" else []

        # Evaluate on hold out splits
        eval_splits += ["test", "challenge"]
        for split in eval_splits:
            if split == "valid":
                cfg_tmp = self.cfg_valid_data_frames
            elif split == "test":
                cfg_tmp = self.cfg_test
            # elif split == "challenge":
            #     cfg_tmp = self.cfg_challenge

            split_path = os.path.join(cfg_tmp.path)
            if not os.path.exists(split_path):
                continue

            data_obj = build_dataset(self.config, split=split)
            split_data = data_obj.building_dataset(
                data_obj.cfg,
                None,
                {"repartitions": self.repartitions, "classes": classes},
            )
            split_data.print_info()

            pred_file = None
            if self.save_dir is not None:
                pred_file = os.path.join(
                    self.save_dir, f"pred-{split}_{self.best_epoch:03d}"
                )

            infer_and_process_predictions_e2e(
                self.model,
                self.dali,
                split_data,
                split.upper(),
                classes,
                pred_file,
                calc_stats=split != "challenge",
                dataloader_params=cfg_tmp.dataloader,
            )

            if self.dali:
                split_data.delete()

        logging.info(f"Final evaluation completed. Best epoch: {self.best_epoch}")


def build_inferer(cfg, model, default_args=None):
    """Build a inferer from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        model: The model that will be used to infer.
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        inferer: The constructed inferer.
    """

    if cfg.runner.type == "runner_JSON":
        inferer = Inferer(cfg=cfg, model=model, infer_Spotting="infer_JSON")
    elif cfg.runner.type == "runner_pooling":
        inferer = Inferer(cfg=cfg, model=model, infer_Spotting="infer_SN")
    elif cfg.runner.type == "runner_CALF":
        inferer = Inferer(cfg=cfg, model=model, infer_Spotting="infer_SN")
    elif cfg.runner.type == "runner_e2e":
        inferer = Inferer(cfg=cfg, model=model, infer_Spotting="infer_E2E")

    return inferer

class Inferer:
    def __init__(self, cfg, model, infer_Spotting):
        """Initialize the Inferer class.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            model: The model that will be used to infer.
            infer_Spotting: The method that is used to infer.
        """
        self.cfg_model = cfg
        self.model = model
        self.infer_Spotting=infer_Spotting

    def infer(self, cfg, data):
        """Infer actions from data.

        Args:
            data : The data from which we will infer.

        Returns:
            Dict containing predictions
        """
        if self.infer_Spotting=="infer_JSON":
            return self.infer_JSON(cfg, self.model, data)
        elif self.infer_Spotting=="infer_SN":    
            return self.infer_SN(cfg, self.model, data)
        elif self.infer_Spotting=="infer_E2E":
            return self.infer_E2E(cfg, self.model, data)


    def infer_common(self, cfg, model, data):
        """Infer actions from data using a given model.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            model: The model that will be used to infer.
            data : The data from which we will infer.

        Returns:
            Dict containing predictions
        """
        # Run Inference on Dataset
        pass


    def infer_JSON(self, cfg, model, data):
        """Infer actions from data using a given model for NetVlad/CALF methods

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            model: The model that will be used to infer.
            data : The data from which we will infer.

        Returns:
            Dict containing predictions
        """
        return self.infer_common(cfg, model, data)


    def infer_SN(self, cfg, model, data):
        """Infer actions from data using a given model for the SoccerNetV2 data

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            model: The model that will be used to infer.
            data : The data from which we will infer.

        Returns:
            Dict containing predictions
        """
        return self.infer_common(cfg, model, data)


    def infer_E2E(self, cfg, model, data):
        """Infer actions from data using a given model for the e2espot method.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            model: The model that will be used to infer.
            data : The data from which we will infer.

        Returns:
            Dict containing predictions
        """
        pred_file = None
        if self.cfg_model.work_dir is not None:
            pred_file = os.path.join(self.cfg_model.work_dir, cfg.DATA.test.results)
            mAP = infer_and_process_predictions_e2e(
                model,
                getattr(cfg, "dali", False),
                data,
                "infer",
                cfg.DATA.classes,
                pred_file,
                False,
                cfg.DATA.test.dataloader,
                return_pred=False,
            )
            logging.info("Predictions saved")
            logging.info(os.path.join(pred_file + ".json"))
            logging.info("High recall predictions saved")
            logging.info(os.path.join(pred_file + ".recall.json.gz"))
            return mAP