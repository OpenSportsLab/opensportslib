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
from opensportslib.metrics.localization_metric import *
from opensportslib.core.optimizer.builder import build_optimizer
from opensportslib.core.optimizer.builder import build_optimizer
from opensportslib.core.scheduler.builder import build_scheduler
from opensportslib.core.utils.config import store_json
from opensportslib.datasets.builder import build_dataset
import os
import torch
import wandb
import time
import json
import tqdm
import numpy as np
from opensportslib.core.utils.config import load_gz_json, load_json
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
        print(cfg.SYSTEM.work_dir)
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
    
    else:
        trainer = Trainer_pl(cfg, default_args["work_dir"])
            

    return trainer

class Trainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

class Trainer_pl(Trainer):
    """Trainer class used for models that rely on lightning modules.

    Args:
        cfg (dict): Dict config. It should contain the key 'max_epochs' and the key 'GPU'.
    """

    def __init__(self, cfg, work_dir):
        from opensportslib.core.utils.lightning import CustomProgressBar, MyCallback
        import pytorch_lightning as pl

        self.work_dir = work_dir
        call = MyCallback()
        self.trainer = pl.Trainer(
            max_epochs=cfg.TRAIN.max_epochs,
            devices=cfg.SYSTEM.GPU,
            callbacks=[call, CustomProgressBar(refresh_rate=1)],
            num_sanity_val_steps=0,
        )

    def train(self, **kwargs):
        self.trainer.fit(**kwargs)

        best_model = kwargs["model"].best_state

        logging.info("Done training")
        logging.info("Best epoch: {}".format(best_model.get("epoch")))
        torch.save(best_model, os.path.join(self.work_dir, "model.pth.tar"))

        logging.info("Model saved")
        logging.info(os.path.join(self.work_dir, "model.pth.tar"))


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

        self.best_checkpoint_path = None

        self.save_dir = work_dir #os.path.join(work_dir, run_name, run_id)
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
            self.best_checkpoint_path = best_path
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
            self._run_final_evaluation(classes, eval_splits=["valid"])

    def _run_final_evaluation(self, classes, eval_splits=["valid", "test"]):
        from opensportslib.core.utils.checkpoint import load_checkpoint, localization_remap
        """Run final evaluation using best model."""
        # Load best model for evaluation
        best_checkpoint_path = os.path.join(
            self.save_dir, f"best_checkpoint.pt"
        )
        self.model._model, _, _, epoch = load_checkpoint(model=self.model._model,
                                        path=best_checkpoint_path,
                                        key_remap_fn=localization_remap)
        logging.info(f"Loaded best model from epoch {self.best_epoch}")

        for split in eval_splits:
            if split == "valid":
                cfg_tmp = self.cfg_valid_data_frames
                split = "valid_data_frames"
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

    def infer(self, cfg, data, dataloader=None):
        """Infer actions from data.

        Args:
            data : The data from which we will infer.
            dataloader : The dataloader for the test data.

        Returns:
            Dict containing predictions
        """
        if self.infer_Spotting=="infer_JSON":
            return self.infer_JSON(cfg, self.model, data, dataloader)
        elif self.infer_Spotting=="infer_SN":    
            return self.infer_SN(cfg, self.model, data, dataloader)
        elif self.infer_Spotting=="infer_E2E":
            return self.infer_E2E(cfg, self.model, data, dataloader)


    def infer_common(self, cfg, model, data, dataloader=None):
        """Infer actions from data using a given model.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            model: The model that will be used to infer.
            data : The data from which we will infer.

        Returns:
            Dict containing predictions
        """
        # Run Inference on Dataset
        from opensportslib.core.utils.lightning import CustomProgressBar, MyCallback
        import pytorch_lightning as pl

        if cfg.work_dir is not None and dataloader is not None:
            
            evaluator = pl.Trainer(
                callbacks=[CustomProgressBar()],
                devices=[cfg.training.GPU],
                num_sanity_val_steps=0,
            )
            evaluator.predict(model, dataloader)
            return model.json_data


    def infer_JSON(self, cfg, model, data, dataloader=None):
        """Infer actions from data using a given model for NetVlad/CALF methods

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            model: The model that will be used to infer.
            data : The data from which we will infer.

        Returns:
            Dict containing predictions
        """
        return self.infer_common(cfg, model, data, dataloader)


    def infer_SN(self, cfg, model, data, dataloader=None):
        """Infer actions from data using a given model for the SNV2 data

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            model: The model that will be used to infer.
            data : The data from which we will infer.

        Returns:
            Dict containing predictions
        """
        return self.infer_common(cfg, model, data, dataloader)


    def infer_E2E(self, cfg, model, data, dataloader=None):
        """Infer actions from data using a given model for the e2espot method.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
            model: The model that will be used to infer.
            data : The data from which we will infer.

        Returns:
            Dict containing predictions
        """
        pred_file = None
        if cfg.SYSTEM.work_dir is not None:
            pred_file = os.path.join(cfg.SYSTEM.work_dir, cfg.DATA.test.results)
            mAP = infer_and_process_predictions_e2e(
                model,
                getattr(cfg, "dali", False),
                data,
                "infer",
                cfg.DATA.classes,
                pred_file,
                True,
                cfg.DATA.test.dataloader,
                return_pred=False,
            )
            wandb.log({
                "test/Avg_mAP": mAP,
            })
            pred_json_file = os.path.join(pred_file + ".json")
            pred_recall_file = os.path.join(pred_file + ".recall.json.gz")
            logging.info("Predictions saved")
            logging.info(pred_json_file)
            logging.info("High recall predictions saved")
            logging.info(pred_recall_file)
            #json_gz_file = cfg.DATA.test.results + ".recall.json.gz"
            return pred_recall_file


def build_evaluator(cfg, default_args=None):
    """Build a evaluator from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        evaluator: The constructed evaluator.
    """
    if cfg.MODEL.runner.type == "runner_JSON":
        evaluator = Evaluator(cfg=cfg, evaluate_Spotting="evaluate_pred_JSON")
    elif cfg.MODEL.runner.type == "runner_pooling":
        evaluator = Evaluator(cfg=cfg, evaluate_Spotting="evaluate_pred_SN")
    elif cfg.MODEL.runner.type == "runner_CALF":
        evaluator = Evaluator(cfg=cfg, evaluate_Spotting="evaluate_pred_SN")
    elif cfg.MODEL.runner.type == "runner_e2e":
        evaluator = Evaluator(cfg=cfg, evaluate_Spotting="evaluate_pred_E2E")

    return evaluator


class Evaluator:
    """Evaluator class that is used to make easier the process of evaluate since there is only
    one evaluate method that uses the evaluate_Spotting method.

    Args:
        cfg (dict): Config dict.
        evaluate_Spotting (method): The method that is used to evaluate.
    """

    def __init__(self, cfg, evaluate_Spotting):
        self.cfg = cfg
        self.extract_fps = getattr(cfg.DATA, "extract_fps", 2)
        self.evaluate_Spotting = evaluate_Spotting

    def evaluate(self, cfg_testset, json_gz_file=None):
        """Evaluate predictions.

        Args:
            cfg_testset (dict): Config dict that contains informations for the predictions.
        """
        if self.evaluate_Spotting == "evaluate_pred_JSON":
            return self.evaluate_pred_JSON(cfg_testset, self.cfg.SYSTEM.work_dir, json_gz_file, metric=cfg_testset.metric)
        elif self.evaluate_Spotting == "evaluate_pred_SN":
            return self.evaluate_pred_SN(cfg_testset, self.cfg.SYSTEM.work_dir, json_gz_file, metric=cfg_testset.metric)
        elif self.evaluate_Spotting == "evaluate_pred_E2E":
            return self.evaluate_pred_E2E(cfg_testset, self.cfg.SYSTEM.work_dir, json_gz_file, metric=cfg_testset.metric)


    # def evaluate_common_JSON(self, cfg, results, metric):
    #     if cfg.path == None:
    #         return
    #     with open(cfg.path) as f:
    #         GT_data = json.load(f)

    #     print(results)
    #     pred_path_is_json = False
    #     if results.endswith(".json"):
    #         pred_path_is_json = True
    #         with open(results) as f:
    #             pred_data = json.load(f)

    #     targets_numpy = list()
    #     detections_numpy = list()
    #     closests_numpy = list()

    #     if "labels" in GT_data.keys():
    #         classes = GT_data["labels"]
    #     else:
    #         assert isinstance(cfg.classes, list) or os.path.isfile(cfg.classes)
    #         if isinstance(cfg.classes, list):
    #             classes = cfg.classes

    #     classes = sorted(classes)
    #     EVENT_DICTIONARY = {x: i for i, x in enumerate(classes)}
    #     INVERSE_EVENT_DICTIONARY = {i: x for i, x in enumerate(classes)}

    #     if "videos" in GT_data.keys():
    #         videos = GT_data["videos"]
    #     else:
    #         videos = [GT_data]

    #     for game in tqdm.tqdm(videos):
    #         print(game.keys())
    #         # fetch labels
    #         labels = game["annotations"]
    #         if not pred_path_is_json:
    #             try:
    #                 pred_file = os.path.join(results, os.path.splitext(game["path"])[0], "results_spotting.json")
    #                 print(pred_file)
    #                 with open(pred_file) as f:
    #                     pred_data = json.load(f)
    #             except FileNotFoundError:
    #                 continue
    #         predictions = pred_data["predictions"]
    #         # convert labels to dense vector
    #         dense_labels = label2vector(
    #             labels,
    #             num_classes=len(classes),
    #             EVENT_DICTIONARY=EVENT_DICTIONARY,
    #             framerate=(
    #                 pred_data["fps"] if "fps" in pred_data.keys() else self.extract_fps
    #             ),
    #         )
    #         print(dense_labels.shape)
    #         # convert predictions to vector
    #         dense_predictions = predictions2vector(
    #             predictions,
    #             vector_size=game["num_frames"] if "num_frames" in game.keys() else None,
    #             framerate=(
    #                 pred_data["fps"] if "fps" in pred_data.keys() else self.extract_fps
    #             ),
    #             num_classes=len(classes),
    #             EVENT_DICTIONARY=EVENT_DICTIONARY,
    #         )
    #         print(dense_predictions.shape)

    #         targets_numpy.append(dense_labels)
    #         detections_numpy.append(dense_predictions)

    #         closest_numpy = np.zeros(dense_labels.shape) - 1
    #         # Get the closest action index
    #         closests_numpy.append(get_closest_action_index(dense_labels, closest_numpy))

    #     if targets_numpy:
    #         return compute_performances_mAP(
    #             metric,
    #             targets_numpy,
    #             detections_numpy,
    #             closests_numpy,
    #             INVERSE_EVENT_DICTIONARY,
    #         )
    #     else:
    #         logging.warning("No predictions found for evaluation. Returning None.")
    #         return None

    
      
    def evaluate_common_JSON(self, cfg, results, metric):

        if cfg.path is None:
            return

        # --------------------------------------------------
        # LOAD GT
        # --------------------------------------------------
        with open(cfg.path) as f:
            GT_data = json.load(f)

        # --------------------------------------------------
        # LOAD PRED FILE (json / json.gz / folder)
        # --------------------------------------------------
        pred_data = None
        pred_path_is_file = results.endswith(".json") or results.endswith(".json.gz")

        if pred_path_is_file:
            pred_data = load_gz_json(results) if results.endswith(".gz") else load_json(results)

        # detect v2 prediction
        pred_is_v2 = isinstance(pred_data, dict) and pred_data is not None and "data" in pred_data
        # --------------------------------------------------
        # CLASSES
        # --------------------------------------------------
        if isinstance(GT_data.get("labels"), dict):
            classes = list(GT_data["labels"].values())[0]["labels"]
        elif "labels" in GT_data:
            classes = GT_data["labels"]
        else:
            classes = cfg.classes

        classes = sorted(classes)
        EVENT_DICTIONARY = {x: i for i, x in enumerate(classes)}
        INVERSE_EVENT_DICTIONARY = {i: x for i, x in enumerate(classes)}

        # --------------------------------------------------
        # GT VIDEOS
        # --------------------------------------------------
        if "videos" in GT_data:
            videos = GT_data["videos"]
            gt_is_v2 = False
        else:
            videos = GT_data["data"]
            gt_is_v2 = True

        # --------------------------------------------------
        # BUILD PRED LOOKUP IF V2
        # --------------------------------------------------
        pred_lookup = {}
        if pred_is_v2:
            for item in pred_data["data"]:
                video_path = item["inputs"][0]["path"]
                pred_lookup[video_path] = item

        targets_numpy = []
        detections_numpy = []
        closests_numpy = []

        # ==================================================
        # LOOP
        # ==================================================
        for game in tqdm.tqdm(videos):

            # ---------------- GT ----------------
            if gt_is_v2:
                video_path = game["inputs"][0]["path"]
                labels = [{"label": e.get("label"),  
                           "gameTime": e.get("gameTime"),
                           "position": int(e.get("position_ms")),
                          } for e in game.get("events", [])]
            else:
                video_path = game["path"]
                labels = game["annotations"]

            # ---------------- PRED ----------------
            if pred_path_is_file:

                # ===== V2 PRED =====
                if pred_is_v2:
                    if video_path not in pred_lookup:
                        continue

                    item = pred_lookup[video_path]
                    fps = item["inputs"][0].get("fps", self.extract_fps)

                    predictions = [
                        {
                           "label": e.get("label"),  
                           "gameTime": e.get("gameTime"),
                           "confidence": e.get("confidence"),
                           "position": int(e.get("position_ms")),
                           "frame": e.get("frame")
                        }
                        for e in item.get("events", [])
                    ]

                # ===== OLD PRED =====
                else:
                    if "predictions" not in pred_data:
                        continue

                    predictions = pred_data["predictions"]
                    fps = pred_data.get("fps", self.extract_fps)

            else:
                # ===== FOLDER MODE =====
                pred_file = os.path.join(results, os.path.splitext(video_path)[0], "results_spotting.json")
                
                if not os.path.exists(pred_file):
                    continue
                
                with open(pred_file) as f:
                    pred_data_local = json.load(f)

                if "data" in pred_data_local:
                    # v2 file inside folder
                    item = pred_data_local["data"][0]
                    fps = item["inputs"][0].get("fps", self.extract_fps)

                    predictions = [
                        {
                           "label": e.get("label"),  
                           "gameTime": e.get("gameTime"),
                           "confidence": e.get("confidence"),
                           "position": int(e.get("position_ms")),
                           "frame": e.get("frame")
                        }
                        for e in item.get("events", [])
                    ]
                else:
                    predictions = pred_data_local["predictions"]
                    fps = pred_data_local.get("fps", self.extract_fps)

            # ---------------- VECTORS ----------------
            dense_labels = label2vector(labels, num_classes=len(classes), EVENT_DICTIONARY=EVENT_DICTIONARY, framerate=fps)

            dense_predictions = predictions2vector(
                predictions,
                vector_size=None,
                framerate=fps,
                num_classes=len(classes),
                EVENT_DICTIONARY=EVENT_DICTIONARY,
            )

            targets_numpy.append(dense_labels)
            detections_numpy.append(dense_predictions)

            closest_numpy = np.zeros(dense_labels.shape) - 1
            closests_numpy.append(get_closest_action_index(dense_labels, closest_numpy))

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------
        if targets_numpy:
            return compute_performances_mAP(
                metric,
                targets_numpy,
                detections_numpy,
                closests_numpy,
                INVERSE_EVENT_DICTIONARY,
            )
        else:
            logging.warning("No predictions found.")
            return None

    def evaluate_pred_E2E(self, cfg, work_dir, pred_path, metric="loose"):
        """Evaluate predictions infered with E2E method and display performances.
        Args:
            cfg (dict): It should containt the keys; classes (list of classes), path (path of the groundtruth data).
            It should contain the key nms_window if evaluation of raw predictions. It should containt the key extract_fps if predictions file do not contain the fps at which the frames have been processed to infer.
            work_dir: The folder path under which the prediction files are stored.
            pred_path: The path for predictions files. It can be:
                - folder path (that contains predictions files)
                - file path (if raw prediction that needs to be processed first)
            metric (string): metric used to evaluate.
                In ["loose","tight","at1","at2","at3","at4","at5"].
                Default: "loose".

        Returns
            The different mAPs computed.
        """

        results = pred_path

        if os.path.isfile(results) and (
            results.endswith(".gz") or results.endswith(".json")
        ):
            pred = (load_gz_json if results.endswith(".gz") else load_json)(results)
            # --------------------------------------------------
            # SUPPORT NEW V2 FORMAT (dict)
            # --------------------------------------------------
            if isinstance(pred, dict) and "data" in pred:
                internal = []

                for item in pred["data"]:
                    video = item["inputs"][0]["path"]
                    fps = item["inputs"][0].get("fps", self.extract_fps)

                    events = []
                    for ev in item.get("events", []):
                        events.append({
                            "frame": ev.get("frame"),
                            "label": ev.get("label"),
                            "confidence": ev.get("confidence"),
                            "position": int(ev.get("position_ms")),
                            "gameTime": ev.get("gameTime"),
                        })

                    internal.append({
                        "video": video,
                        "fps": fps,
                        "events": events,
                    })

                pred = internal
            nms_window = cfg.nms_window
            if isinstance(pred, list):
                if nms_window > 0:
                    logging.info("Applying NMS: " + str(nms_window))
                    pred = non_maximum_supression(pred, nms_window)

                eval_dir = os.path.join(work_dir, pred_path.split(".gz")[0].split(".json")[0])
                only_one_file = store_eval_files_json(pred, eval_dir)
                logging.info("Done processing prediction files!")
                if only_one_file:
                    results = os.path.join(eval_dir, "results_spotting.json")
                else:
                    results = eval_dir
        return self.evaluate_common_JSON(cfg, results, metric)


    def evaluate_pred_JSON(self, cfg, work_dir, pred_path, metric="loose"):
        """Evaluate predictions infered with Json files and display performances.
        Args:
            cfg (dict): It should containt the key path (path of the groundtruth data). It should containt the key classes (list of classes) if the different classes are not in the ground truth data.
            work_dir: The folder path under which the prediction files are stored.
            pred_path: The path for predictions files. It can be:
                - folder path (that contains predictions files)
                - json file path if evaluate only one json file.
            metric (string): metric used to evaluate.
                In ["loose","tight","at1","at2","at3","at4","at5"].
                Default: "loose".

        Returns
            The different mAPs computed.
        """
        return self.evaluate_common_JSON(cfg, os.path.join(work_dir, pred_path), metric)


    def evaluate_pred_SN(self, cfg, work_dir, pred_path, metric="loose"):
        """Evaluate predictions infered using SNV2 splits and display performances. This method should be used only for SNV2 dataset.
        Args:
            cfg (dict): It should containt the key path (path of the groundtruth data). It should containt the key classes (list of classes) if the different classes are not in the ground truth data.
            work_dir: The folder path under which the prediction files are stored.
            pred_path: The path for predictions files.
            metric (string): metric used to evaluate.
                In ["loose","tight","at1","at2","at3","at4","at5"].
                Default: "loose".

        Returns
            The different mAPs computed.
        """
        from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2
        from SoccerNet.Evaluation.ActionSpotting import evaluate
        # challenge sets to be tested on EvalAI
        if "challenge" in cfg.split:
            print("Visit eval.ai to evaluate performances on Challenge set")
            return None
        # GT_path = cfg.data_root
        pred_path = os.path.join(work_dir, pred_path)
        results = evaluate(
            SoccerNet_path=cfg.data_root,
            Predictions_path=pred_path,
            split=cfg.split,
            prediction_file="results_spotting.json",
            version=getattr(cfg, "version", 2),
            metric=metric,
        )
        rows = []
        for i in range(len(results["a_mAP_per_class"])):
            label = INVERSE_EVENT_DICTIONARY_V2[i]
            rows.append(
                (
                    label,
                    "{:0.2f}".format(results["a_mAP_per_class"][i] * 100),
                    "{:0.2f}".format(results["a_mAP_per_class_visible"][i] * 100),
                    "{:0.2f}".format(results["a_mAP_per_class_unshown"][i] * 100),
                )
            )
        rows.append(
            (
                "Average mAP",
                "{:0.2f}".format(results["a_mAP"] * 100),
                "{:0.2f}".format(results["a_mAP_visible"] * 100),
                "{:0.2f}".format(results["a_mAP_unshown"] * 100),
            )
        )

        logging.info("Best Performance at end of training ")
        logging.info("Metric: " + metric)
        print(tabulate(rows, headers=["", "Any", "Visible", "Unseen"]))
        # logging.info("a_mAP visibility all: " +  str(results["a_mAP"]))
        # logging.info("a_mAP visibility all per class: " +  str( results["a_mAP_per_class"]))

        return results


