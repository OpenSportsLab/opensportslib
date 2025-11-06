import os
import yaml
import torch
import argparse
import logging
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from datasets.dataloader import ActionDataset
from models.model import ClassificationModel
from train import trainer
from utils.config_utils import load_config



def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "train.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
    )

def main():
    # ---------------------------
    # Argument parser
    # ---------------------------
    parser = argparse.ArgumentParser(description="Action Classification")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # ---------------------------
    # Load config
    # ---------------------------
    config = load_config(args.config)
    data_config = config.get("DATA")
    train_config = config.get("TRAIN")

    # Setup logging
    setup_logging(config.get("SYSTEM").get("log_dir"))

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.get("SYSTEM").get("seed"))

    # Dataset & DataLoader
    train_set = ActionDataset(data_config, split="train")
    val_set = ActionDataset(data_config, split="valid")
    test_set = ActionDataset(data_config, split="test")

    train_loader = DataLoader(
        train_set,
        batch_size=data_config.get("batch_size"),
        shuffle=True,
        num_workers=data_config.get("num_workers"),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=data_config.get("num_workers"),
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=data_config.get("num_workers"),
        pin_memory=False,
    )

    # Model
    model = ClassificationModel(config).to(device)
    model_name = config.get("MODEL").get("name")

    # Loss, Optimizer, Scheduler

    if train_config.get("loss_fn")=="cross_entropy":
        criterion = nn.CrossEntropyLoss()

    if train_config.get("optimizer")=="adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config["learning_rate"],
            betas=(0.9, 0.999), 
            eps=1e-7,
            weight_decay=train_config["weight_decay"],
            amsgrad=train_config["amsgrad"]
        )

    if train_config.get("lr_decay")=="StepLR":
        scheduler = StepLR(optimizer, step_size=train_config["step_size"], gamma=train_config["gamma"])

    # Call Trainer
    trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        best_model_path=train_config.get("save_dir"),
        epoch_start=0,
        model_name=model_name,
        max_epochs=train_config.get("epochs"),
        device=device,
        project_name=config.get("WANDB", {}).get("project", "classification"),
        run_name=config.get("WANDB", {}).get("run_name", model_name),
        config_dict=config,
    )


if __name__ == "__main__":
    main()
