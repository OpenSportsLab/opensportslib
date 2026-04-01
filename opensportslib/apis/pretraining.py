# opensportslib/apis/pretraining.py

"""Public API for self-supervised pre-training tasks.

Supports three SSL methods:
    - MAE (Masked Autoencoder / VideoMAE-style)
    - DINO (self-distillation with no labels)
    - SimCLR (contrastive learning)

Handles single-GPU and multi-GPU (DDP) training,
delegating heavy lifting to Trainer_Pretraining.
"""

import os
import logging
from opensportslib.core.utils.config import expand


class PretrainingAPI:
    """Top-level entry point for SSL pre-training.

    Loads a YAML config, resolves paths, and exposes a train()
    method that transparently handles single-GPU and DDP execution.

    Args:
        config: path to the YAML configuration file.
        data_dir: override for DATA.data_dir in the config.
        save_dir: override for the checkpoint output directory.
    """

    def __init__(self, config=None, data_dir=None, save_dir=None):
        from opensportslib.core.utils.config import load_config_omega
        import uuid

        if config is None:
            raise ValueError("config path is required")

        config_path = expand(config)
        self.config = load_config_omega(config_path)

        # let the caller override the dataset root directory.
        self.config.DATA.data_dir = expand(
            data_dir or self.config.DATA.data_dir
        )

        # checkpoint output directory.
        self.run_id = os.environ.get("RUN_ID") or str(uuid.uuid4())[:8]
        os.environ["RUN_ID"] = self.run_id

        self.save_dir = expand(
            save_dir or self.config.SYSTEM.save_dir or "./checkpoints"
        )
        save_filename = os.path.join(
            f"ssl_{self.config.SSL.method}", self.run_id
        )
        self.config.SYSTEM.save_dir = os.path.join(self.save_dir, save_filename)
        os.makedirs(self.config.SYSTEM.save_dir, exist_ok=True)

        rank = int(os.environ.get("RANK", 0))
        self.best_checkpoint = None

        log_dir = expand(self.config.SYSTEM.log_dir or "./log_dir")
        os.makedirs(os.path.join(self.config.SYSTEM.save_dir, log_dir), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "train.log")),
                logging.StreamHandler(),
            ],
            force=True,
        )
        if rank == 0:
            logging.info(f"DATA DIR     : {self.config.DATA.data_dir}")
            logging.info(f"MODEL SAVEDIR: {self.config.SYSTEM.save_dir}")
            logging.info(f"SSL METHOD   : {self.config.SSL.method}")

    # -----------------------------------------------------------------
    # internal DDP worker
    # -----------------------------------------------------------------
    @staticmethod
    def _worker_ddp(
        rank,
        world_size,
        config,
        return_queue=None,
        train_dir=None,
        valid_dir=None,
        pretrained=None,
        use_wandb=False,
    ):
        """Execute a single pre-training job on one GPU.

        Spawned once per GPU by train(). Each process gets its own
        Trainer_Pretraining instance to avoid shared mutable state.

        Args:
            rank: GPU rank (0-indexed).
            world_size: total number of participating GPUs.
            config: resolved configuration namespace.
            return_queue: multiprocessing.Queue for returning results.
            train_dir: path to training data directory.
            valid_dir: path to validation data directory (optional).
            pretrained: path to a checkpoint for warm-starting.
            use_wandb: whether to enable W&B logging.
        """
        import torch
        from opensportslib.core.trainer.pretraining_trainer import Trainer_Pretraining
        from opensportslib.core.utils.ddp import ddp_setup, ddp_cleanup
        from opensportslib.core.utils.wandb import init_wandb
        from opensportslib.core.utils.seed import set_reproducibility
        from opensportslib.datasets.pretraining_dataset import build as build_pretrain_dataset
        from opensportslib.models.builder import build_model

        # configure logging per process
        logging.basicConfig(
            level=logging.INFO,
            format=f"[RANK {rank}] %(asctime)s | %(levelname)s | %(message)s",
            force=True,
        )
        if rank != 0:
            logging.getLogger().setLevel(logging.ERROR)

        if rank == 0:
            init_wandb(config, run_id=os.environ["RUN_ID"], use_wandb=use_wandb)

        if getattr(config.SYSTEM, "use_seed", False):
            set_reproducibility(config.SYSTEM.seed)

        is_ddp = world_size > 1
        if is_ddp:
            torch.cuda.set_device(rank)
            ddp_setup(rank, world_size)
            device = torch.device(f"cuda:{rank}")
        else:
            from opensportslib.core.utils.config import select_device
            device = select_device(config.SYSTEM)

        trainer = Trainer_Pretraining(config)
        trainer.device = device

        # build model
        if pretrained:
            model, _, _, epoch = trainer.load(pretrained)
        else:
            model = build_model(config, device)

        trainer.model = model

        # build datasets
        train_data = build_pretrain_dataset(config, data_dir=train_dir, split="train")
        valid_data = None
        if valid_dir:
            valid_data = build_pretrain_dataset(config, data_dir=valid_dir, split="valid")

        best_ckpt = trainer.train(
            model, train_data, valid_data,
            rank=rank, world_size=world_size,
        )

        if rank == 0 and return_queue is not None:
            best_ckpt = getattr(trainer.trainer, "best_checkpoint_path", None)
            return_queue.put(best_ckpt)

        if is_ddp:
            ddp_cleanup()

    # -----------------------------------------------------------------
    # public training interface
    # -----------------------------------------------------------------

    def train(
        self,
        train_dir=None,
        valid_dir=None,
        pretrained=None,
        use_ddp=False,
        use_wandb=True,
    ):
        """Run a full SSL pre-training loop.

        Args:
            train_dir: path to the training data directory.
                Defaults to DATA.data_dir from the config.
            valid_dir: path to validation data directory (optional).
            pretrained: optional checkpoint path for warm-starting.
            use_ddp: if True and more than one GPU is visible,
                spawn one process per GPU.
            use_wandb: whether to enable W&B logging.

        Returns:
            path to the best checkpoint.
        """
        import torch
        import torch.multiprocessing as mp
        from opensportslib.core.utils.config import resolve_config_omega

        train_dir = expand(train_dir or self.config.DATA.data_dir)
        valid_dir = expand(valid_dir) if valid_dir else None

        self.config = resolve_config_omega(self.config)
        logging.info("Configuration:")
        logging.info(self.config)

        world_size = torch.cuda.device_count() or self.config.SYSTEM.GPU
        use_ddp = use_ddp and world_size > 1

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        if use_ddp:
            logging.info(f"Launching DDP on {world_size} GPUs")
            mp.spawn(
                PretrainingAPI._worker_ddp,
                args=(
                    world_size, self.config, queue,
                    train_dir, valid_dir, pretrained, use_wandb,
                ),
                nprocs=world_size,
            )
        else:
            logging.info("Single GPU pre-training")
            PretrainingAPI._worker_ddp(
                rank=0,
                world_size=1,
                config=self.config,
                return_queue=queue,
                train_dir=train_dir,
                valid_dir=valid_dir,
                pretrained=pretrained,
                use_wandb=use_wandb,
            )

        self.best_checkpoint = queue.get()
        return self.best_checkpoint
