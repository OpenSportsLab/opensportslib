# soccernetpro/apis/classification.py

"""public API for classification tasks.

supports three dataset/task combinations:
    - MVFoul (video-based foul classification)
    - SoccerNet-GAR with video modality
    - SoccerNet-GAR with tracking modality

handles single-GPU and multi-GPU (DDP) training and inference,
delegating heavy lifting to Trainer_Classification.
"""

import os

from soccernetpro.core.utils.config import expand 


class ClassificationAPI:
    """top-level entry point for classification training and inference.

    loads a YAML config, resolves paths, and exposes train() / 
    infer() methods that transparently handle single-GPU and
    DDP execution.

    Args:
        config: path to the YAML configuration file.
        data_dir: override for DATA.data_dir in the config.
            if None, the value from the config is used.
        save_dir: override for the checkpoint output directory.
            falls back to TRAIN.save_dir, then "./checkpoints".
    """

    def __init__(self, config=None, data_dir=None, save_dir=None):
        from soccernetpro.core.utils.config import (
            load_config_omega
        )

        if config is None:
            raise ValueError("config path is required")

        config_path = expand(config)
        self.config = load_config_omega(config_path)

        # let the caller override the dataset root directory.
        self.config.DATA.data_dir = expand(
            data_dir or self.config.DATA.data_dir
        )

        # checkpoint output directory; never derived from BASE_DIR so the
        # user always has explicit control over where weights are written.
        self.save_dir = expand(
            save_dir or self.config.TRAIN.save_dir or "./checkpoints"
        )
        os.makedirs(self.save_dir, exist_ok=True)

        # DDP rank; used for logging and checkpointing.
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print("DATA DIR     :", self.config.DATA.data_dir)
            print("MODEL SAVEDIR:", self.save_dir)

        self.trainer=None

    # -----------------------------------------------------------------
    # internal DDP worker
    # -----------------------------------------------------------------

    def _worker_ddp(
        self, 
        rank, 
        world_size, 
        mode, 
        return_queue=None, 
        train_set=None, 
        valid_set=None, 
        test_set=None, 
        pretrained=None
    ):
        """execute a single training or inference job on one GPU.

        spawned once per GPU by train() / infer(). Each process gets 
        its own Trainer_Classification instance so that no mutable
        state is shared across ranks.

        Args:
            rank: GPU rank (0-indexed).
            world_size: total number of participating GPUs.
            mode: "train" or "infer".
            return_queue: multiprocessing.Queue used by rank-0 to
                return metrics to the calling process (inference only).
            train_set: path to the training set annotations file.
            valid_set: path to the validation set annotations file.
            test_set: path to the test set annotations file.
            pretrained: path to a saved checkpoint for warm-starting
                or inference.
        """
        import torch
        from soccernetpro.core.trainer.classification_trainer import Trainer_Classification
        from soccernetpro.core.utils.ddp import ddp_setup, ddp_cleanup
        from soccernetpro.core.utils.seed import set_reproducibility
        from soccernetpro.datasets.builder import build_dataset
        from soccernetpro.models.builder import build_model

        # reproducibility: 
        # we default to reproducible training, but allow the user to
        # disable this via SYSTEM.use_seed=False in the config.
        if getattr(self.config.SYSTEM, "use_seed", False):
            set_reproducibility(self.config.SYSTEM.seed)

        is_ddp = world_size > 1
        if is_ddp:
            torch.cuda.set_device(rank)
            ddp_setup(rank, world_size)
            device = torch.device(f"cuda:{rank}")
        else:
            from soccernetpro.core.utils.config import select_device
            device = select_device(self.config.SYSTEM)
        
        # each process creates a fresh trainer to avoid shared mutable state.
        trainer = Trainer_Classification(self.config)
        trainer.device = device

        # build or restore the model.
        if pretrained:
            model, processor, scheduler, epoch = trainer.load(pretrained)
        else:
            model, processor = build_model(self.config, device)

        trainer.model = model

        if mode == "train":
            train_data = build_dataset(
                self.config, train_set, processor, split="train"
            )
            valid_data = build_dataset(
                self.config, valid_set, processor, split="valid"
            )
            trainer.train(
                model, train_data, valid_data, 
                rank=rank, world_size=world_size
            )

        elif mode == "infer":
            test_data = build_dataset(
                self.config, test_set, processor, split="test"
            )

            metrics = trainer.infer(
                test_data, rank=rank, world_size=world_size
            )

            if rank == 0 and return_queue is not None:
                print(metrics)
                return_queue.put(metrics)

        if is_ddp:
            ddp_cleanup()

    # -----------------------------------------------------------------
    # public training interface
    # -----------------------------------------------------------------

    def train(
        self, 
        train_set=None, 
        valid_set=None, 
        test_set=None, 
        pretrained=None, 
        use_ddp=False
    ):
        """run a full training loop.

        Args:
            train_set: path to training annotationns. defaults to the
                value in the loaded config.
            valid_set: path to validation annotations.
            test_set: currently unused.
            pretrained: optional checkpoint path for warm-starting.
            use_ddp: if True and more than one GPU is visible,
                spawn one process per GPU via torch.multiprocessing.spawn.
        """
        import torch
        import torch.multiprocessing as mp
        from soccernetpro.core.utils.config import (
            resolve_config_omega
        )

        train_set = expand(train_set or self.config.DATA.annotations.train)
        valid_set = expand(valid_set or self.config.DATA.annotations.valid)

        self.config = resolve_config_omega(self.config)
        print("Config : ", self.config)

        world_size = torch.cuda.device_count() or self.config.SYSTEM.GPU
        use_ddp = use_ddp and world_size > 1

        if use_ddp:
            print(f"Launching DDP on {world_size} GPUs")
            mp.spawn(
                self._worker_ddp,
                args=(
                    world_size, "train", None, 
                    train_set, valid_set, None, pretrained
                ),
                nprocs=world_size,
            )
        else:
            print("Single GPU training")
            self._worker_ddp(
                rank=0,
                world_size=1,
                mode="train",
                return_queue=None,
                train_set=train_set,
                valid_set=valid_set,
                pretrained=pretrained,
            )

    def infer(
        self, 
        test_set=None, 
        pretrained=None, 
        predictions=None, 
        use_ddp=False
    ):
        """run inference or evaluate saved predictions.
        
        when "predictions" is None, the model runs a forward pass
        over the test set and returns live metrics. when "predictions"
        points to a saved prediction file, only the evaluation step runs
        (no GPU needed).

        Args:
            test_set: path to test annotations.
            pretrained: checkpoint path (required when running live inference).
            predictions: path to a previously saved prediction file.
                if provided, evaluation is run offline without a model.
            use_ddp: if True, distribute inference across all visible GPUs.

        Returns:
            a metrics dictionary produced by the trainer.
        """
        import torch
        import torch.multiprocessing as mp
        from soccernetpro.core.utils.config import (
            resolve_config_omega
        )

        test_set = expand(test_set or self.config.DATA.annotations.test)

        self.config = resolve_config_omega(self.config)
        print("Config : ", self.config)

        if not predictions:
            # live inference: run the model on test data.
            world_size = torch.cuda.device_count()
            use_ddp = use_ddp and world_size > 1

            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            
            if use_ddp:
                mp.spawn(
                    self._worker_ddp,
                    args=(
                        world_size, "infer", queue, 
                        None, None, test_set, pretrained
                    ),
                    nprocs=world_size,
                )
            else:
                self._worker_ddp(
                    rank=0,
                    world_size=1,
                    mode="infer",
                    return_queue=queue,
                    test_set=test_set,
                    pretrained=pretrained,
                )
            
            # rank-0 pushes metrics into the queue; retrieve them here.
            metrics = queue.get()
        else:
            # offline evaluation from a saved prediction file.
            from soccernetpro.datasets.builder import build_dataset
            from soccernetpro.core.trainer.classification_trainer import Trainer_Classification

            self.trainer = Trainer_Classification(self.config)
            test_data = build_dataset(
                self.config, test_set, None, split="test"
            )
            metrics = self.trainer.evaluate(
                pred_path=predictions, 
                gt_path=test_set, 
                class_names=test_data.label_map, 
                exclude_labels=test_data.exclude_labels
            )
            
        return metrics
