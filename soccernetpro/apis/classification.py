# soccernetpro/apis/classification.py

from soccernetpro.core.utils.config import expand 
import os
 
class ClassificationAPI:
    def __init__(self, config=None, data_dir=None, save_dir=None):
        from soccernetpro.core.utils.config import load_config_omega

        if config is None:
            raise ValueError("config path is required")

        # Load config
        config_path = expand(config)
        self.config = load_config_omega(config_path)

        # User must control dataset folder
        self.config.DATA.data_dir = expand(data_dir or self.config.DATA.data_dir)

        # User controls model saving location (never use BASE_DIR)
        self.save_dir = expand(save_dir or self.config.TRAIN.save_dir or "./checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        rank = int(os.environ.get("RANK", 0))

        if rank == 0:
            print("DATA DIR     :", self.config.DATA.data_dir)
            print("MODEL SAVEDIR:", self.save_dir)

        #self.trainer = Trainer_Classification(self.config)
        self.trainer=None

    def _worker_ddp(self, rank, world_size, mode, return_queue=None, train_set=None, valid_set=None, test_set=None, pretrained=None):
        import torch
        from soccernetpro.core.utils.ddp import ddp_setup, ddp_cleanup
        from soccernetpro.datasets.builder import build_dataset
        from soccernetpro.models.builder import build_model
        from soccernetpro.core.utils.seed import set_reproducibility

        seed = getattr(self.config.SYSTEM, 'seed', 42)
        set_reproducibility(seed)
        is_ddp = world_size > 1
        if is_ddp:
            torch.cuda.set_device(rank)
            ddp_setup(rank, world_size)
            device = torch.device(f"cuda:{rank}")
        else:
            device = self.trainer.device
        
        # fresh trainer per process
        from soccernetpro.core.trainer.classification_trainer import Trainer_Classification

        trainer = Trainer_Classification(self.config)
        #trainer = type(self.trainer)(self.config)
        trainer.device = device

        # model
        if pretrained:
            model, processor, scheduler, epoch = trainer.load(pretrained)
        else:
            model, processor = build_model(self.config, device)

        # =====================================================
        # TRAIN MODE
        # =====================================================
        if mode == "train":
            train_data = build_dataset(self.config, train_set, processor, split="train")
            valid_data = build_dataset(self.config, valid_set, processor, split="valid")

            trainer.train(model, train_data, valid_data, rank=rank, world_size=world_size)

        # =====================================================
        # INFER MODE
        # =====================================================
        elif mode == "infer":
            test_data = build_dataset(self.config, test_set, processor, split="test")

            metrics = trainer.infer(test_data, rank=rank, world_size=world_size)

            if rank == 0 and return_queue is not None:
                print(metrics)
                return_queue.put(metrics)

        if is_ddp:
            ddp_cleanup()

    def train(self, train_set=None, valid_set=None, test_set=None, pretrained=None, use_ddp=False):
        import torch
        import torch.multiprocessing as mp
        from soccernetpro.core.utils.config import resolve_config_omega

        # Expand annotation paths (user or config)
        train_set = expand(train_set or self.config.DATA.annotations.train)
        valid_set = expand(valid_set or self.config.DATA.annotations.valid)

        self.config = resolve_config_omega(self.config)
        print("CONFIG:", self.config)

        world_size = torch.cuda.device_count() or self.config.SYSTEM.GPU
        use_ddp = use_ddp and world_size > 1
        if use_ddp:
            print(f"Launching DDP on {world_size} GPUs")
            mp.spawn(
                self._worker_ddp,
                args=(world_size, "train", None, train_set, valid_set, None, pretrained),
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

        # # Load model
        # if pretrained:
        #     self.model, self.processor, self.scheduler, epoch = self.trainer.load(expand(pretrained))
        # else:
        #     self.model, self.processor = build_model(self.config, self.trainer.device)


        # # Datasets
        # train_data = build_dataset(self.config, train_set, self.processor, split="train")
        # print(f"Train Dataset length: {len(train_data)}")
        # frames= train_data[0]
        # print(f"Frames shape: {frames['pixel_values'].shape}")  # 
        # print(f"Label: {frames['labels']}")

        # valid_data = build_dataset(self.config, valid_set, self.processor, split="valid")
        # print(f"Valid Dataset length: {len(valid_data)}")
        # frames = valid_data[0]
        # print(f"Frames shape: {frames['pixel_values'].shape}")  # 
        # print(f"Label: {frames['labels']}")

        
        # # Train
        # self.trainer.train(self.model, train_data, valid_data)

        # # Save in user-controlled location
        # #save_path = os.path.join(self.save_dir, "final_model")
        # #self.trainer.save(self.model, save_path, self.processor)
        # #print("Model saved at:", save_path)


    def infer(self, test_set=None, pretrained=None, predictions=None, use_ddp=False):
        import torch
        import torch.multiprocessing as mp
        from soccernetpro.core.utils.config import resolve_config_omega

        test_set = expand(test_set or self.config.DATA.annotations.test)
        if not predictions:
            self.config = resolve_config_omega(self.config)
            print("CONFIG:", self.config)

            world_size = torch.cuda.device_count()
            use_ddp = use_ddp and world_size > 1
            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            if use_ddp:
                mp.spawn(
                    self._worker_ddp,
                    args=(world_size, "infer", queue, None, None, test_set, pretrained),
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
            
            # get result from rank0
            metrics = queue.get()
        else:
            from soccernetpro.datasets.builder import build_dataset
            test_data = build_dataset(self.config, test_set, None, split="test")
            metrics = self.trainer.evaluate(pred_path=predictions, gt_path=test_set, class_names=test_data.label_map, exclude_labels=test_data.exclude_labels)
        return metrics
