from soccernetpro.core.utils.config import expand 
import os
import logging
import time
 
class LocalizationAPI:
    def __init__(self, config=None, data_dir=None, save_dir=None):
        from soccernetpro.core.utils.config import load_config_omega
        from soccernetpro.core.utils.load_annotations import check_config
        #from ..core.trainer import Trainer

        if config is None:
            raise ValueError("config path is required")

        # Load config
        ### load data_dor first then do load config with omega to resolve $paths
        config_path = expand(config)
        self.config = load_config_omega(config_path)
        # User must control dataset folder
        self.config.DATA.data_dir = expand(data_dir or self.config.DATA.data_dir)

        check_config(self.config)
        # User controls model saving location (never use BASE_DIR)
        #self.save_dir = expand(save_dir or self.config.TRAIN.save_dir or "./checkpoints")
        #os.makedirs(self.save_dir, exist_ok=True)
        log_dir = expand(self.config.SYSTEM.log_dir or "./log_dir")
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "train.log")),
                logging.StreamHandler(),  # still prints to console
            ],
        )

        logger = logging.getLogger(__name__)
        print("CONFIG PATH  :", config_path)
        print("DATA DIR     :", self.config.DATA.data_dir)
        print("Classes :", self.config.DATA.classes)
        #print("MODEL SAVEDIR:", self.save_dir)

        #self.trainer = Trainer(self.config)


    def train(self, train_set=None, valid_set=None, pretrained=None):
        from soccernetpro.datasets.builder import build_dataset
        from soccernetpro.models.builder import build_model
        from soccernetpro.core.trainer.localization_trainer import build_trainer
        from soccernetpro.core.utils.default_args import get_default_args_trainer, get_default_args_train
        from soccernetpro.core.utils.config import select_device, resolve_config_omega
        import random
        import numpy as np
        import torch
        # # Load model
        # if pretrained:
        #     self.model, self.processor, _ = self.trainer.load(expand(pretrained))
        # else:
        #     self.model, self.processor = build_model(self.config, self.trainer.device)
        # Expand annotation paths (user or config)
        self.config.DATA.train.path = expand(train_set or self.config.DATA.train.path)
        self.config.DATA.valid.path = expand(valid_set or self.config.DATA.valid.path)

        self.config = resolve_config_omega(self.config)
        logging.info("Configuration:")
        logging.info(self.config)
        #print(self.config)

        def set_seed(seed):
            random.seed(seed)  # Python random module
            np.random.seed(seed)  # NumPy
            torch.manual_seed(seed)  # PyTorch
            torch.cuda.manual_seed(seed)  # PyTorch CUDA
            torch.cuda.manual_seed_all(seed)  # Multi-GPU training

            # Ensures deterministic behavior
            torch.backends.cudnn.deterministic = True  
            torch.backends.cudnn.benchmark = False  

            # Ensures deterministic behavior for CUDA operations
            torch.use_deterministic_algorithms(True, warn_only=True)

        set_seed(self.config.SYSTEM.seed)
        # Start Timing
        start = time.time()

        device = select_device(self.config.SYSTEM)
        self.model = build_model(self.config, device=device)
        print(self.model)


        # Datasets
        # Train
        data_obj_train = build_dataset(self.config, split="train")
        dataset_Train = data_obj_train.building_dataset(
            cfg=data_obj_train.cfg,
            gpu=self.config.SYSTEM.GPU,
            default_args=data_obj_train.default_args,
        )
        train_loader = data_obj_train.building_dataloader(dataset_Train, cfg=data_obj_train.cfg.dataloader, gpu=self.config.SYSTEM.GPU, dali=True)
        print(len(train_loader))
        # Valid
        data_obj_valid = build_dataset(self.config,split="valid")
        dataset_Valid = data_obj_valid.building_dataset(
            cfg=data_obj_valid.cfg,
            gpu= self.config.SYSTEM.GPU,
            default_args=data_obj_valid.default_args,
        )
        valid_loader = data_obj_valid.building_dataloader(dataset_Valid, cfg=data_obj_valid.cfg.dataloader, gpu=self.config.SYSTEM.GPU, dali=True)
        print(len(valid_loader))

        # Trainer
        trainer = build_trainer(
            cfg=self.config,
            model=self.model,
            default_args=get_default_args_trainer(self.config, len(train_loader)),
            resume_from = pretrained
        )
        # Start training`
        logging.info("Start training")

        trainer.train(
            **get_default_args_train(
                self.model,
                train_loader,
                valid_loader,
                self.config.DATA.classes,
                self.config.TRAIN.type,
            )
        )

        logging.info(f"Total Execution Time is {time.time()-start} seconds")
  

    def infer(self, test_set=None, pretrained=None):
        from soccernetpro.datasets.builder import build_dataset
        from soccernetpro.models.builder import build_model
        from soccernetpro.core.trainer.localization_trainer import build_inferer
        from soccernetpro.core.utils.config import select_device, resolve_config_omega
        from soccernetpro.core.utils.checkpoint import load_checkpoint, localization_remap
        import time

        self.config.DATA.test.path = expand(test_set or self.config.DATA.test.path)
        self.config = resolve_config_omega(self.config)
        logging.info("Configuration:")
        logging.info(self.config)
        # Start Timing
        start = time.time()

        device = select_device(self.config.SYSTEM)
        self.model = build_model(self.config, device=device)
        print("Model type:", type(self.model))
        print("Torch model type:", type(self.model._model))
        # Load model
        if pretrained:
            self.model._model, _, _, epoch = load_checkpoint(model=self.model._model,
                                        path=expand(pretrained),
                                        device=device,
                                        key_remap_fn=localization_remap)
        
        # Datasets
        # Test
        data_obj_test = build_dataset(self.config, split="test")
        dataset_Test = data_obj_test.building_dataset(
            cfg=data_obj_test.cfg,
            gpu=self.config.SYSTEM.GPU,
            default_args=data_obj_test.default_args,
        )
        test_loader = data_obj_test.building_dataloader(dataset_Test, cfg=data_obj_test.cfg.dataloader, gpu=self.config.SYSTEM.GPU, dali=True)
        print(len(test_loader))

        # Inference
        inferer = build_inferer(cfg=self.config.MODEL,
                                model=self.model)
        metrics = inferer.infer(cfg=self.config, data=test_loader)
        #print(f"Inference Metrics: {metrics}")
        logging.info(f"Total Execution Time is {time.time()-start} seconds")
        return metrics
