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
        check_config(self.config)
        # User must control dataset folder
        self.config.DATA.data_dir = expand(data_dir or self.config.DATA.data_dir)

        # User controls model saving location (never use BASE_DIR)
        #self.save_dir = expand(save_dir or self.config.TRAIN.save_dir or "./checkpoints")
        #os.makedirs(self.save_dir, exist_ok=True)

        print("DATA DIR     :", self.config.DATA.data_dir)
        print("Classes :", self.config.DATA.classes)
        #print("MODEL SAVEDIR:", self.save_dir)

        #self.trainer = Trainer(self.config)


    def train(self, train_set=None, valid_set=None, pretrained=None):
        from soccernetpro.datasets.builder import build_dataset
        from soccernetpro.models.builder import build_model
        from soccernetpro.core.trainer.localization_trainer import build_trainer
        from soccernetpro.core.utils.default_args import get_default_args_trainer, get_default_args_train
        import random
        import numpy as np
        import torch
        # # Load model
        # if pretrained:
        #     self.model, self.processor, _ = self.trainer.load(expand(pretrained))
        # else:
        #     self.model, self.processor = build_model(self.config, self.trainer.device)
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_model(self.config, device=device)
        print(self.model)


        # Datasets
        # Train
        data_obj_train = build_dataset(self.config, split="train")
        dataset_Train = data_obj_train.building_dataset(
            cfg=data_obj_train.cfg,
            gpu= self.config.TRAIN.GPU,
            default_args=data_obj_train.default_args,
        )
        print(dataset_Train)
        train_loader = data_obj_train.building_dataloader(dataset_Train, cfg=data_obj_train.cfg.dataloader, gpu=self.config.TRAIN.GPU, dali=True)
        print(len(train_loader))
        # Valid
        data_obj_valid = build_dataset(self.config,split="valid")
        dataset_Valid = data_obj_valid.building_dataset(
            cfg=data_obj_valid.cfg,
            gpu= self.config.TRAIN.GPU,
            default_args=data_obj_valid.default_args,
        )
        valid_loader = data_obj_valid.building_dataloader(dataset_Valid, cfg=data_obj_valid.cfg.dataloader, gpu=self.config.TRAIN.GPU, dali=True)
        print(get_default_args_trainer(self.config, len(train_loader)))
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
    # return
        # valid_data = build_dataset(self.config, valid_set, self.processor, split="valid")
        # print(f"Valid Dataset length: {len(valid_data)}")
        # frames = valid_data[0]
        # print(f"Frames shape: {frames['pixel_values'].shape}")  # 
        # print(f"Label: {frames['labels']}")

        # # Train
        # self.trainer.train_model(self.model, train_data, valid_data)

        # # Save in user-controlled location
        # save_path = os.path.join(self.save_dir, "final_model")
        # self.trainer.save(self.model, save_path, self.processor)
        # print("Model saved at:", save_path)


    def infer(self, test_set=None, pretrained=None):
        from soccernetpro.datasets.builder import build_dataset
        from soccernetpro.models.builder import build_model

        # Load model
        if pretrained:
            self.model, self.processor, _ = self.trainer.load(expand(pretrained))

        test_set = expand(test_set or self.config.DATA.annotations.valid)
        test_data = build_dataset(self.config, test_set, self.processor, split="test")

        preds, metrics = self.trainer.infer(test_data)
        return metrics
