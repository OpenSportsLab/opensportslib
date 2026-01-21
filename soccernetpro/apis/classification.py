from soccernetpro.core.utils.config import expand 
import os
 
class ClassificationAPI:
    def __init__(self, config=None, data_dir=None, save_dir=None):
        from soccernetpro.core.utils.config import load_config
        from soccernetpro.core.trainer.classification_trainer import Trainer_Classification

        if config is None:
            raise ValueError("config path is required")

        # Load config
        config_path = expand(config)
        self.config = load_config(config_path)

        # User must control dataset folder
        self.config.DATA.data_dir = expand(data_dir or self.config.DATA.data_dir)

        # User controls model saving location (never use BASE_DIR)
        self.save_dir = expand(save_dir or self.config.TRAIN.save_dir or "./checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        print("DATA DIR     :", self.config.DATA.data_dir)
        print("MODEL SAVEDIR:", self.save_dir)

        self.trainer = Trainer_Classification(self.config)


    def train(self, train_set=None, valid_set=None, test_set=None, pretrained=None):
        from soccernetpro.datasets.builder import build_dataset
        from soccernetpro.models.builder import build_model

        # Load model
        if pretrained:
            self.model, self.processor, self.scheduler, epoch = self.trainer.load(expand(pretrained))
        else:
            self.model, self.processor = build_model(self.config, self.trainer.device)


        # Expand annotation paths (user or config)
        train_set = expand(train_set or self.config.DATA.annotations.train)
        valid_set = expand(valid_set or self.config.DATA.annotations.valid)

        # Datasets
        train_data = build_dataset(self.config, train_set, self.processor, split="train")
        print(f"Train Dataset length: {len(train_data)}")
        frames= train_data[0]
        print(f"Frames shape: {frames['pixel_values'].shape}")  # 
        print(f"Label: {frames['labels']}")

        valid_data = build_dataset(self.config, valid_set, self.processor, split="valid")
        print(f"Valid Dataset length: {len(valid_data)}")
        frames = valid_data[0]
        print(f"Frames shape: {frames['pixel_values'].shape}")  # 
        print(f"Label: {frames['labels']}")

        
        # Train
        self.trainer.train(self.model, train_data, valid_data)

        # Save in user-controlled location
        #save_path = os.path.join(self.save_dir, "final_model")
        #self.trainer.save(self.model, save_path, self.processor)
        #print("Model saved at:", save_path)


    def infer(self, test_set=None, pretrained=None):
        from soccernetpro.datasets.builder import build_dataset
        from soccernetpro.models.builder import build_model

        # Load model
        if pretrained:
            self.model, self.processor, _ = self.trainer.load(expand(pretrained))

        test_set = expand(test_set or self.config.DATA.annotations.test)
        test_data = build_dataset(self.config, test_set, self.processor, split="test")

        metrics = self.trainer.infer(test_data)
        print(metrics)
        return metrics
