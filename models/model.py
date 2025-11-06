import torch
from models.mvaggregate import MVAggregate
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


class ClassificationModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        model_cfg = config["MODEL"]

        self.name = model_cfg.get("name", "mvit_v2_s")
        self.num_classes = model_cfg.get("num_classes", 8)
        self.feat_dim = model_cfg.get("feature_dim", 400)
        self.agr_type =  model_cfg.get("agr_type", "max")
        self.lifting_net = torch.nn.Sequential()

        # Load pretrained MViT_V2_S
        weights_model = MViT_V2_S_Weights.DEFAULT
        network = mvit_v2_s(weights=weights_model)

        network.fc = torch.nn.Sequential()

        # Wrap with multi-view aggregation
        self.mvnetwork = MVAggregate(
            model=network,
            num_classes=self.num_classes,
            agr_type=self.agr_type,
            feat_dim=self.feat_dim,
            lifting_net=self.lifting_net,
        )

    def forward(self, mvimages):
        """
        mvimages: tensor of shape (B, V, C, T, H, W)
        returns: pred_action, attention
        """
        return self.mvnetwork(mvimages)



if __name__ == "__main__":

    from utils.config_utils import load_config
    # Load config
    config = load_config("configs/default.yaml")
    
    model = ClassificationModel(config)
    model.eval()
    
    # Simulate multi-view input: (batch, views, channels, frames, H, W)
    x = torch.randn(2, 2, 3, 16, 224, 224)  # 2 samples, 3 views, 8 frames
    #x = torch.randn(8, 3, 3, 16, 224, 224)  # single sample, single view
    #x = torch.rand(2, 2, 3, 16, 224, 398)
    
    with torch.no_grad():
        pred_action, attention = model(x)
    
    print("Action output shape:", pred_action.shape)  # [B, num_classes]
    print("Attention shape:", attention.shape)        # [B, V] or [B, V, V] depending on aggregation

