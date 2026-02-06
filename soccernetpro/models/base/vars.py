import __future__
import torch
from soccernetpro.models.backbones.builder import build_backbone
from soccernetpro.models.neck.builder import build_neck
from soccernetpro.models.heads.builder import build_head

class MVNetwork(torch.nn.Module):

    def __init__(self, config, backbone, neck, head):
        super().__init__()
        print("Building MVNetwork Model")
        self.lifting_net = torch.nn.Sequential()
        
        self.backbone = build_backbone(backbone)
        self.mvaggregate = build_neck(neck, default_args=dict(
                model=self.backbone,
                feat_dim=self.backbone.feat_dim,
                lifting_net=self.lifting_net
            )
        )
        head.num_classes = config.DATA.num_classes
        head.feat_dim = self.backbone.feat_dim
        self.head = build_head(head)

    def forward(self, mvimages):
        features, attention = self.mvaggregate(mvimages)
        return self.head(features), attention