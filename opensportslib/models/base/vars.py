# opensportslib/models/base/vars.py

import __future__
import torch
from opensportslib.models.backbones.builder import build_backbone
from opensportslib.models.neck.builder import build_neck
from opensportslib.models.heads.builder import build_head
from opensportslib.core.config.accessors import (
    get_data_classes,
    get_data_num_classes,
)

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
        print(f"Data classes: {get_data_classes(config)}")
        head.num_classes = get_data_num_classes(config)
        head.feat_dim = self.backbone.feat_dim
        self.head = build_head(head)

    def forward(self, mvimages):
        features, attention = self.mvaggregate(mvimages)
        return self.head(features), attention
