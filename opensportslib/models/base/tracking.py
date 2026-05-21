import torch
import torch.nn as nn

from opensportslib.models.backbones.builder import build_backbone
from opensportslib.models.neck.builder import build_neck
from opensportslib.models.heads.builder import build_head
from opensportslib.datasets.utils.tracking import FEATURE_DIM
from opensportslib.core.config.accessors import get_component_params_by_kind, get_data_sampling


class TrackingModel(nn.Module):
    """
    Tracking-based classification model.
    Combines graph backbone, temporal neck, and classification head.
    """
    
    def __init__(self, config, device):
        super().__init__()
        print("Building TrackingModel")
        
        self.device = device
        sampling = get_data_sampling(config)
        self.num_frames = sampling.get("num_frames")
        
        # backbone: graph encoder
        self.backbone = build_backbone(
            get_component_params_by_kind(config, "encoder"),
            default_args={"input_dim": FEATURE_DIM}
        )
        
        # neck: temporal aggregation
        self.neck = build_neck(
            get_component_params_by_kind(config, "adapter"),
            default_args={"window_size": self.num_frames}
        )
        
        # head: classifier
        self.head = build_head(
            get_component_params_by_kind(config, "head"),
            default_args={"input_dim": self.neck.feat_dim}
        )
    
    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                - x: (B*T*N, F) all node features batched
                - edge_index: (2, E) all edges with proper offsets
                - batch: (B*T*N,) graph assignment per node
                - batch_size: int
                - seq_len: int
        
        Returns:
            logits: (B, num_classes)
        """
        x = batch['x']
        edge_index = batch['edge_index']
        batch_idx = batch['batch']
        batch_size = batch['batch_size']
        seq_len = batch['seq_len']
        
        # single forward through backbone for all B*T graphs
        graph_emb = self.backbone(x, edge_index, batch_idx)  # (B*T, H)
        
        # reshape to (B, T, H)
        x = graph_emb.view(batch_size, seq_len, -1)
        
        # temporal aggregation
        x = self.neck(x)  # (B, H)
        
        # classification
        logits = self.head(x)  # (B, num_classes)
        
        return logits
