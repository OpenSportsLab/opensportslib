import torch
import torch.nn as nn

from opensportslib.models.backbones.builder import build_backbone
from opensportslib.models.neck.builder import build_neck
from opensportslib.models.heads.builder import build_head
from opensportslib.core.config.accessors import (
    get_component_name_by_kind,
    get_component_params_by_kind,
    get_data_params,
    get_data_sampling,
)
from opensportslib.core.utils.config_normalize import normalize_builder_cfg


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
        params = get_data_params(config)
        objects_cfg = params.get("objects", {}) if isinstance(params, dict) else {}
        feature_dim = int(objects_cfg.get("feature_dim", 8))
        self.num_frames = sampling.get("num_frames")

        def _component_cfg(kind):
            params = dict(get_component_params_by_kind(config, kind) or {})
            params.setdefault("type", get_component_name_by_kind(config, kind))
            return normalize_builder_cfg(params, kind=kind)

        # backbone: graph encoder
        self.backbone = build_backbone(
            _component_cfg("encoder"),
            default_args={"input_dim": feature_dim}
        )

        # neck: temporal aggregation
        self.neck = build_neck(
            _component_cfg("adapter"),
            default_args={"window_size": self.num_frames}
        )

        # head: classifier
        self.head = build_head(
            _component_cfg("head"),
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
