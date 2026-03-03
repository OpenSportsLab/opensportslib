import torch
from torch import nn
from opensportslib.core.utils.data import batch_tensor, unbatch_tensor

def build_neck(cfg, default_args=None):
    if cfg.type == "MV_Aggregate":
        neck = MVAggregate(
            agr_type=cfg.agr_type,
            model=default_args["model"],
            feat_dim=default_args["feat_dim"],
            lifting_net=default_args["lifting_net"] if "lifting_net" in default_args else nn.Sequential()
        )
    
    elif cfg.type == "TemporalAggregation":
        neck = TemporalAggregation(
            temporal_type=cfg.agr_type,
            hidden_dim=cfg.hidden_dim,
            window_size=default_args["window_size"],
            dropout=cfg.dropout,
            use_position_encoding=getattr(cfg, "use_position_encoding", False),
            num_attention_heads=getattr(cfg, "num_attention_heads", 4),
            lstm_dropout=getattr(cfg, "lstm_dropout", 0.1)
        )
    else:
        raise ValueError(f"Unknown neck type: {cfg.type}")
    return neck


class MVAggregate(nn.Module):
    def __init__(self, agr_type, model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type
        self.model = model
        self.feat_dim = feat_dim
        self.lifting_net = lifting_net
        print("Inside NECK BUILDER - AGR TYPE:", self.agr_type)

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)
        else:
            # avg
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )
    
    def forward(self, mvimages):
        pooled_view, attention = self.aggregation_model(mvimages)
        inter = self.inter(pooled_view)
        return inter, attention

        

class WeightedAggregate(nn.Module):
    def __init__(self,  model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        

        self.relu = nn.ReLU()
   


    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))


        ##################### VIEW ATTENTION #####################

        # S = source length 
        # N = batch size
        # E = embedding dimension
        # L = target length

        aux = torch.matmul(aux, self.attention_weights)
        # Dimension S, E for two views (2,512)

        # Dimension N, S, E
        aux_t = aux.permute(0, 2, 1)

        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)
        
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T

        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))

        final_attention_weights = torch.sum(final_attention_weights, 1)

        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))

        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class ViewAvgAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux

class TemporalAggregation(nn.Module):
    def __init__(self, temporal_type, hidden_dim, window_size, dropout=0.1,
                 use_position_encoding=False, num_attention_heads=4, lstm_dropout=0.3):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.temporal_type = temporal_type
        self.hidden_dim = hidden_dim
        self.feat_dim = hidden_dim * 2 if temporal_type == "bilstm" else hidden_dim
        self.use_position_encoding = use_position_encoding
        self.lstm_dropout = lstm_dropout

        # learnable temporal position encoding (only used when explicitly enabled)
        if self.use_position_encoding:
            self.temporal_position_encoding = nn.Parameter(
                torch.randn(1, window_size, hidden_dim) * 0.02
            )

        # build temporal module
        self.temporal = self._build_temporal_module(temporal_type, hidden_dim, dropout)

    def _build_temporal_module(self, temporal_type, hidden_dim, dropout):
        if temporal_type == 'bilstm':
            return nn.LSTM(
                hidden_dim, hidden_dim, num_layers=2,
                batch_first=True, bidirectional=True, dropout=self.lstm_dropout
            )

        elif temporal_type == 'tcn':
            return nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )

        elif temporal_type == 'attention':
            return nn.MultiheadAttention(
                hidden_dim, num_heads=self.num_attention_heads, 
                dropout=dropout, batch_first=True
            )

        else:  # avgpool, maxpool
            return None

    def forward(self, x):
        seq_len = x.size(1)

        if self.use_position_encoding:
            x = x + self.temporal_position_encoding[:, :seq_len, :]

        if self.temporal_type == 'avgpool':
            x = torch.mean(x, dim=1)

        elif self.temporal_type == 'maxpool':
            x = torch.max(x, dim=1)[0]

        elif self.temporal_type == 'tcn':
            x = x.permute(0, 2, 1)
            x = self.temporal(x)
            x = x.permute(0, 2, 1)
            x = torch.max(x, dim=1)[0]

        elif self.temporal_type == 'attention':
            x, _ = self.temporal(x, x, x)
            x = torch.max(x, dim=1)[0]

        elif self.temporal_type == 'bilstm':
            lstm_out, _ = self.temporal(x)
            x = torch.max(lstm_out, dim=1)[0]

        return x