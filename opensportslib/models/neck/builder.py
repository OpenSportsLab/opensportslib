import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
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
    elif cfg.type == "MaxPool":
        neck = MaxPool(nb_frames=cfg.nb_frames)
    elif cfg.type == "MaxPool++":
        neck = MaxPool_temporally_aware(nb_frames=cfg.nb_frames)
    elif cfg.type == "AvgPool":
        neck = AvgPool(nb_frames=cfg.nb_frames)
    elif cfg.type == "AvgPool++":
        neck = AvgPool_temporally_aware(nb_frames=cfg.nb_frames)
    elif cfg.type == "NetRVLAD":
        neck = NetRVLAD(
            vocab_size=cfg.vocab_size,
            input_dim=cfg.input_dim,
        )
    elif cfg.type == "NetRVLAD++":
        neck = NetRVLAD_temporally_aware(
            vocab_size=cfg.vocab_size,
            input_dim=cfg.input_dim,
        )
    elif cfg.type == "NetVLAD":
        neck = NetVLAD(
            vocab_size=cfg.vocab_size,
            input_dim=cfg.input_dim,
        )
    elif cfg.type == "NetVLAD++":
        neck = NetVLAD_temporally_aware(
            vocab_size=cfg.vocab_size,
            input_dim=cfg.input_dim,
        )
    elif cfg.type == "CNN++":
        neck = CNN_temporally_aware(
            input_size=cfg.input_size,
            num_classes=cfg.num_classes,
            chunk_size=cfg.chunk_size,
            dim_capsule=cfg.dim_capsule,
            receptive_field=cfg.receptive_field,
            num_detections=cfg.num_detections,
            framerate=cfg.framerate,
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


class CNN_temporally_aware(torch.nn.Module):
    def __init__(
        self,
        input_size=512,
        num_classes=3,
        chunk_size=240,
        dim_capsule=16,
        receptive_field=80,
        num_detections=5,
        framerate=2,
    ):
        super(CNN_temporally_aware, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.dim_capsule = dim_capsule
        self.receptive_field = receptive_field
        self.num_detections = num_detections
        self.chunk_size = chunk_size
        self.framerate = framerate

        self.pyramid_size_1 = int(np.ceil(receptive_field / 7))
        self.pyramid_size_2 = int(np.ceil(receptive_field / 3))
        self.pyramid_size_3 = int(np.ceil(receptive_field / 2))
        self.pyramid_size_4 = int(np.ceil(receptive_field))

        # Base Convolutional Layers
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=128, kernel_size=(1, input_size)
        )
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1))

        # Temporal Pyramidal Module
        self.pad_p_1 = nn.ZeroPad2d(
            (
                0,
                0,
                (self.pyramid_size_1 - 1) // 2,
                self.pyramid_size_1 - 1 - (self.pyramid_size_1 - 1) // 2,
            )
        )
        self.pad_p_2 = nn.ZeroPad2d(
            (
                0,
                0,
                (self.pyramid_size_2 - 1) // 2,
                self.pyramid_size_2 - 1 - (self.pyramid_size_2 - 1) // 2,
            )
        )
        self.pad_p_3 = nn.ZeroPad2d(
            (
                0,
                0,
                (self.pyramid_size_3 - 1) // 2,
                self.pyramid_size_3 - 1 - (self.pyramid_size_3 - 1) // 2,
            )
        )
        self.pad_p_4 = nn.ZeroPad2d(
            (
                0,
                0,
                (self.pyramid_size_4 - 1) // 2,
                self.pyramid_size_4 - 1 - (self.pyramid_size_4 - 1) // 2,
            )
        )
        self.conv_p_1 = nn.Conv2d(
            in_channels=32, out_channels=8, kernel_size=(self.pyramid_size_1, 1)
        )
        self.conv_p_2 = nn.Conv2d(
            in_channels=32, out_channels=16, kernel_size=(self.pyramid_size_2, 1)
        )
        self.conv_p_3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(self.pyramid_size_3, 1)
        )
        self.conv_p_4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(self.pyramid_size_4, 1)
        )

        # -------------------
        # Segmentation module
        # -------------------

        self.kernel_seg_size = 3
        self.pad_seg = nn.ZeroPad2d(
            (
                0,
                0,
                (self.kernel_seg_size - 1) // 2,
                self.kernel_seg_size - 1 - (self.kernel_seg_size - 1) // 2,
            )
        )
        self.conv_seg = nn.Conv2d(
            in_channels=152,
            out_channels=dim_capsule * num_classes,
            kernel_size=(self.kernel_seg_size, 1),
        )
        self.batch_seg = nn.BatchNorm2d(
            num_features=self.chunk_size, momentum=0.01, eps=0.001
        )

    def forward(self, inputs):
        # -------------------------------------
        # Temporal Convolutional neural network
        # -------------------------------------

        # Base Convolutional Layers
        conv_1 = F.relu(self.conv_1(inputs))
        # print("Conv_1 size: ", conv_1.size())

        conv_2 = F.relu(self.conv_2(conv_1))
        # print("Conv_2 size: ", conv_2.size())

        # Temporal Pyramidal Module
        conv_p_1 = F.relu(self.conv_p_1(self.pad_p_1(conv_2)))
        # print("Conv_p_1 size: ", conv_p_1.size())
        conv_p_2 = F.relu(self.conv_p_2(self.pad_p_2(conv_2)))
        # print("Conv_p_2 size: ", conv_p_2.size())
        conv_p_3 = F.relu(self.conv_p_3(self.pad_p_3(conv_2)))
        # print("Conv_p_3 size: ", conv_p_3.size())
        conv_p_4 = F.relu(self.conv_p_4(self.pad_p_4(conv_2)))
        # print("Conv_p_4 size: ", conv_p_4.size())

        concatenation = torch.cat((conv_2, conv_p_1, conv_p_2, conv_p_3, conv_p_4), 1)
        # print("Concatenation size: ", concatenation.size())

        # -------------------
        # Segmentation module
        # -------------------

        conv_seg = self.conv_seg(self.pad_seg(concatenation))
        # print("Conv_seg size: ", conv_seg.size())

        conv_seg_permuted = conv_seg.permute(0, 2, 3, 1)
        # print("Conv_seg_permuted size: ", conv_seg_permuted.size())

        conv_seg_reshaped = conv_seg_permuted.view(
            conv_seg_permuted.size()[0],
            conv_seg_permuted.size()[1],
            self.dim_capsule,
            self.num_classes,
        )
        # print("Conv_seg_reshaped size: ", conv_seg_reshaped.size())

        # conv_seg_reshaped_permuted = conv_seg_reshaped.permute(0,3,1,2)
        # print("Conv_seg_reshaped_permuted size: ", conv_seg_reshaped_permuted.size())

        conv_seg_norm = torch.sigmoid(self.batch_seg(conv_seg_reshaped))
        # print("Conv_seg_norm: ", conv_seg_norm.size())

        # conv_seg_norm_permuted = conv_seg_norm.permute(0,2,3,1)
        # print("Conv_seg_norm_permuted size: ", conv_seg_norm_permuted.size())

        output_segmentation = torch.sqrt(
            torch.sum(torch.square(conv_seg_norm - 0.5), dim=2) * 4 / self.dim_capsule
        )
        # print("Output_segmentation size: ", output_segmentation.size())

        return conv_seg, output_segmentation


class MaxPool(torch.nn.Module):
    def __init__(self, nb_frames):
        super(MaxPool, self).__init__()
        self.pooling_layer = nn.MaxPool1d(nb_frames, stride=1)

    def forward(self, inputs):
        return self.pooling_layer(inputs.permute((0, 2, 1))).squeeze(-1)


class MaxPool_temporally_aware(torch.nn.Module):
    def __init__(self, nb_frames):
        super(MaxPool_temporally_aware, self).__init__()
        self.pooling_layer_before = nn.MaxPool1d(int(nb_frames / 2), stride=1)
        self.pooling_layer_after = nn.MaxPool1d(int(nb_frames / 2), stride=1)

    def forward(self, inputs):
        nb_frames_50 = int(inputs.shape[1] / 2)
        input_before = inputs[:, :nb_frames_50, :]
        input_after = inputs[:, nb_frames_50:, :]
        inputs_before_pooled = self.pooling_layer_before(
            input_before.permute((0, 2, 1))
        ).squeeze(-1)
        inputs_after_pooled = self.pooling_layer_after(
            input_after.permute((0, 2, 1))
        ).squeeze(-1)
        inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)
        return inputs_pooled


class AvgPool(torch.nn.Module):
    def __init__(self, nb_frames):
        super(AvgPool, self).__init__()
        self.pooling_layer = nn.AvgPool1d(nb_frames, stride=1)

    def forward(self, inputs):
        return self.pooling_layer(inputs.permute((0, 2, 1))).squeeze(-1)


class AvgPool_temporally_aware(torch.nn.Module):
    def __init__(self, nb_frames):
        super(AvgPool_temporally_aware, self).__init__()
        self.pooling_layer_before = nn.AvgPool1d(int(nb_frames / 2), stride=1)
        self.pooling_layer_after = nn.AvgPool1d(int(nb_frames / 2), stride=1)

    def forward(self, inputs):
        nb_frames_50 = int(inputs.shape[1] / 2)
        input_before = inputs[:, :nb_frames_50, :]
        input_after = inputs[:, nb_frames_50:, :]
        inputs_before_pooled = self.pooling_layer_before(
            input_before.permute((0, 2, 1))
        ).squeeze(-1)
        inputs_after_pooled = self.pooling_layer_after(
            input_after.permute((0, 2, 1))
        ).squeeze(-1)
        inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)
        return inputs_pooled


class NetRVLAD(torch.nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(NetRVLAD, self).__init__()
        self.pooling_layer = NetRVLAD_core(
            cluster_size=vocab_size, feature_size=input_dim, add_batch_norm=True
        )

    def forward(self, inputs):
        return self.pooling_layer(inputs)


class NetRVLAD_temporally_aware(torch.nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(NetRVLAD_temporally_aware, self).__init__()
        self.pooling_layer_before = NetRVLAD_core(
            cluster_size=int(vocab_size / 2),
            feature_size=input_dim,
            add_batch_norm=True,
        )
        self.pooling_layer_after = NetRVLAD_core(
            cluster_size=int(vocab_size / 2),
            feature_size=input_dim,
            add_batch_norm=True,
        )

    def forward(self, inputs):
        nb_frames_50 = int(inputs.shape[1] / 2)
        inputs_before_pooled = self.pooling_layer_before(inputs[:, :nb_frames_50, :])
        inputs_after_pooled = self.pooling_layer_after(inputs[:, nb_frames_50:, :])
        inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)
        return inputs_pooled


class NetVLAD(torch.nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(NetVLAD, self).__init__()
        self.pooling_layer = NetVLAD_core(
            cluster_size=vocab_size, feature_size=input_dim, add_batch_norm=True
        )

    def forward(self, inputs):
        return self.pooling_layer(inputs)


class NetVLAD_temporally_aware(torch.nn.Module):
    def __init__(self, vocab_size, input_dim):
        super(NetVLAD_temporally_aware, self).__init__()
        self.pooling_layer_before = NetVLAD_core(
            cluster_size=int(vocab_size / 2),
            feature_size=input_dim,
            add_batch_norm=True,
        )
        self.pooling_layer_after = NetVLAD_core(
            cluster_size=int(vocab_size / 2),
            feature_size=input_dim,
            add_batch_norm=True,
        )

    def forward(self, inputs):
        nb_frames_50 = int(inputs.shape[1] / 2)
        inputs_before_pooled = self.pooling_layer_before(inputs[:, :nb_frames_50, :])
        inputs_after_pooled = self.pooling_layer_after(inputs[:, nb_frames_50:, :])
        inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)
        return inputs_pooled


class NetVLAD_core(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD_core, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter(
            (1 / math.sqrt(feature_size)) * torch.randn(feature_size, cluster_size)
        )
        self.clusters2 = nn.Parameter(
            (1 / math.sqrt(feature_size)) * torch.randn(1, feature_size, cluster_size)
        )

        self.add_batch_norm = add_batch_norm
        self.out_dim = cluster_size * feature_size

    def forward(self, x):
        # x [BS, T, D]
        max_sample = x.size()[1]

        # LOUPE
        if self.add_batch_norm:  # normalization along feature dimension
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1, self.feature_size)
        assignment = torch.matmul(x, self.clusters)

        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = torch.sum(assignment, -2, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = torch.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)

        return vlad


class NetRVLAD_core(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetRVLAD_core, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter(
            (1 / math.sqrt(feature_size)) * torch.randn(feature_size, cluster_size)
        )
        # self.clusters2 = nn.Parameter((1/math.sqrt(feature_size))
        #         *th.randn(1, feature_size, cluster_size))
        # self.clusters = nn.Parameter(torch.rand(1,feature_size, cluster_size))
        # self.clusters2 = nn.Parameter(torch.rand(1,feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        # self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size * feature_size
        #  (+ 128 params?)

    def forward(self, x):
        max_sample = x.size()[1]

        # LOUPE
        if self.add_batch_norm:  # normalization along feature dimension
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1, self.feature_size)
        assignment = torch.matmul(x, self.clusters)

        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        # a_sum = th.sum(assignment,-2,keepdim=True)
        # a = a_sum*self.clusters2

        assignment = assignment.transpose(1, 2)

        x = x.view(-1, max_sample, self.feature_size)
        rvlad = torch.matmul(assignment, x)
        rvlad = rvlad.transpose(-1, 1)

        # vlad = vlad.transpose(1,2)
        # vlad = vlad - a

        # L2 intra norm
        rvlad = F.normalize(rvlad)

        # flattening + L2 norm
        rvlad = rvlad.reshape(-1, self.cluster_size * self.feature_size)
        rvlad = F.normalize(rvlad)

        return rvlad