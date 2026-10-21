"""Controlled temporal variants for T-DEED component experiments.

The release does not publish the search-selected ablation recipes. These are
explicit configurable comparisons, not asserted replicas of those searches.
"""

import math
import torch
from torch import nn
from .modules import SGPBlock, EDSGPMIXERLayers
from .deterministic import AdaptiveMaxPool, LinearUpsample


class RecurrentTemporal(nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        self.gru = nn.GRU(
            width, width // 2, num_layers=layers, bidirectional=True, batch_first=True
        )

    def forward(self, x):
        return self.gru(x)[0]


class PlainSGP(nn.Module):
    def __init__(self, width, layers, kernel, ratio):
        super().__init__()
        self.layers = nn.Sequential(
            *(SGPBlock(width, kernel_size=kernel, k=ratio) for _ in range(layers))
        )

    def forward(self, x):
        return self.layers(x.transpose(1, 2)).transpose(1, 2)


class SkipEncoderDecoder(nn.Module):
    def __init__(self, width, length, layers, kernel, ratio, skip):
        super().__init__()
        self.skip = skip
        self.blocks = nn.ModuleList(
            SGPBlock(width, kernel_size=kernel, k=ratio) for _ in range(2 * layers + 1)
        )
        self.pools = nn.ModuleList(
            AdaptiveMaxPool(math.ceil(length / 2 ** (i + 1))) for i in range(layers)
        )
        self.upsamples = nn.ModuleList(
            LinearUpsample(math.ceil(length / 2**i)) for i in range(layers)
        )
        self.projections = (
            nn.ModuleList(nn.Conv1d(width * 2, width, 1) for _ in range(layers))
            if skip == "concat"
            else None
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        saved = []
        layers = len(self.pools)
        for i in range(layers):
            x = self.blocks[i](x)
            saved.append(x)
            x = self.pools[i](x)
        x = self.blocks[layers](x)
        for i in reversed(range(layers)):
            x = self.upsamples[i](x)
            if self.skip == "sum":
                x = x + saved[i]
            if self.skip == "concat":
                x = self.projections[i](torch.cat([saved[i], x], dim=1))
            x = self.blocks[2 * layers - i](x)
        return x.transpose(1, 2)


def build_temporal(name, width, length, layers, kernel, ratio):
    if name == "sgp_mixer_sum":
        return EDSGPMIXERLayers(
            width, length, num_layers=layers, ks=kernel, k=ratio, concat=False
        )
    if name == "plain_sgp":
        return PlainSGP(width, layers, kernel, ratio)
    if name == "plain_gru":
        return RecurrentTemporal(width, layers)
    if name == "plain_transformer":
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, 8, 4 * width, dropout=0.1, batch_first=True, norm_first=True
            ),
            num_layers=layers,
            enable_nested_tensor=False,
        )
    if name in {"skip_none", "skip_sum", "skip_concat"}:
        return SkipEncoderDecoder(
            width, length, layers, kernel, ratio, name.removeprefix("skip_")
        )
    raise ValueError(f"Unknown T-DEED temporal variant: {name}")
