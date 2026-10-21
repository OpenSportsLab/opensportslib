"""Deterministic CUDA backward for reference-equivalent temporal resampling.

PyTorch's linear-upsample/adaptive-max-pool CUDA backward uses atomic updates.
These alternatives keep the native forward and gather gradients in a fixed
reduction order, so strict deterministic training can remain enabled.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size):
        ctx.length = x.shape[-1]
        return F.interpolate(x, size=size, mode="linear", align_corners=True)

    @staticmethod
    def backward(ctx, gradient):
        n, m = ctx.length, gradient.shape[-1]
        dtype = torch.float64 if gradient.dtype == torch.float64 else torch.float32
        position = torch.arange(m, device=gradient.device, dtype=dtype)
        position = position * ((n - 1) / (m - 1) if m > 1 else 0)
        lower = position.floor().long()
        upper = (lower + 1).clamp_max(n - 1)
        fraction = position - lower
        index = torch.arange(n, device=gradient.device)
        matrix = (lower[:, None] == index) * (1 - fraction[:, None]) + (
            upper[:, None] == index
        ) * fraction[:, None]
        with torch.autocast(gradient.device.type, enabled=False):
            result = gradient.to(dtype).matmul(matrix)
        return result.to(gradient.dtype), None


class _AdaptiveMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size):
        result, indices = F.adaptive_max_pool1d(x, size, return_indices=True)
        ctx.save_for_backward(indices)
        ctx.length = x.shape[-1]
        return result

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        positions = torch.arange(ctx.length, device=gradient.device)
        contributions = (indices[..., None] == positions) * gradient[..., None]
        return contributions.sum(-2), None


class LinearUpsample(nn.Module):
    def __init__(self, size, mode="linear", align_corners=True):
        super().__init__()
        self.size = size

    def forward(self, x):
        if x.is_cuda and torch.are_deterministic_algorithms_enabled():
            return _Linear.apply(x, self.size)
        return F.interpolate(x, size=self.size, mode="linear", align_corners=True)


class AdaptiveMaxPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        if x.is_cuda and torch.are_deterministic_algorithms_enabled():
            return _AdaptiveMax.apply(x, self.output_size)
        return F.adaptive_max_pool1d(x, self.output_size)
