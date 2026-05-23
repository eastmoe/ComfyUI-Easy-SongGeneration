import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinc(x):
    if hasattr(torch, "sinc"):
        return torch.sinc(x)
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / (math.pi * x),
    )


def _kaiser_beta(kernel_size, half_width):
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        return 0.1102 * (attenuation - 8.7)
    if attenuation >= 21.0:
        return 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (attenuation - 21.0)
    return 0.0


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    half_size = kernel_size // 2
    beta = _kaiser_beta(kernel_size, half_width)
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ = filter_ / filter_.sum()
    return filter_.view(1, 1, kernel_size)


def kaiser_sinc_filter2d(cutoff, half_width, kernel_size):
    half_size = kernel_size // 2
    beta = _kaiser_beta(kernel_size, half_width)
    if kernel_size % 2 == 0:
        axis = torch.arange(-half_size, half_size) + 0.5
    else:
        axis = torch.arange(kernel_size) - half_size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    window_arg = 1 - (radius / half_size / math.sqrt(2)).square()
    window_arg = torch.clamp(window_arg, min=0)
    window = torch.i0(beta * torch.sqrt(window_arg)) / torch.i0(torch.tensor(beta))
    if cutoff == 0:
        filter_ = torch.zeros_like(radius)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * radius)
        filter_ = filter_ / filter_.sum()
    return filter_.view(1, 1, kernel_size, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative")
        if cutoff > 0.5:
            raise ValueError("cutoff must be no larger than 0.5")
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x):
        channels = x.shape[1]
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)


class LowPassFilter2d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative")
        if cutoff > 0.5:
            raise ValueError("cutoff must be no larger than 0.5")
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer("filter", kaiser_sinc_filter2d(cutoff, half_width, kernel_size))

    def forward(self, x):
        channels = x.shape[1]
        if self.padding:
            x = F.pad(
                x,
                (self.pad_left, self.pad_right, self.pad_left, self.pad_right),
                mode=self.padding_mode,
            )
        return F.conv2d(x, self.filter.expand(channels, -1, -1, -1), stride=self.stride, groups=channels)
