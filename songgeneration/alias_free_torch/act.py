import torch.nn as nn

from .resample import DownSample1d, DownSample2d, UpSample1d, UpSample2d


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class Activation2d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample2d(up_ratio, up_kernel_size)
        self.downsample = DownSample2d(down_ratio, down_kernel_size)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))
