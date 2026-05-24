import math

import numpy as np
import torch.nn as nn

from dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int = 64, strides: list = [2, 4, 8, 8], d_latent: int = 64):
        super().__init__()
        block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block.append(EncoderBlock(d_model, stride=stride))
        block.extend([Snake1d(d_model), WNConv1d(d_model, d_latent, kernel_size=3, padding=1)])
        self.block = nn.Sequential(*block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, input_channel, channels, rates, d_out: int = 1):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        output_dim = channels
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers.append(DecoderBlock(input_dim, output_dim, stride))
        layers.extend([Snake1d(output_dim), WNConv1d(output_dim, d_out, kernel_size=7, padding=3), nn.Tanh()])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

