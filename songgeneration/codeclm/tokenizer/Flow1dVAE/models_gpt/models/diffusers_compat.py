# Copyright 2025 The HuggingFace Team. All rights reserved.
# Adapted from Hugging Face Diffusers v0.37.1 under the Apache License, Version 2.0.

import math
import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)


def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
    """Local subset of diffusers.utils.torch_utils.randn_tensor."""
    if isinstance(device, str):
        device = torch.device(device)

    rand_device = device
    batch_size = shape[0]
    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = torch.device("cpu")
            if device.type != "mps":
                logger.info(
                    "The passed generator was created on 'cpu' even though a tensor on %s was expected. "
                    "Tensors will be created on 'cpu' and then moved to %s.",
                    device,
                    device,
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        single_shape = (1,) + tuple(shape[1:])
        latents = [
            torch.randn(single_shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        return torch.cat(latents, dim=0).to(device)

    return torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)


def get_timestep_embedding(
    timesteps,
    embedding_dim,
    flip_sin_to_cos=False,
    downscale_freq_shift=1,
    scale=1,
    max_period=10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def get_activation(act_fn):
    act_fn = act_fn.lower()
    if act_fn in {"silu", "swish"}:
        return nn.SiLU()
    if act_fn == "gelu":
        return nn.GELU()
    if act_fn == "relu":
        return nn.ReLU()
    raise ValueError(f"activation function {act_fn} is not supported by the local diffusers compatibility subset")


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels,
        time_embed_dim,
        act_fn="silu",
        out_dim=None,
        post_act_fn=None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False) if cond_proj_dim is not None else None
        self.act = get_activation(act_fn)
        self.linear_2 = nn.Linear(time_embed_dim, out_dim if out_dim is not None else time_embed_dim, sample_proj_bias)
        self.post_act = get_activation(post_act_fn) if post_act_fn is not None else None

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels, flip_sin_to_cos, downscale_freq_shift, scale=1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
