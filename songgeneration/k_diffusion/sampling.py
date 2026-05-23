import math

import torch


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Construct the Karras noise schedule used by k-diffusion samplers."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)


def to_d(x, sigma, denoised):
    """Convert denoiser output to the ODE derivative expected by these samplers."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def _default_extra_args(extra_args):
    return {} if extra_args is None else extra_args


def _call_callback(callback, x, i, sigma, sigma_hat, denoised):
    if callback is not None:
        callback(
            {
                "x": x,
                "i": i,
                "sigma": sigma,
                "sigma_hat": sigma_hat,
                "denoised": denoised,
            }
        )


@torch.no_grad()
def sample_euler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Minimal Euler sampler compatible with k_diffusion.sampling.sample_euler."""
    extra_args = _default_extra_args(extra_args)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(sigmas) - 1):
        gamma = (
            min(s_churn / (len(sigmas) - 1), math.sqrt(2) - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + (
                torch.randn_like(x) * s_noise * (sigma_hat**2 - sigmas[i] ** 2).sqrt()
            )

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        _call_callback(callback, x, i, sigmas[i], sigma_hat, denoised)
        x = x + d * (sigmas[i + 1] - sigma_hat)

    return x


@torch.no_grad()
def sample_heun(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Minimal Heun sampler compatible with k_diffusion.sampling.sample_heun."""
    extra_args = _default_extra_args(extra_args)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(sigmas) - 1):
        gamma = (
            min(s_churn / (len(sigmas) - 1), math.sqrt(2) - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + (
                torch.randn_like(x) * s_noise * (sigma_hat**2 - sigmas[i] ** 2).sqrt()
            )

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        _call_callback(callback, x, i, sigmas[i], sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat

        if sigmas[i + 1] == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            x = x + (d + d_2) * 0.5 * dt

    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Minimal DPM++ 2M sampler compatible with k_diffusion.sampling.sample_dpmpp_2m."""
    extra_args = _default_extra_args(extra_args)
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    h_last = None

    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        _call_callback(callback, x, i, sigmas[i], sigmas[i], denoised)

        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t = -sigmas[i].log()
            t_next = -sigmas[i + 1].log()
            h = t_next - t

            if old_denoised is None or h_last is None:
                denoised_d = denoised
            else:
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised

            x = (sigmas[i + 1] / sigmas[i]) * x - (-h).expm1() * denoised_d
            old_denoised = denoised
            h_last = h

    return x
