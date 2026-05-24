import json
import contextlib
import gc
import io
import warnings

import torch


def _load_weights(path):
    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(path, weights_only=True, mmap=True, **kwargs)
    except TypeError:
        try:
            return torch.load(path, weights_only=True, **kwargs)
        except TypeError:
            return torch.load(path, **kwargs)
    except Exception as exc:
        message = str(exc)
        if "mmap" in message:
            return torch.load(path, weights_only=True, **kwargs)
        if "Weights only load failed" in message or "weights_only" in message:
            return torch.load(path, **kwargs)
        raise


def _load_state_dict_assign(model, state_dict, *, strict=False):
    try:
        return model.load_state_dict(state_dict, strict=strict, assign=True)
    except TypeError:
        return model.load_state_dict(state_dict, strict=strict)


def _import_autoencoder_factory():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp\.autocast.*")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            from third_party.stable_audio_tools.stable_audio_tools.models.autoencoders import create_autoencoder_from_config
    return create_autoencoder_from_config

def get_model(model_config, path):
    with open(model_config) as f:
        model_config = json.load(f)
    state_dict = _load_weights(path)
    create_autoencoder_from_config = _import_autoencoder_factory()
    model = create_autoencoder_from_config(model_config)
    weights = state_dict.get("state_dict", state_dict)
    _load_state_dict_assign(model, weights, strict=False)
    del weights
    del state_dict
    gc.collect()
    model.requires_grad_(False)
    return model
