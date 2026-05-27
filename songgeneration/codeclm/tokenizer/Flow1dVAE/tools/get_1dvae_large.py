import contextlib
import gc
import io
import json
import warnings

import torch


def _load_weights(path):
    lfs_pointer = b"version https://git-lfs.github.com/spec/v1"
    try:
        with open(path, "rb") as handle:
            if handle.read(len(lfs_pointer)) == lfs_pointer:
                raise RuntimeError(
                    f"{path} is a Git LFS pointer, not downloaded model weights. "
                    "Install Git LFS and run `git lfs pull`, or use the model download workflow, then retry."
                )
    except FileNotFoundError:
        pass

    def _load(*, weights_only, mmap):
        kwargs = {"map_location": "cpu"}
        if weights_only is not None:
            kwargs["weights_only"] = weights_only
        if mmap:
            kwargs["mmap"] = True
        return torch.load(path, **kwargs)

    def _is_weights_only_error(exc):
        message = str(exc)
        return "Weights only load failed" in message or "weights_only" in message

    def _load_unsafe_pickle(*, mmap):
        try:
            return _load(weights_only=False, mmap=mmap)
        except TypeError as exc:
            message = str(exc)
            if mmap and "mmap" in message:
                return _load(weights_only=False, mmap=False)
            if "weights_only" in message:
                return _load(weights_only=None, mmap=False)
            raise

    try:
        return _load(weights_only=True, mmap=True)
    except TypeError as exc:
        message = str(exc)
        if "mmap" in message:
            try:
                return _load(weights_only=True, mmap=False)
            except Exception as retry_exc:
                if _is_weights_only_error(retry_exc):
                    return _load_unsafe_pickle(mmap=False)
                raise
        if "weights_only" in message:
            return _load(weights_only=None, mmap=False)
        raise
    except Exception as exc:
        if _is_weights_only_error(exc):
            return _load_unsafe_pickle(mmap=True)
        message = str(exc)
        if "mmap" in message:
            try:
                return _load(weights_only=True, mmap=False)
            except Exception as retry_exc:
                if _is_weights_only_error(retry_exc):
                    return _load_unsafe_pickle(mmap=False)
                raise
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
