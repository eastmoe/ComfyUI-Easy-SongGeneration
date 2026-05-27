from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

PLUGIN_DIR = Path(__file__).resolve().parents[2]
SONGGEN_DIR = PLUGIN_DIR / "songgeneration"
DEFAULT_LOCALE = "zh-cn"
SONGGEN_MODEL_TYPE = "SONGGEN_MODEL"
AUTO_PROMPT_TYPES = [
    "None",
    "Pop",
    "Latin",
    "Rock",
    "Electronic",
    "Metal",
    "Country",
    "R&B/Soul",
    "Ballad",
    "Jazz",
    "World",
    "Hip-Hop",
    "Funk",
    "Soundtrack",
    "Auto",
]
V1_MODEL_NAMES = {
    "songgeneration_base",
    "songgeneration_base_new",
    "songgeneration_base_full",
    "songgeneration_large",
}

_RUNTIME_LOCK = threading.RLock()
_MODEL_CACHE: dict[tuple[Any, ...], "SongGenerationModelHandle"] = {}
_MODEL_CACHE_OWNERS: dict[str, tuple[Any, ...]] = {}
_DTYPE_CHOICES = ["float16", "bfloat16", "float32"]
_DIFFUSION_DTYPE_CHOICES = ["bfloat16", "float32"]
_QUANTIZATION_CHOICES = ["none", "fp4", "fp8", "int4", "int8"]
_QUANTIZATION_TARGETS = ["LLM", "LLM+Diffusion", "LLM+Diffusion+VAE"]
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _load_localization(locale: str) -> dict[str, Any]:
    locale_name = (locale or DEFAULT_LOCALE).strip().lower()
    paths = [
        PLUGIN_DIR / "locales" / locale_name / "nodes.json",
        PLUGIN_DIR / "local" / locale_name / "nodes.json",
    ]
    path = paths[0]
    try:
        for candidate in paths:
            if candidate.is_file():
                path = candidate
                with candidate.open("r", encoding="utf-8") as file:
                    data = json.load(file)
                break
        else:
            return {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[Easy-SongGeneration] Failed to load localization file {path}: {exc}", flush=True)
        return {}
    return data if isinstance(data, dict) else {}


_LOCALIZATION = _load_localization(os.environ.get("COMFYUI_EASY_SONGGENERATION_LOCALE", DEFAULT_LOCALE))


def _tr(path: str, default: Any) -> Any:
    value: Any = _LOCALIZATION
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _tr_text(path: str, default: str) -> str:
    value = _tr(path, default)
    return value if isinstance(value, str) else default


def _tr_mapping(path: str, default: dict[str, str]) -> dict[str, str]:
    value = _tr(path, default)
    if isinstance(value, dict) and all(isinstance(key, str) and isinstance(val, str) for key, val in value.items()):
        return value
    return default


def _tr_names(path: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = _tr(path, list(default))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return default


CATEGORY = _tr_text("category", "eastmoe/Comfy-Easy-SongGeneration")


def _ui(display_name: str, tooltip: str, **extra: Any) -> dict[str, Any]:
    extra["display_name"] = display_name
    extra["tooltip"] = tooltip
    return extra


def _ui_text(path: str, display_name: str, tooltip: str, **extra: Any) -> dict[str, Any]:
    value = _tr(path, {})
    if isinstance(value, dict):
        display_name = value.get("display_name", display_name)
        tooltip = value.get("tooltip", tooltip)
    return _ui(display_name, tooltip, **extra)


def _comfy_models_dir() -> Path:
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        return Path(folder_paths.models_dir)
    comfy_root = PLUGIN_DIR.parent.parent
    return comfy_root / "models"


def _songgen_model_root() -> Path:
    root = _comfy_models_dir() / "SongGeneration"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _register_model_folder() -> None:
    if folder_paths is None:
        return
    folder_paths.add_model_folder_path("songgeneration", str(_songgen_model_root()), is_default=True)


_register_model_folder()


def _songgen_cache_root() -> Path:
    root = _comfy_models_dir() / "SongGeneration-cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dtype_from_choice(choice: str) -> torch.dtype:
    value = (choice or "float16").strip().lower()
    if value in {"fp32", "float32"}:
        return torch.float32
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {choice}")


def _normalize_quantization_mode(quantization: str) -> str | None:
    mode = (quantization or "none").strip().lower()
    if mode in {"none", "off", "false", ""}:
        return None
    if mode in {"fp4", "fp4_e2m1", "fp4_e2m1fn", "fp4_e2m1fn_x2", "float4_e2m1fn_x2"}:
        return "fp4"
    if mode in {"fp8", "fp8_e4m3fn"}:
        return "fp8"
    if mode in {"int4", "int8"}:
        return mode
    raise ValueError(f"Unsupported quantization format: {quantization}")


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(GIT_LFS_POINTER_PREFIX)) == GIT_LFS_POINTER_PREFIX
    except FileNotFoundError:
        return False


def _torch_load_weights(path: Path, map_location: str | torch.device = "cpu"):
    load_path = str(path)
    use_mmap = str(map_location) == "cpu"
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{load_path} is a Git LFS pointer, not downloaded model weights. "
            "Install Git LFS and run `git lfs pull`, or use the model download workflow, then retry."
        )

    def _load(*, weights_only: bool | None, mmap: bool):
        kwargs: dict[str, Any] = {"map_location": map_location}
        if weights_only is not None:
            kwargs["weights_only"] = weights_only
        if mmap:
            kwargs["mmap"] = True
        return torch.load(load_path, **kwargs)

    def _is_weights_only_error(exc: Exception) -> bool:
        message = str(exc)
        return "Weights only load failed" in message or "weights_only" in message

    def _load_unsafe_pickle(*, mmap: bool):
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
        return _load(weights_only=True, mmap=use_mmap)
    except TypeError as exc:
        message = str(exc)
        if use_mmap and "mmap" in message:
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
            return _load_unsafe_pickle(mmap=use_mmap)
        message = str(exc)
        if use_mmap and "mmap" in message:
            try:
                return _load(weights_only=True, mmap=False)
            except Exception as retry_exc:
                if _is_weights_only_error(retry_exc):
                    return _load_unsafe_pickle(mmap=False)
                raise
        raise


def _signature_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _path_signature(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None}
    text = str(path)
    candidate = Path(text).expanduser()
    if candidate.is_file():
        stat = candidate.stat()
        return {
            "path": str(candidate.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return {"path": text, "missing": True}


def _quantization_cache_path(scope: str, signature: dict[str, Any], mode: str) -> tuple[Path, dict[str, Any]]:
    metadata = {
        "cache_version": 1,
        "format": "songgeneration-weight-only-linear",
        "scope": scope,
        "mode": mode,
        "signature": signature,
    }
    safe_scope = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in scope)
    filename = f"{safe_scope}-{mode}-{_signature_digest(metadata)}.pt"
    return _songgen_cache_root() / filename, metadata
