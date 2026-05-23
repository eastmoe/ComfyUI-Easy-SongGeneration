from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torchaudio

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from comfy import model_management
except ImportError:
    model_management = None

try:
    from comfy.utils import ProgressBar as ComfyProgressBar
except ImportError:
    ComfyProgressBar = None


PLUGIN_DIR = Path(__file__).resolve().parent
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
_DTYPE_CHOICES = ["float16", "bfloat16", "float32"]
_QUANTIZATION_CHOICES = ["none", "fp4", "fp8", "int4", "int8"]
_QUANTIZATION_TARGETS = ["LLM", "LLM+Diffusion", "LLM+Diffusion+VAE"]


def _load_localization(locale: str) -> dict[str, Any]:
    locale_name = (locale or DEFAULT_LOCALE).strip().lower()
    path = PLUGIN_DIR / "local" / locale_name / "nodes.json"
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
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


def _comfy_models_dir() -> Path:
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        return Path(folder_paths.models_dir)
    comfy_root = PLUGIN_DIR.parent.parent
    return comfy_root / "models" if (comfy_root / "models").exists() else PLUGIN_DIR / "models"


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


def _torch_load_weights(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=map_location)
    except Exception as exc:
        message = str(exc)
        if "Weights only load failed" in message or "weights_only" in message:
            return torch.load(str(path), map_location=map_location)
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


def _check_interrupted() -> None:
    if model_management is not None and hasattr(model_management, "throw_exception_if_processing_interrupted"):
        model_management.throw_exception_if_processing_interrupted()


class _SongGenProgress:
    def __init__(self, total: int, label: str, *, use_tqdm: bool = True) -> None:
        self.total = max(1, int(total))
        self.current = 0
        self.label = label
        self.started = time.monotonic()
        self.last_log = self.started
        self.pbar = ComfyProgressBar(self.total) if ComfyProgressBar is not None else None
        self.tqdm = (
            _tqdm(
                total=self.total,
                desc=f"[Easy-SongGeneration] {label}",
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )
            if use_tqdm and _tqdm is not None
            else None
        )
        if self.tqdm is None:
            print(f"[Easy-SongGeneration] {label}...", flush=True)
        self._send()

    def _send(self) -> None:
        if self.pbar is not None:
            self.pbar.update_absolute(self.current, self.total)

    def update(self, amount: int = 1, label: str | None = None) -> None:
        self.update_absolute(self.current + int(amount), total=self.total, label=label)

    def update_absolute(self, value: int, *, total: int | None = None, label: str | None = None) -> None:
        _check_interrupted()
        if total is not None:
            self.total = max(1, int(total))
        if label:
            self.label = label
            if self.tqdm is not None:
                self.tqdm.set_description_str(f"[Easy-SongGeneration] {self.label}")
        previous = self.current
        self.current = max(0, min(self.total, int(value)))
        self._send()
        if self.tqdm is not None:
            self.tqdm.total = self.total
            delta = self.current - previous
            if delta > 0:
                self.tqdm.update(delta)
            else:
                self.tqdm.n = self.current
                self.tqdm.refresh()
        now = time.monotonic()
        if self.tqdm is None and (now - self.last_log >= 5.0 or self.current >= self.total):
            self.last_log = now
            print(f"[Easy-SongGeneration] {self.label}: {self.current}/{self.total}", flush=True)

    def finish(self, label: str | None = None) -> None:
        if label:
            self.label = label
            if self.tqdm is not None:
                self.tqdm.set_description_str(f"[Easy-SongGeneration] {self.label}")
        self.current = self.total
        self._send()
        if self.tqdm is not None:
            self.tqdm.n = self.total
            self.tqdm.refresh()
            self.tqdm.close()
            self.tqdm = None
        else:
            print(f"[Easy-SongGeneration] {self.label}: {self.current}/{self.total}", flush=True)

    def close(self) -> None:
        if self.tqdm is not None:
            self.tqdm.close()
            self.tqdm = None


class _ProgressBridge:
    def __init__(self) -> None:
        self.label: str | None = None
        self.progress: _SongGenProgress | None = None

    def update(self, current: int, total: int, label: str | None = None) -> None:
        label = label or "生成进度"
        current = int(current)
        total = max(1, int(total))
        if self.progress is None or self.label != label or current < self.progress.current:
            self.close(finish=True)
            self.label = label
            self.progress = _SongGenProgress(total, label, use_tqdm=False)
        self.progress.update_absolute(current, total=total, label=label)

    def close(self, *, finish: bool = False) -> None:
        if self.progress is not None:
            if finish:
                self.progress.finish()
            else:
                self.progress.close()
            self.progress = None
            self.label = None


def _get_comfy_progress_module():
    text = str(SONGGEN_DIR)
    if text not in sys.path:
        sys.path.insert(0, text)
    import comfy_progress

    return comfy_progress


def _split_tensor_name(name: str) -> tuple[str, str]:
    if "." not in name:
        return "", name
    return name.rsplit(".", 1)


def _replace_module(model: torch.nn.Module, module_name: str, module: torch.nn.Module) -> None:
    parent_name, child_name = _split_tensor_name(module_name)
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def _set_module_tensor(model: torch.nn.Module, tensor_name: str, tensor: torch.Tensor) -> None:
    module_name, attr_name = _split_tensor_name(tensor_name)
    module = model.get_submodule(module_name) if module_name else model
    if attr_name in module._parameters:
        old_param = module._parameters[attr_name]
        requires_grad = bool(old_param.requires_grad) if old_param is not None else False
        module._parameters[attr_name] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        return
    if attr_name in module._buffers:
        module._buffers[attr_name] = tensor
        return
    raise KeyError(f"Cannot assign tensor {tensor_name!r}: target attribute does not exist.")


def _load_state_dict_assign(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], *, strict: bool):
    try:
        return model.load_state_dict(state_dict, strict=strict, assign=True)
    except TypeError:
        return model.load_state_dict(state_dict, strict=strict)


def _load_prefixed_state_dict_segmented(
    model: torch.nn.Module,
    checkpoint: dict[str, torch.Tensor],
    prefix: str,
    *,
    progress_label: str,
) -> None:
    prefix_text = prefix if prefix.endswith(".") else f"{prefix}."
    expected_keys = set(model.state_dict().keys())
    keys = [key for key in checkpoint.keys() if key.startswith(prefix_text)]
    progress = _SongGenProgress(len(keys), progress_label)
    loaded = 0
    for key in keys:
        _check_interrupted()
        target_key = key[len(prefix_text) :]
        tensor = checkpoint[key]
        if target_key in expected_keys and isinstance(tensor, torch.Tensor):
            _set_module_tensor(model, target_key, tensor.detach().cpu())
            loaded += 1
        progress.update(1)
    if loaded == 0:
        raise RuntimeError(f"No tensors with prefix {prefix_text!r} were loaded from checkpoint.")


def _move_module_segmented(module: torch.nn.Module, device: torch.device, dtype: torch.dtype, label: str) -> torch.nn.Module:
    children = list(module.named_children())
    progress = _SongGenProgress(max(1, len(children) + 1), label)
    for _name, child in children:
        _check_interrupted()
        child.to(device=device, dtype=dtype)
        progress.update(1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    module.to(device=device, dtype=dtype)
    progress.update(1)
    return module


def _quantization_chunk_rows(weight: torch.Tensor, mode: str) -> int:
    if weight.ndim != 2 or weight.shape[1] == 0:
        return 1
    max_temp_mb = int(os.environ.get("SONGGEN_QUANTIZE_MAX_TEMP_MB", "256"))
    multiplier = 4 if mode in {"int4", "fp4"} else 2
    bytes_per_row = max(1, weight.shape[1] * multiplier * 4)
    return max(1, min(weight.shape[0], (max_temp_mb * 1024 * 1024) // bytes_per_row))


def _fp4_e2m1fn_codes(values: torch.Tensor) -> torch.Tensor:
    abs_values = values.abs()
    magnitude = torch.zeros_like(abs_values, dtype=torch.uint8)
    magnitude = torch.where(abs_values >= 0.25, torch.ones_like(magnitude), magnitude)
    magnitude = torch.where(abs_values >= 0.75, torch.full_like(magnitude, 2), magnitude)
    magnitude = torch.where(abs_values >= 1.25, torch.full_like(magnitude, 3), magnitude)
    magnitude = torch.where(abs_values >= 1.75, torch.full_like(magnitude, 4), magnitude)
    magnitude = torch.where(abs_values >= 2.50, torch.full_like(magnitude, 5), magnitude)
    magnitude = torch.where(abs_values >= 3.50, torch.full_like(magnitude, 6), magnitude)
    magnitude = torch.where(abs_values >= 5.00, torch.full_like(magnitude, 7), magnitude)
    sign = (values < 0).to(torch.uint8) << 3
    return magnitude | sign


def _dequantize_fp4_e2m1fn(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    packed = packed.to(device=device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(out_features, -1)[:, :in_features]
    magnitude = codes & 0x07
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device, dtype=dtype)
    values = lut[magnitude.long()]
    signs = torch.where((codes & 0x08) != 0, -1.0, 1.0).to(dtype=dtype, device=device)
    return values * signs * scale.to(device=device, dtype=dtype)[:, None]


def _quantize_weight_tensor(weight: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor | None, str, torch.dtype | None]:
    if weight.ndim != 2:
        raise ValueError(f"Only 2D Linear weights can be quantized, got shape={tuple(weight.shape)}")
    mode = _normalize_quantization_mode(mode)
    if mode is None:
        raise ValueError("Quantization mode is disabled.")
    weight = weight.detach().cpu()
    out_features, in_features = weight.shape
    chunk_rows = _quantization_chunk_rows(weight, mode)
    if mode == "int8":
        qweight = torch.empty((out_features, in_features), dtype=torch.int8)
        scale = torch.empty((out_features,), dtype=torch.float32)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            chunk = weight[start:end].float()
            chunk_scale = chunk.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            qweight[start:end] = torch.round(chunk / chunk_scale[:, None]).clamp(-127, 127).to(torch.int8)
            scale[start:end] = chunk_scale
        return qweight, scale, "int8", None
    if mode == "int4":
        qweight = torch.empty((out_features, (in_features + 1) // 2), dtype=torch.uint8)
        scale = torch.empty((out_features,), dtype=torch.float32)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            chunk = weight[start:end].float()
            chunk_scale = chunk.abs().amax(dim=1).clamp(min=1e-8) / 7.0
            q = torch.round(chunk / chunk_scale[:, None]).clamp(-8, 7).to(torch.int8)
            q = (q + 8).to(torch.uint8)
            if in_features % 2:
                q = torch.nn.functional.pad(q, (0, 1))
            qweight[start:end] = q[:, 0::2] | (q[:, 1::2] << 4)
            scale[start:end] = chunk_scale
        return qweight, scale, "int4", None
    if mode == "fp4":
        qweight = torch.empty((out_features, (in_features + 1) // 2), dtype=torch.uint8)
        scale = torch.empty((out_features,), dtype=torch.float32)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            chunk = weight[start:end].float()
            chunk_scale = chunk.abs().amax(dim=1).clamp(min=1e-8) / 6.0
            q = _fp4_e2m1fn_codes((chunk / chunk_scale[:, None]).clamp(-6.0, 6.0))
            if in_features % 2:
                q = torch.nn.functional.pad(q, (0, 1))
            qweight[start:end] = q[:, 0::2] | (q[:, 1::2] << 4)
            scale[start:end] = chunk_scale
        return qweight, scale, "fp4", None
    if mode == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("Current PyTorch does not support torch.float8_e4m3fn.")
        return weight.to(torch.float8_e4m3fn), None, "fp8", torch.float8_e4m3fn
    raise ValueError(f"Unsupported quantization format: {mode}")


class QuantizedLinear(torch.nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor | None,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        mode: str,
        fp8_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.mode = mode
        self.fp8_dtype = fp8_dtype
        self.register_buffer("qweight", qweight.contiguous())
        if scale is not None:
            self.register_buffer("scale", scale.contiguous())
        else:
            self.scale = None
        if bias is not None:
            self.register_buffer("bias", bias.detach().cpu().clone())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, module: torch.nn.Linear, mode: str) -> "QuantizedLinear":
        qweight, scale, internal_mode, fp8_dtype = _quantize_weight_tensor(module.weight.detach(), mode)
        bias = module.bias.detach() if module.bias is not None else None
        return cls(qweight, scale, bias, module.in_features, module.out_features, internal_mode, fp8_dtype)

    @classmethod
    def empty_from_linear(cls, module: torch.nn.Linear, mode: str, *, has_bias: bool) -> "QuantizedLinear":
        mode = _normalize_quantization_mode(mode)
        if mode is None:
            raise ValueError("Quantization mode is disabled.")
        out_features = int(module.out_features)
        in_features = int(module.in_features)
        fp8_dtype = None
        if mode == "int8":
            qweight = torch.empty((out_features, in_features), dtype=torch.int8)
            scale = torch.empty((out_features,), dtype=torch.float32)
            internal_mode = "int8"
        elif mode in {"int4", "fp4"}:
            qweight = torch.empty((out_features, (in_features + 1) // 2), dtype=torch.uint8)
            scale = torch.empty((out_features,), dtype=torch.float32)
            internal_mode = mode
        elif mode == "fp8":
            fp8_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)
            qweight = torch.empty((out_features, in_features), dtype=fp8_dtype)
            scale = None
            internal_mode = "fp8"
        else:
            raise ValueError(f"Unsupported quantization format: {mode}")
        bias = torch.empty((out_features,), dtype=module.weight.dtype) if has_bias else None
        return cls(qweight, scale, bias, in_features, out_features, internal_mode, fp8_dtype)

    def _apply(self, fn):
        qweight = self._buffers.pop("qweight")
        super()._apply(fn)
        moved = fn(qweight)
        if self.mode in {"int8", "int4", "fp4"} and moved.dtype != qweight.dtype:
            moved = moved.to(qweight.dtype)
        elif self.fp8_dtype is not None and moved.dtype != self.fp8_dtype:
            moved = moved.to(self.fp8_dtype)
        self._buffers["qweight"] = moved
        return self

    def _weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.mode == "int8":
            return self.qweight.to(device=device, dtype=dtype) * self.scale.to(device=device, dtype=dtype)[:, None]
        if self.mode == "int4":
            packed = self.qweight.to(device=device)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            unpacked = torch.stack((low, high), dim=-1).reshape(self.out_features, -1)[:, : self.in_features]
            unpacked = unpacked.to(torch.int16) - 8
            return unpacked.to(dtype=dtype) * self.scale.to(device=device, dtype=dtype)[:, None]
        if self.mode == "fp4":
            return _dequantize_fp4_e2m1fn(
                self.qweight,
                self.scale,
                out_features=self.out_features,
                in_features=self.in_features,
                dtype=dtype,
                device=device,
            )
        return self.qweight.to(device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self._weight(input.dtype, input.device)
        bias = self.bias
        if bias is not None:
            bias = bias.to(device=input.device, dtype=input.dtype)
        return torch.nn.functional.linear(input, weight, bias)


def _replace_linear_modules(
    module: torch.nn.Module,
    quantization: str,
    *,
    module_names: list[str] | None = None,
    prefix: str = "",
    progress: _SongGenProgress | None = None,
) -> int:
    mode = _normalize_quantization_mode(quantization)
    if mode is None:
        return 0
    count = 0
    for name, child in list(module.named_children()):
        _check_interrupted()
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, QuantizedLinear.from_linear(child, mode))
            if module_names is not None:
                module_names.append(full_name)
            count += 1
            if progress is not None:
                progress.update(1, label=f"量化 Linear: {full_name}")
        else:
            count += _replace_linear_modules(child, mode, module_names=module_names, prefix=full_name, progress=progress)
    return count


def _count_linear_modules(module: torch.nn.Module) -> int:
    return sum(1 for child in module.modules() if isinstance(child, torch.nn.Linear))


def _replace_quantized_modules_from_names(
    model: torch.nn.Module,
    module_names: list[str],
    *,
    mode: str,
    state_dict: dict[str, torch.Tensor],
    progress: _SongGenProgress | None = None,
) -> None:
    for module_name in module_names:
        _check_interrupted()
        module = model.get_submodule(module_name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"Quantization cache expected Linear module {module_name!r}, got {type(module).__name__}.")
        _replace_module(
            model,
            module_name,
            QuantizedLinear.empty_from_linear(module, mode, has_bias=f"{module_name}.bias" in state_dict),
        )
        if progress is not None:
            progress.update(1, label=f"恢复量化 Linear: {module_name}")


def _float8_dtype_name(dtype: torch.dtype) -> str | None:
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return "float8_e4m3fn"
    return None


def _float8_dtype_from_name(name: str) -> torch.dtype:
    if name == "float8_e4m3fn" and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    raise RuntimeError(f"Current PyTorch does not support cached dtype: {name}")


def _pack_quantized_cache_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    packed: dict[str, torch.Tensor] = {}
    packed_float8_dtypes: dict[str, str] = {}
    for key, tensor in state_dict.items():
        tensor = tensor.detach().cpu().contiguous()
        dtype_name = _float8_dtype_name(tensor.dtype)
        if dtype_name is None:
            packed[key] = tensor
        else:
            packed[key] = tensor.view(torch.uint8).clone()
            packed_float8_dtypes[key] = dtype_name
    return packed, packed_float8_dtypes


def _unpack_quantized_cache_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = payload["state_dict"]
    for key, dtype_name in payload.get("packed_float8_dtypes", {}).items():
        if key not in state_dict:
            raise RuntimeError(f"Quantization cache is missing fp8 tensor {key!r}.")
        tensor = state_dict[key]
        if tensor.dtype != torch.uint8:
            raise RuntimeError(f"Cached fp8 tensor {key!r} was not stored as uint8.")
        state_dict[key] = tensor.contiguous().view(_float8_dtype_from_name(dtype_name))
    return state_dict


def _apply_quantization_cache(
    module: torch.nn.Module,
    *,
    scope: str,
    signature: dict[str, Any],
    quantization: str,
    rebuild_cache: bool,
) -> tuple[int, str]:
    mode = _normalize_quantization_mode(quantization)
    if mode is None:
        return 0, "disabled"
    cache_path, metadata = _quantization_cache_path(scope, signature, mode)
    if cache_path.exists() and not rebuild_cache:
        payload = _torch_load_weights(cache_path, map_location="cpu")
        if payload.get("metadata") == metadata:
            state_dict = _unpack_quantized_cache_state_dict(payload)
            module_names = list(payload.get("quantized_module_names", []))
            progress = _SongGenProgress(max(1, len(module_names)), f"加载量化缓存 {scope} ({mode})")
            try:
                _replace_quantized_modules_from_names(module, module_names, mode=mode, state_dict=state_dict, progress=progress)
                progress.finish()
            except Exception:
                progress.close()
                raise
            _load_state_dict_assign(module, state_dict, strict=False)
            print(f"[Easy-SongGeneration] Loaded quantization cache: {cache_path}", flush=True)
            return len(module_names), f"loaded {cache_path.name}"
        print(f"[Easy-SongGeneration] Ignoring stale quantization cache: {cache_path}", flush=True)

    module_names: list[str] = []
    progress = _SongGenProgress(max(1, _count_linear_modules(module)), f"构建量化缓存 {scope} ({mode})")
    try:
        count = _replace_linear_modules(module, mode, module_names=module_names, progress=progress)
        progress.finish()
    except Exception:
        progress.close()
        raise
    if count:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict, packed_float8_dtypes = _pack_quantized_cache_state_dict(module.state_dict())
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(
            {
                "metadata": metadata,
                "quantized_module_names": module_names,
                "state_dict": state_dict,
                "packed_float8_dtypes": packed_float8_dtypes,
            },
            tmp_path,
        )
        os.replace(tmp_path, cache_path)
        print(f"[Easy-SongGeneration] Saved quantization cache: {cache_path}", flush=True)
    return count, f"built {cache_path.name}" if count else "no Linear modules"


def _add_songgeneration_paths(runtime_roots: list[Path]) -> None:
    paths = [
        SONGGEN_DIR,
        SONGGEN_DIR / "codeclm" / "tokenizer",
        SONGGEN_DIR / "codeclm" / "tokenizer" / "Flow1dVAE",
        *runtime_roots,
    ]
    for path in reversed(paths):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)

    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    for root in runtime_roots + [SONGGEN_DIR]:
        hub = root / "third_party" / "hub"
        if hub.exists():
            os.environ.setdefault("TRANSFORMERS_CACHE", str(hub))
            break


def _model_choices() -> list[str]:
    root = _songgen_model_root()
    choices = []
    for config in root.rglob("config.yaml"):
        model_pt = config.parent / "model.pt"
        if model_pt.is_file():
            choices.append(config.parent.relative_to(root).as_posix())
    return sorted(choices, key=str.lower) or ["No local SongGeneration models found"]


def _resolve_model_dir(model_name: str) -> Path:
    name = (model_name or "").strip()
    if not name or name == "No local SongGeneration models found":
        raise FileNotFoundError(
            f"No SongGeneration checkpoint was found under {_songgen_model_root()}. "
            "Place a folder containing config.yaml and model.pt there."
        )
    path = Path(name).expanduser()
    if path.is_absolute():
        return path
    return _songgen_model_root() / name


def _infer_version(model_dir: Path, version: str) -> str:
    if version != "auto":
        return version
    model_name = model_dir.name.lower().replace("-", "_")
    return "v1" if model_name in V1_MODEL_NAMES else "v2"


def _runtime_roots(model_dir: Path, runtime_root: str) -> list[Path]:
    roots = []
    text = (runtime_root or "auto").strip()
    if text and text.lower() != "auto":
        roots.append(Path(text).expanduser())
    roots.extend([model_dir, model_dir.parent, _songgen_model_root(), SONGGEN_DIR])
    deduped = []
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _resolve_existing_path(value: str | None, roots: list[Path]) -> str | None:
    if not value:
        return value
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return str(candidate)
    return str(path)


def _resolve_runtime_file(relative_path: str, roots: list[Path]) -> str:
    for root in roots:
        candidate = root / relative_path
        if candidate.is_file():
            return str(candidate)
    return str(SONGGEN_DIR / relative_path)


def _register_resolvers(model_dir: Path):
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda: model_dir.name)
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))
    return OmegaConf


def _audio_to_waveform(audio: dict[str, Any], batch_index: int = 0) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise TypeError("Expected ComfyUI AUDIO input with waveform and sample_rate.")
    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().float().cpu()
    if waveform.ndim == 3:
        batch_index = max(0, min(int(batch_index), waveform.shape[0] - 1))
        waveform = waveform[batch_index]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Unsupported AUDIO waveform shape: {tuple(waveform.shape)}")
    return waveform.contiguous(), int(audio.get("sample_rate") or 48000)


def _write_temp_wav(audio: dict[str, Any], batch_index: int = 0) -> str:
    waveform, sample_rate = _audio_to_waveform(audio, batch_index=batch_index)
    if sample_rate != 48000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 48000)
        sample_rate = 48000
    handle = tempfile.NamedTemporaryFile(prefix="songgeneration_prompt_", suffix=".wav", delete=False)
    handle.close()
    pcm = (waveform.clamp(-1.0, 1.0).t().contiguous().numpy() * 32767.0).astype(np.int16)
    with wave.open(handle.name, "wb") as wav_file:
        wav_file.setnchannels(int(waveform.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return handle.name


def _to_comfy_audio(waveform: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    waveform = waveform.detach().cpu().float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3:
        raise ValueError(f"Generated audio has unsupported shape: {tuple(waveform.shape)}")
    return {"waveform": waveform.contiguous(), "sample_rate": int(sample_rate)}


@dataclass(frozen=True)
class GenerationOptions:
    lyrics: str
    descriptions: str
    generate_type: str
    auto_prompt_audio_type: str
    prompt_audio: dict[str, Any] | None
    prompt_audio_batch_index: int
    seed: int
    duration: float
    extend_stride: float
    temperature: float
    cfg_coef: float
    top_k: int
    top_p: float
    use_sampling: bool
    record_tokens: bool
    record_window: int
    chunk_size: int


class SongGenerationModelHandle:
    def __init__(
        self,
        *,
        cache_key: tuple[Any, ...],
        model_dir: Path,
        runtime_roots: list[Path],
        version: str,
        gpu_id: int | None,
        use_flash_attn: bool,
        segmented_load: bool,
        quantization: str,
        quantization_target: str,
        rebuild_quantization_cache: bool,
        llm_dtype: torch.dtype,
        diffusion_dtype: torch.dtype,
        vae_dtype: torch.dtype,
    ) -> None:
        self.cache_key = cache_key
        self.model_dir = model_dir
        self.runtime_roots = runtime_roots
        self.version = version
        self.gpu_id = gpu_id
        self.use_flash_attn = use_flash_attn
        self.segmented_load = bool(segmented_load)
        self.quantization = quantization
        self.quantization_target = quantization_target
        self.rebuild_quantization_cache = bool(rebuild_quantization_cache)
        self.llm_dtype = llm_dtype
        self.diffusion_dtype = diffusion_dtype
        self.vae_dtype = vae_dtype
        self.model = None
        self.cfg = None
        self.sample_rate = 48000
        self.auto_prompt = None
        self._modules: dict[str, Any] = {}
        self.quantization_info: dict[str, Any] = {}
        self._load()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _device(self) -> torch.device:
        if self.gpu_id is None or self.gpu_id < 0:
            return torch.device("cuda")
        return torch.device(f"cuda:{int(self.gpu_id)}")

    def _quantizes(self, name: str) -> bool:
        mode = _normalize_quantization_mode(self.quantization)
        if mode is None:
            return False
        target = (self.quantization_target or "LLM").lower()
        if name == "llm":
            return True
        if name == "diffusion":
            return "diffusion" in target
        if name == "vae":
            return "vae" in target
        return False

    def _configure_tango_tokenizer(self, tokenizer: torch.nn.Module, *, scope_prefix: str) -> torch.nn.Module:
        device = self._device()
        tango = getattr(tokenizer, "model", None)
        if tango is None:
            return tokenizer.eval().to(device=device)

        diffusion = getattr(tango, "model", None)
        if isinstance(diffusion, torch.nn.Module):
            if self._quantizes("diffusion"):
                try:
                    count, status = _apply_quantization_cache(
                        diffusion,
                        scope=f"{self.model_dir.name}-{scope_prefix}-diffusion",
                        signature={
                            "kind": "diffusion",
                            "checkpoint": _path_signature(getattr(tango, "model_path", None)),
                            "cfg_audio_tokenizer": str(getattr(self.cfg, "audio_tokenizer_checkpoint", "")),
                            "cfg_audio_tokenizer_sep": str(getattr(self.cfg, "audio_tokenizer_checkpoint_sep", "")) if "audio_tokenizer_checkpoint_sep" in self.cfg.keys() else "",
                        },
                        quantization=self.quantization,
                        rebuild_cache=self.rebuild_quantization_cache,
                    )
                    self.quantization_info[f"{scope_prefix}_diffusion"] = {"count": count, "status": status}
                except Exception as exc:
                    raise RuntimeError(f"Failed to apply diffusion quantization cache: {exc}") from exc
            if self.segmented_load:
                _move_module_segmented(diffusion, device, self.diffusion_dtype, f"移动 {scope_prefix} Diffusion 到 {device}")
            else:
                diffusion.to(device=device, dtype=self.diffusion_dtype)
            if hasattr(diffusion, "init_device_dtype"):
                diffusion.init_device_dtype(device, self.diffusion_dtype)

        vae = getattr(tango, "vae", None)
        if isinstance(vae, torch.nn.Module):
            if self._quantizes("vae"):
                count, status = _apply_quantization_cache(
                    vae,
                    scope=f"{self.model_dir.name}-{scope_prefix}-vae",
                    signature={
                        "kind": "vae",
                        "checkpoint": _path_signature(str(getattr(self.cfg, "vae_model", ""))),
                        "config": _path_signature(str(getattr(self.cfg, "vae_config", ""))),
                    },
                    quantization=self.quantization,
                    rebuild_cache=self.rebuild_quantization_cache,
                )
                self.quantization_info[f"{scope_prefix}_vae"] = {"count": count, "status": status}
            if self.segmented_load:
                _move_module_segmented(vae, device, self.vae_dtype, f"移动 {scope_prefix} VAE 到 {device}")
            else:
                vae.to(device=device, dtype=self.vae_dtype)

        tango.device = str(device)
        tango.diffusion_dtype = self.diffusion_dtype
        tango.vae_dtype = self.vae_dtype
        return tokenizer.eval()

    def _load_lm(self, builders, cfg, ckpt_path: Path) -> torch.nn.Module:
        audiolm = builders.get_lm_model(cfg, version=self.version)
        mode = _normalize_quantization_mode(self.quantization)
        signature = {"kind": "llm", "checkpoint": _path_signature(ckpt_path), "version": self.version}
        if self._quantizes("llm") and mode is not None:
            cache_path, metadata = _quantization_cache_path(f"{self.model_dir.name}-llm", signature, mode)
            if cache_path.exists() and not self.rebuild_quantization_cache:
                try:
                    payload = _torch_load_weights(cache_path, map_location="cpu")
                    if payload.get("metadata") == metadata:
                        state_dict = _unpack_quantized_cache_state_dict(payload)
                        module_names = list(payload.get("quantized_module_names", []))
                        progress = _SongGenProgress(max(1, len(module_names)), f"加载 LLM 量化缓存 ({mode})")
                        try:
                            _replace_quantized_modules_from_names(
                                audiolm, module_names, mode=mode, state_dict=state_dict, progress=progress
                            )
                            progress.finish()
                        except Exception:
                            progress.close()
                            raise
                        _load_state_dict_assign(audiolm, state_dict, strict=False)
                        self.quantization_info["llm"] = {"count": len(module_names), "status": f"loaded {cache_path.name}"}
                        return audiolm
                    print(f"[Easy-SongGeneration] Ignoring stale LLM quantization cache: {cache_path}", flush=True)
                except Exception as exc:
                    print(f"[Easy-SongGeneration] Failed to load LLM quantization cache, rebuilding: {exc}", flush=True)
                    audiolm = builders.get_lm_model(cfg, version=self.version)

        checkpoint = _torch_load_weights(ckpt_path, map_location="cpu")
        if self.segmented_load:
            _load_prefixed_state_dict_segmented(audiolm, checkpoint, "audiolm", progress_label="分段加载 LLM 权重")
        else:
            audiolm_state_dict = {k.replace("audiolm.", ""): v for k, v in checkpoint.items() if k.startswith("audiolm.")}
            audiolm.load_state_dict(audiolm_state_dict, strict=False)
        del checkpoint
        gc.collect()

        if self._quantizes("llm") and mode is not None:
            count, status = _apply_quantization_cache(
                audiolm,
                scope=f"{self.model_dir.name}-llm",
                signature=signature,
                quantization=mode,
                rebuild_cache=self.rebuild_quantization_cache,
            )
            self.quantization_info["llm"] = {"count": count, "status": status}
        return audiolm

    def _load(self) -> None:
        progress = _SongGenProgress(7, "加载 SongGeneration 模型")
        try:
            _add_songgeneration_paths(self.runtime_roots)
            progress.update(1, label="准备运行时路径")

            import generate as sg_generate
            from codeclm.models import CodecLM, builders

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. SongGeneration inference requires a CUDA GPU.")
            if self.gpu_id is not None and self.gpu_id >= 0:
                torch.cuda.set_device(int(self.gpu_id))
            progress.update(1, label="初始化 CUDA")

            OmegaConf = _register_resolvers(self.model_dir)
            cfg_path = self.model_dir / "config.yaml"
            ckpt_path = self.model_dir / "model.pt"
            if not cfg_path.is_file() or not ckpt_path.is_file():
                raise FileNotFoundError(f"Expected config.yaml and model.pt under {self.model_dir}")

            cfg = OmegaConf.load(cfg_path)
            cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            cfg.mode = "inference"
            cfg.lm.use_flash_attn_2 = bool(self.use_flash_attn)
            cfg.audio_tokenizer_checkpoint = _resolve_existing_path(str(cfg.audio_tokenizer_checkpoint), self.runtime_roots)
            if "audio_tokenizer_checkpoint_sep" in cfg.keys():
                cfg.audio_tokenizer_checkpoint_sep = _resolve_existing_path(
                    str(cfg.audio_tokenizer_checkpoint_sep), self.runtime_roots
                )
            self.cfg = cfg
            self.sample_rate = int(getattr(cfg, "sample_rate", 48000))
            progress.update(1, label="读取模型配置")

            auto_prompt_path = _resolve_runtime_file("tools/new_auto_prompt.pt", self.runtime_roots)
            self.auto_prompt = _torch_load_weights(Path(auto_prompt_path), map_location="cpu")
            progress.update(1, label="加载自动参考音频提示")

            seperate_tokenizer = None
            if "audio_tokenizer_checkpoint_sep" in cfg.keys():
                tokenizer_builder = getattr(builders, "get_audio_tokenizer_model_cpu", builders.get_audio_tokenizer_model)
                seperate_tokenizer = tokenizer_builder(cfg.audio_tokenizer_checkpoint_sep, cfg)
                seperate_tokenizer = self._configure_tango_tokenizer(seperate_tokenizer, scope_prefix="separate-tokenizer")
            progress.update(1, label="加载音频分词器")

            audiolm = self._load_lm(builders, cfg, ckpt_path)
            audiolm = audiolm.eval()
            progress.update(1, label="加载 LLM")
            if self.segmented_load:
                audiolm = _move_module_segmented(audiolm, self._device(), self.llm_dtype, f"移动 LLM 到 {self._device()}")
            else:
                _check_interrupted()
                audiolm = audiolm.to(device=self._device(), dtype=self.llm_dtype)

            self.model = CodecLM(
                name="ComfyUI-SongGeneration",
                lm=audiolm,
                audiotokenizer=None,
                max_duration=cfg.max_dur,
                seperate_tokenizer=seperate_tokenizer,
            )
            self._modules = {"sg_generate": sg_generate, "builders": builders}
            progress.finish("SongGeneration 模型加载完成")
        except Exception:
            progress.close()
            raise

    def release(self, *, clear_cuda_cache: bool = True) -> str:
        _MODEL_CACHE.pop(self.cache_key, None)
        self.model = None
        self.auto_prompt = None
        self.cfg = None
        self._modules.clear()
        gc.collect()
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
        if clear_cuda_cache and model_management is not None:
            model_management.soft_empty_cache()
        return "SongGeneration model released."

    def _make_args(self, options: GenerationOptions) -> SimpleNamespace:
        top_k = None if int(options.top_k) <= 0 else int(options.top_k)
        temperature = None if float(options.temperature) <= 0 else float(options.temperature)
        duration = None if float(options.duration) <= 0 else float(options.duration)
        return SimpleNamespace(
            gpu_id=self.gpu_id,
            duration=duration,
            extend_stride=float(options.extend_stride),
            temperature=temperature,
            cfg_coef=float(options.cfg_coef),
            top_k=top_k,
            top_p=float(options.top_p),
            use_sampling=bool(options.use_sampling),
            record_tokens=bool(options.record_tokens),
            record_window=int(options.record_window),
            chunk_size=int(options.chunk_size),
        )

    def _prepare_audio_prompt(self, prompt_audio: dict[str, Any], batch_index: int) -> tuple[Any, Any, Any, Any, Any, Any]:
        sg_generate = self._modules["sg_generate"]
        builders = self._modules["builders"]
        temp_path = _write_temp_wav(prompt_audio, batch_index=batch_index)
        try:
            demucs_model_path = _resolve_runtime_file("third_party/demucs/ckpt/htdemucs.pth", self.runtime_roots)
            demucs_config_path = _resolve_runtime_file("third_party/demucs/ckpt/htdemucs.yaml", self.runtime_roots)
            separator = sg_generate.Separator(demucs_model_path, demucs_config_path, gpu_id=self.gpu_id or 0)
            tokenizer_builder = getattr(builders, "get_audio_tokenizer_model_cpu", builders.get_audio_tokenizer_model)
            audio_tokenizer = tokenizer_builder(self.cfg.audio_tokenizer_checkpoint, self.cfg)
            audio_tokenizer = self._configure_tango_tokenizer(audio_tokenizer, scope_prefix="prompt-tokenizer")
            with torch.no_grad():
                pmt_wav, vocal_wav, bgm_wav = separator.run(temp_path)
            raw_pmt_wav, raw_vocal_wav, raw_bgm_wav = pmt_wav, vocal_wav, bgm_wav
            if pmt_wav.dim() == 2:
                pmt_wav = pmt_wav[None]
            if vocal_wav.dim() == 2:
                vocal_wav = vocal_wav[None]
            if bgm_wav.dim() == 2:
                bgm_wav = bgm_wav[None]
            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(pmt_wav.cuda())
                if self.model.seperate_tokenizer is not None:
                    vocal_wav, bgm_wav = self.model.seperate_tokenizer.encode(vocal_wav.cuda(), bgm_wav.cuda())
            del audio_tokenizer
            del separator
            torch.cuda.empty_cache()
            return pmt_wav, vocal_wav, bgm_wav, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _prepare_conditioning(self, options: GenerationOptions) -> dict[str, Any]:
        if options.prompt_audio is not None:
            pmt_wav, vocal_wav, bgm_wav, raw_pmt_wav, raw_vocal_wav, raw_bgm_wav = self._prepare_audio_prompt(
                options.prompt_audio, options.prompt_audio_batch_index
            )
            return {
                "pmt_wav": pmt_wav,
                "vocal_wav": vocal_wav,
                "bgm_wav": bgm_wav,
                "melody_is_wav": False,
                "raw_pmt_wav": raw_pmt_wav,
                "raw_vocal_wav": raw_vocal_wav,
                "raw_bgm_wav": raw_bgm_wav,
            }

        auto_type = options.auto_prompt_audio_type
        if auto_type and auto_type != "None":
            if auto_type not in self.auto_prompt:
                raise ValueError(f"Unsupported auto_prompt_audio_type: {auto_type}")
            sg_generate = self._modules["sg_generate"]
            lang = sg_generate.check_language_by_text(options.lyrics)
            pool = self.auto_prompt[auto_type][lang]
            prompt_token = pool[np.random.randint(0, len(pool))]
            return {
                "pmt_wav": prompt_token[:, [0], :],
                "vocal_wav": prompt_token[:, [1], :],
                "bgm_wav": prompt_token[:, [2], :],
                "melody_is_wav": False,
            }

        return {"pmt_wav": None, "vocal_wav": None, "bgm_wav": None, "melody_is_wav": True}

    def generate(self, options: GenerationOptions) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("SongGeneration model has been released. Run the loader node again.")
        with _RUNTIME_LOCK:
            _check_interrupted()
            seed = int(options.seed) if int(options.seed) >= 0 else int(time.time())
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.enabled = False

            args = self._make_args(options)
            gen_params = self._modules["sg_generate"].resolve_generation_params(
                args, self.cfg.max_dur, low_mem=False
            )
            self.model.set_generation_params(**gen_params)
            conditioning = self._prepare_conditioning(options)

            if self.version == "v1":
                descriptions = options.descriptions.lower() if options.descriptions else None
            elif options.generate_type == "bgm":
                prompt = options.descriptions.lower() if options.descriptions else "."
                descriptions = "[Musicality-very-high], [Pure-Music], " + prompt
            else:
                prompt = options.descriptions.lower() if options.descriptions else "."
                descriptions = "[Musicality-very-high], " + prompt

            generate_inp = {
                "lyrics": [options.lyrics.replace("  ", " ")] if options.generate_type != "bgm" else ".",
                "descriptions": [descriptions],
                "melody_wavs": conditioning["pmt_wav"],
                "vocal_wavs": conditioning["vocal_wav"],
                "bgm_wavs": conditioning["bgm_wav"],
                "melody_is_wav": conditioning["melody_is_wav"],
            }
            start_time = time.time()
            lm_progress = _ProgressBridge()
            decode_progress = _ProgressBridge()

            def _lm_progress(current: int, total: int) -> None:
                lm_progress.update(current, total, "LLM token 生成")

            def _decode_progress(current: int, total: int, label: str | None = None) -> None:
                decode_progress.update(current, total, label or "Diffusion 音频解码")

            progress_module = _get_comfy_progress_module()
            self.model.set_custom_progress_callback(_lm_progress)
            lm_success = False
            try:
                with progress_module.progress_hooks(progress_callback=_decode_progress, interrupt_callback=_check_interrupted):
                    with torch.autocast(
                        device_type="cuda",
                        dtype=self.llm_dtype,
                        enabled=self.llm_dtype in (torch.float16, torch.bfloat16),
                    ):
                        with torch.no_grad():
                            tokens = self.model.generate(**generate_inp, return_tokens=True)
                lm_success = True
            finally:
                self.model.set_custom_progress_callback(None)
                lm_progress.close(finish=lm_success)
            mid_time = time.time()

            decode_success = False
            try:
                with progress_module.progress_hooks(progress_callback=_decode_progress, interrupt_callback=_check_interrupted):
                    with torch.no_grad():
                        raw_args = ()
                        if "raw_pmt_wav" in conditioning and options.generate_type == "mixed":
                            raw_args = (
                                conditioning["raw_pmt_wav"],
                                conditioning["raw_vocal_wav"],
                                conditioning["raw_bgm_wav"],
                            )
                        if options.generate_type == "separate":
                            mixed = self.model.generate_audio(
                                tokens, *raw_args, chunked=True, chunk_size=args.chunk_size, gen_type="mixed"
                            )
                            vocal = self.model.generate_audio(
                                tokens, *raw_args, chunked=True, chunk_size=args.chunk_size, gen_type="vocal"
                            )
                            bgm = self.model.generate_audio(
                                tokens, *raw_args, chunked=True, chunk_size=args.chunk_size, gen_type="bgm"
                            )
                        else:
                            mixed = self.model.generate_audio(
                                tokens, *raw_args, chunked=True, chunk_size=args.chunk_size, gen_type=options.generate_type
                            )
                            vocal = None
                            bgm = None
                decode_success = True
            finally:
                decode_progress.close(finish=decode_success)
            end_time = time.time()

            metadata = {
                "model": self.model_dir.name,
                "version": self.version,
                "seed": seed,
                "generate_type": options.generate_type,
                "sample_rate": self.sample_rate,
                "lm_seconds": round(mid_time - start_time, 3),
                "diffusion_seconds": round(end_time - mid_time, 3),
            }
            return {"mixed": mixed, "vocal": vocal, "bgm": bgm, "metadata": metadata}


def _load_model(
    model_name: str,
    version: str,
    runtime_root: str,
    gpu_id: int,
    use_flash_attn: bool,
    reload_model: bool,
    segmented_load: bool,
    quantization: str,
    quantization_target: str,
    rebuild_quantization_cache: bool,
    llm_precision: str,
    diffusion_precision: str,
    vae_precision: str,
) -> SongGenerationModelHandle:
    model_dir = _resolve_model_dir(model_name)
    runtime_roots = _runtime_roots(model_dir, runtime_root)
    resolved_version = _infer_version(model_dir, version)
    llm_dtype = _dtype_from_choice(llm_precision)
    diffusion_dtype = _dtype_from_choice(diffusion_precision)
    vae_dtype = _dtype_from_choice(vae_precision)
    normalized_quantization = _normalize_quantization_mode(quantization) or "none"
    key = (
        str(model_dir.resolve()),
        tuple(str(root.resolve()) for root in runtime_roots if root.exists()),
        resolved_version,
        int(gpu_id),
        bool(use_flash_attn),
        bool(segmented_load),
        normalized_quantization,
        quantization_target,
        bool(rebuild_quantization_cache),
        str(llm_dtype),
        str(diffusion_dtype),
        str(vae_dtype),
    )
    if reload_model and key in _MODEL_CACHE:
        _MODEL_CACHE[key].release(clear_cuda_cache=True)
    cached = _MODEL_CACHE.get(key)
    if cached is not None and cached.loaded:
        return cached
    handle = SongGenerationModelHandle(
        cache_key=key,
        model_dir=model_dir,
        runtime_roots=runtime_roots,
        version=resolved_version,
        gpu_id=None if int(gpu_id) < 0 else int(gpu_id),
        use_flash_attn=bool(use_flash_attn),
        segmented_load=bool(segmented_load),
        quantization=normalized_quantization,
        quantization_target=quantization_target,
        rebuild_quantization_cache=bool(rebuild_quantization_cache),
        llm_dtype=llm_dtype,
        diffusion_dtype=diffusion_dtype,
        vae_dtype=vae_dtype,
    )
    _MODEL_CACHE[key] = handle
    return handle


def _base_inputs() -> dict[str, Any]:
    return {
        "songgen_model": (SONGGEN_MODEL_TYPE, _ui("模型", "SongGeneration 模型加载节点输出。")),
        "lyrics": (
            "STRING",
            _ui("歌词", "SongGeneration 段落格式歌词，例如 [verse] ... ; [chorus] ...。", multiline=True),
        ),
        "descriptions": (
            "STRING",
            _ui("描述", "风格、情绪、乐器、人声等逗号分隔提示词。", multiline=True, default="female, pop, energetic, piano, drum kit"),
        ),
        "seed": ("INT", _ui("种子", "-1 使用当前时间。", default=-1, min=-1, max=2**31 - 1)),
        "duration": ("FLOAT", _ui("时长", "0 使用模型 config.yaml 的 max_dur。", default=0.0, min=0.0, max=270.0, step=1.0)),
        "extend_stride": ("FLOAT", _ui("扩展步长", "长音频生成步长，通常保持 5。", default=5.0, min=1.0, max=60.0, step=1.0)),
        "temperature": ("FLOAT", _ui("温度", "0 使用原推理默认值。", default=0.0, min=0.0, max=2.0, step=0.05)),
        "cfg_coef": ("FLOAT", _ui("CFG", "Classifier-free guidance 系数。", default=1.5, min=0.0, max=10.0, step=0.1)),
        "top_k": ("INT", _ui("Top K", "0 使用原推理默认值。", default=0, min=0, max=10000)),
        "top_p": ("FLOAT", _ui("Top P", "0 关闭 top-p。", default=0.0, min=0.0, max=1.0, step=0.01)),
        "use_sampling": ("BOOLEAN", _ui("采样", "关闭后使用 greedy decoding。", default=True)),
        "record_tokens": ("BOOLEAN", _ui("记录 Tokens", "保持与原推理脚本一致。", default=True)),
        "record_window": ("INT", _ui("Token 窗口", "Token recording window。", default=50, min=1, max=1000)),
        "chunk_size": ("INT", _ui("解码块大小", "Diffusion decoding chunk size。", default=128, min=16, max=1024, step=16)),
    }


def _options_from_kwargs(generate_type: str, auto_prompt_audio_type: str, prompt_audio: Any, kwargs: dict[str, Any]) -> GenerationOptions:
    return GenerationOptions(
        lyrics=kwargs["lyrics"],
        descriptions=kwargs.get("descriptions") or "",
        generate_type=generate_type,
        auto_prompt_audio_type=auto_prompt_audio_type,
        prompt_audio=prompt_audio,
        prompt_audio_batch_index=int(kwargs.get("prompt_audio_batch_index", 0)),
        seed=int(kwargs.get("seed", -1)),
        duration=float(kwargs.get("duration", 0.0)),
        extend_stride=float(kwargs.get("extend_stride", 5.0)),
        temperature=float(kwargs.get("temperature", 0.0)),
        cfg_coef=float(kwargs.get("cfg_coef", 1.5)),
        top_k=int(kwargs.get("top_k", 0)),
        top_p=float(kwargs.get("top_p", 0.0)),
        use_sampling=bool(kwargs.get("use_sampling", True)),
        record_tokens=bool(kwargs.get("record_tokens", True)),
        record_window=int(kwargs.get("record_window", 50)),
        chunk_size=int(kwargs.get("chunk_size", 128)),
    )


class SongGenerationLoadModel:
    CATEGORY = CATEGORY
    RETURN_TYPES = (SONGGEN_MODEL_TYPE, "STRING")
    RETURN_NAMES = _tr_names("return_names.EasySongGenerationLoadModel", ("songgen_model", "info"))
    FUNCTION = "load"
    DESCRIPTION = _tr_text(
        "descriptions.EasySongGenerationLoadModel",
        "Load a SongGeneration checkpoint from ComfyUI/models/SongGeneration.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_model_choices(), _ui("模型目录", "包含 config.yaml 和 model.pt 的模型子文件夹。")),
                "version": (["auto", "v2", "v1"], _ui("版本", "auto 会根据模型目录名推断 v1/v2。")),
                "runtime_root": (
                    "STRING",
                    _ui("运行时根目录", "auto 会搜索模型目录、ComfyUI/models/SongGeneration 和插件目录。", default="auto"),
                ),
                "gpu_id": ("INT", _ui("GPU ID", "-1 使用当前 CUDA 设备。", default=-1, min=-1, max=16)),
                "use_flash_attn": ("BOOLEAN", _ui("Flash Attention", "环境支持时可开启。", default=True)),
                "segmented_load": ("BOOLEAN", _ui("分段加载", "按模块分段加载/移动权重，减少加载时显存峰值。", default=True)),
                "quantization": (
                    _QUANTIZATION_CHOICES,
                    _ui("量化格式", "Linear 权重量化格式；none 表示不量化。缓存位于 ComfyUI/models/SongGeneration-cache。"),
                ),
                "quantization_target": (
                    _QUANTIZATION_TARGETS,
                    _ui("量化范围", "选择要量化的模块。VAE 量化可能影响音质或兼容性。"),
                ),
                "rebuild_quantization_cache": (
                    "BOOLEAN",
                    _ui("重建量化缓存", "忽略已有量化缓存并重新生成。", default=False),
                ),
                "llm_precision": (_DTYPE_CHOICES, _ui("LLM 精度", "LLM 推理/权重计算精度。", default="float16")),
                "diffusion_precision": (_DTYPE_CHOICES, _ui("Diffusion 精度", "音频 Diffusion 解码模型计算精度。", default="float16")),
                "vae_precision": (_DTYPE_CHOICES, _ui("VAE 精度", "音频 VAE 编解码计算精度。", default="float32")),
                "reload_model": ("BOOLEAN", _ui("重新加载", "忽略缓存并重新加载权重。", default=False)),
            }
        }

    def load(
        self,
        model,
        version,
        runtime_root,
        gpu_id,
        use_flash_attn,
        segmented_load,
        quantization,
        quantization_target,
        rebuild_quantization_cache,
        llm_precision,
        diffusion_precision,
        vae_precision,
        reload_model,
    ):
        handle = _load_model(
            model,
            version,
            runtime_root,
            gpu_id,
            use_flash_attn,
            reload_model,
            segmented_load,
            quantization,
            quantization_target,
            rebuild_quantization_cache,
            llm_precision,
            diffusion_precision,
            vae_precision,
        )
        info = {
            "model": handle.model_dir.name,
            "path": str(handle.model_dir),
            "version": handle.version,
            "sample_rate": handle.sample_rate,
            "segmented_load": handle.segmented_load,
            "quantization": handle.quantization,
            "quantization_target": handle.quantization_target,
            "cache_dir": str(_songgen_cache_root()),
            "llm_precision": str(handle.llm_dtype).replace("torch.", ""),
            "diffusion_precision": str(handle.diffusion_dtype).replace("torch.", ""),
            "vae_precision": str(handle.vae_dtype).replace("torch.", ""),
            "quantization_info": handle.quantization_info,
        }
        return (handle, json.dumps(info, ensure_ascii=False, indent=2))


class SongGenerationReleaseModel:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = _tr_names("return_names.EasySongGenerationReleaseModel", ("status",))
    FUNCTION = "release"
    DESCRIPTION = _tr_text(
        "descriptions.EasySongGenerationReleaseModel",
        "Release a loaded SongGeneration model and clear CUDA cache.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "songgen_model": (SONGGEN_MODEL_TYPE, _ui("模型", "要释放的 SongGeneration 模型。")),
                "clear_cuda_cache": ("BOOLEAN", _ui("清理显存缓存", "释放后调用 torch.cuda.empty_cache。", default=True)),
            }
        }

    def release(self, songgen_model, clear_cuda_cache):
        return (songgen_model.release(clear_cuda_cache=bool(clear_cuda_cache)),)


class _GenerateOneBase:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "metadata")
    FUNCTION = "generate"
    GENERATE_TYPE = "mixed"
    DESCRIPTION = "Generate SongGeneration audio as a ComfyUI AUDIO object."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_base_inputs(),
                "auto_prompt_audio_type": (AUTO_PROMPT_TYPES, _ui("自动参考风格", "None 表示不使用自动参考音频。")),
            },
            "optional": {
                "prompt_audio": ("AUDIO", _ui("参考音频", "可选 ComfyUI AUDIO，会优先于自动参考风格。")),
                "prompt_audio_batch_index": (
                    "INT",
                    _ui("音频批次", "当 AUDIO 包含 batch 时选择其中一条。", default=0, min=0, max=4096),
                ),
            },
        }

    def generate(self, songgen_model, auto_prompt_audio_type="None", prompt_audio=None, **kwargs):
        options = _options_from_kwargs(self.GENERATE_TYPE, auto_prompt_audio_type, prompt_audio, kwargs)
        result = songgen_model.generate(options)
        metadata = json.dumps(result["metadata"], ensure_ascii=False, indent=2)
        return (_to_comfy_audio(result["mixed"][0], songgen_model.sample_rate), metadata)


class SongGenerationGenerateMixed(_GenerateOneBase):
    GENERATE_TYPE = "mixed"
    RETURN_NAMES = _tr_names("return_names.EasySongGenerationGenerateMixed", ("audio", "metadata"))
    DESCRIPTION = _tr_text(
        "descriptions.EasySongGenerationGenerateMixed",
        "Generate a full mixed song with vocals and accompaniment.",
    )


class SongGenerationGenerateVocal(_GenerateOneBase):
    GENERATE_TYPE = "vocal"
    RETURN_NAMES = _tr_names("return_names.EasySongGenerationGenerateVocal", ("audio", "metadata"))
    DESCRIPTION = _tr_text("descriptions.EasySongGenerationGenerateVocal", "Generate vocal-only audio.")


class SongGenerationGenerateBGM(_GenerateOneBase):
    GENERATE_TYPE = "bgm"
    RETURN_NAMES = _tr_names("return_names.EasySongGenerationGenerateBGM", ("audio", "metadata"))
    DESCRIPTION = _tr_text("descriptions.EasySongGenerationGenerateBGM", "Generate accompaniment / pure music audio.")


class SongGenerationGenerateSeparate:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = _tr_names("return_names.EasySongGenerationGenerateSeparate", ("mixed", "vocal", "bgm", "metadata"))
    FUNCTION = "generate"
    DESCRIPTION = _tr_text(
        "descriptions.EasySongGenerationGenerateSeparate",
        "Generate mixed, vocal-only, and accompaniment tracks.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_base_inputs(),
                "auto_prompt_audio_type": (AUTO_PROMPT_TYPES, _ui("自动参考风格", "None 表示不使用自动参考音频。")),
            },
            "optional": {
                "prompt_audio": ("AUDIO", _ui("参考音频", "可选 ComfyUI AUDIO，会优先于自动参考风格。")),
                "prompt_audio_batch_index": (
                    "INT",
                    _ui("音频批次", "当 AUDIO 包含 batch 时选择其中一条。", default=0, min=0, max=4096),
                ),
            },
        }

    def generate(self, songgen_model, auto_prompt_audio_type="None", prompt_audio=None, **kwargs):
        options = _options_from_kwargs("separate", auto_prompt_audio_type, prompt_audio, kwargs)
        result = songgen_model.generate(options)
        metadata = json.dumps(result["metadata"], ensure_ascii=False, indent=2)
        return (
            _to_comfy_audio(result["mixed"][0], songgen_model.sample_rate),
            _to_comfy_audio(result["vocal"][0], songgen_model.sample_rate),
            _to_comfy_audio(result["bgm"][0], songgen_model.sample_rate),
            metadata,
        )


NODE_CLASS_MAPPINGS = {
    "EasySongGenerationLoadModel": SongGenerationLoadModel,
    "EasySongGenerationReleaseModel": SongGenerationReleaseModel,
    "EasySongGenerationGenerateMixed": SongGenerationGenerateMixed,
    "EasySongGenerationGenerateVocal": SongGenerationGenerateVocal,
    "EasySongGenerationGenerateBGM": SongGenerationGenerateBGM,
    "EasySongGenerationGenerateSeparate": SongGenerationGenerateSeparate,
}

NODE_DISPLAY_NAME_MAPPINGS = _tr_mapping("node_display_names", {
    "EasySongGenerationLoadModel": "Easy SongGeneration - Load Model",
    "EasySongGenerationReleaseModel": "Easy SongGeneration - Release Model",
    "EasySongGenerationGenerateMixed": "Easy SongGeneration - Generate Mixed",
    "EasySongGenerationGenerateVocal": "Easy SongGeneration - Generate Vocal",
    "EasySongGenerationGenerateBGM": "Easy SongGeneration - Generate BGM",
    "EasySongGenerationGenerateSeparate": "Easy SongGeneration - Generate Separate",
})
