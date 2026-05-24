from __future__ import annotations

import os
from typing import Any

import torch

from .config import _normalize_quantization_mode, _quantization_cache_path, _torch_load_weights
from .progress import _SongGenProgress, _check_interrupted

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

    for name, param in list(module.named_parameters(recurse=False)):
        if param is None:
            continue
        module._parameters[name] = torch.nn.Parameter(
            param.detach().to(device=device, dtype=dtype),
            requires_grad=param.requires_grad,
        )
    for name, buffer in list(module.named_buffers(recurse=False)):
        if buffer is None:
            continue
        module._buffers[name] = buffer.detach().to(device=device, dtype=dtype)
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


