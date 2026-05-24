from __future__ import annotations

import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from .config import SONGGEN_DIR, V1_MODEL_NAMES, _songgen_model_root

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
        model_dir = path
    else:
        model_dir = _songgen_model_root() / name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"SongGeneration model directory does not exist: {model_dir}")
    missing = [filename for filename in ("config.yaml", "model.pt") if not (model_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"SongGeneration model directory is missing {', '.join(missing)}: {model_dir}")
    return model_dir


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
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return str(path)
    candidates = [path]
    parts = path.parts
    if parts:
        first = parts[0]
        if first in {".", ""} and len(parts) > 1:
            first = parts[1]
            rest = parts[2:]
            prefix = (parts[0],)
        else:
            rest = parts[1:]
            prefix = ()
        aliases = {"ckpt": "common", "common": "ckpt"}
        alias = aliases.get(first)
        if alias:
            candidates.append(Path(*prefix, alias, *rest))
    for root in roots:
        for candidate_path in candidates:
            candidate = root / candidate_path
            if candidate.exists():
                return str(candidate)
    return str(path)


def _resolve_prefixed_existing_path(value: str | None, roots: list[Path]) -> str | None:
    if not value:
        return value
    text = str(value).strip()
    prefixes = ("Flow1dVAE1rvq_", "Flow1dVAESeparate_")
    for prefix in prefixes:
        if text.startswith(prefix):
            resolved = _resolve_existing_path(text[len(prefix):], roots)
            return f"{prefix}{resolved}" if resolved else value
    return _resolve_existing_path(text, roots)


def _resolve_config_token_paths(value: Any, roots: list[Path]) -> None:
    if isinstance(value, list):
        for item in value:
            _resolve_config_token_paths(item, roots)
        return

    items = getattr(value, "items", None)
    if items is None:
        return

    for key, item in list(items()):
        if key == "token_path" and isinstance(item, str):
            value[key] = _resolve_existing_path(item, roots)
        else:
            _resolve_config_token_paths(item, roots)


def _resolve_runtime_file(relative_path: str, roots: list[Path]) -> str:
    searched = []
    for root in roots:
        candidate = root / relative_path
        searched.append(candidate)
        if candidate.is_file():
            return str(candidate)
    candidate = SONGGEN_DIR / relative_path
    searched.append(candidate)
    if candidate.is_file():
        return str(candidate)
    searched_text = "\n  - ".join(str(path) for path in searched)
    raise FileNotFoundError(f"Required SongGeneration runtime file was not found: {relative_path}\nSearched:\n  - {searched_text}")


def _register_resolvers(model_dir: Path, runtime_roots: list[Path]):
    from omegaconf import OmegaConf

    def register(name: str, resolver) -> None:
        try:
            OmegaConf.register_new_resolver(name, resolver, replace=True)
        except TypeError:
            OmegaConf.register_new_resolver(name, resolver)

    def load_yaml(path: str):
        return list(OmegaConf.load(_resolve_runtime_file(str(path), [model_dir, *runtime_roots])))

    register("eval", lambda x: eval(x))
    register("concat", lambda *x: [xxx for xx in x for xxx in xx])
    register("get_fname", lambda: model_dir.name)
    register("load_yaml", load_yaml)
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
    if waveform.numel() == 0:
        raise ValueError("AUDIO waveform is empty.")
    if not torch.isfinite(waveform).all():
        raise ValueError("AUDIO waveform contains NaN or Inf values.")
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


