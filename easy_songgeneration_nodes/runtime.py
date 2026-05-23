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


