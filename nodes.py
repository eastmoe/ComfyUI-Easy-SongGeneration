from __future__ import annotations

import gc
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
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from comfy import model_management
except ImportError:
    model_management = None


PLUGIN_DIR = Path(__file__).resolve().parent
SONGGEN_DIR = PLUGIN_DIR / "songgeneration"
CATEGORY = "eastmoe/Comfy-Easy-SongGeneration"
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


def _ui(display_name: str, tooltip: str, **extra: Any) -> dict[str, Any]:
    extra["display_name"] = display_name
    extra["tooltip"] = tooltip
    return extra


def _comfy_models_dir() -> Path:
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        return Path(folder_paths.models_dir)
    return PLUGIN_DIR / "models"


def _songgen_model_root() -> Path:
    root = _comfy_models_dir() / "SongGeneration"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _register_model_folder() -> None:
    if folder_paths is None:
        return
    folder_paths.add_model_folder_path("songgeneration", str(_songgen_model_root()), is_default=True)


_register_model_folder()


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
    ) -> None:
        self.cache_key = cache_key
        self.model_dir = model_dir
        self.runtime_roots = runtime_roots
        self.version = version
        self.gpu_id = gpu_id
        self.use_flash_attn = use_flash_attn
        self.model = None
        self.cfg = None
        self.sample_rate = 48000
        self.auto_prompt = None
        self._modules: dict[str, Any] = {}
        self._load()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        _add_songgeneration_paths(self.runtime_roots)

        import generate as sg_generate
        from codeclm.models import CodecLM, builders

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. SongGeneration inference requires a CUDA GPU.")
        if self.gpu_id is not None and self.gpu_id >= 0:
            torch.cuda.set_device(int(self.gpu_id))

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

        auto_prompt_path = _resolve_runtime_file("tools/new_auto_prompt.pt", self.runtime_roots)
        self.auto_prompt = torch.load(auto_prompt_path, map_location="cpu")

        seperate_tokenizer = None
        if "audio_tokenizer_checkpoint_sep" in cfg.keys():
            seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
            seperate_tokenizer = seperate_tokenizer.eval().cuda()

        audiolm = builders.get_lm_model(cfg, version=self.version)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        audiolm_state_dict = {k.replace("audiolm.", ""): v for k, v in checkpoint.items() if k.startswith("audiolm")}
        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        audiolm = audiolm.eval().cuda().to(torch.float16)

        self.model = CodecLM(
            name="ComfyUI-SongGeneration",
            lm=audiolm,
            audiotokenizer=None,
            max_duration=cfg.max_dur,
            seperate_tokenizer=seperate_tokenizer,
        )
        self._modules = {"sg_generate": sg_generate, "builders": builders}

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
            audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
            audio_tokenizer = audio_tokenizer.eval().cuda()
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
            if model_management is not None and hasattr(model_management, "throw_exception_if_processing_interrupted"):
                model_management.throw_exception_if_processing_interrupted()
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    tokens = self.model.generate(**generate_inp, return_tokens=True)
            mid_time = time.time()

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
) -> SongGenerationModelHandle:
    model_dir = _resolve_model_dir(model_name)
    runtime_roots = _runtime_roots(model_dir, runtime_root)
    resolved_version = _infer_version(model_dir, version)
    key = (
        str(model_dir.resolve()),
        tuple(str(root.resolve()) for root in runtime_roots if root.exists()),
        resolved_version,
        int(gpu_id),
        bool(use_flash_attn),
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
    RETURN_NAMES = ("songgen_model", "info")
    FUNCTION = "load"
    DESCRIPTION = "Load a SongGeneration checkpoint from ComfyUI/models/SongGeneration."

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
                "reload_model": ("BOOLEAN", _ui("重新加载", "忽略缓存并重新加载权重。", default=False)),
            }
        }

    def load(self, model, version, runtime_root, gpu_id, use_flash_attn, reload_model):
        handle = _load_model(model, version, runtime_root, gpu_id, use_flash_attn, reload_model)
        info = {
            "model": handle.model_dir.name,
            "path": str(handle.model_dir),
            "version": handle.version,
            "sample_rate": handle.sample_rate,
        }
        return (handle, json.dumps(info, ensure_ascii=False, indent=2))


class SongGenerationReleaseModel:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "release"
    DESCRIPTION = "Release a loaded SongGeneration model and clear CUDA cache."

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
    DESCRIPTION = "Generate a full mixed song with vocals and accompaniment."


class SongGenerationGenerateVocal(_GenerateOneBase):
    GENERATE_TYPE = "vocal"
    DESCRIPTION = "Generate vocal-only audio."


class SongGenerationGenerateBGM(_GenerateOneBase):
    GENERATE_TYPE = "bgm"
    DESCRIPTION = "Generate accompaniment / pure music audio."


class SongGenerationGenerateSeparate:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("mixed", "vocal", "bgm", "metadata")
    FUNCTION = "generate"
    DESCRIPTION = "Generate mixed, vocal-only, and accompaniment tracks."

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

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasySongGenerationLoadModel": "Easy SongGeneration - Load Model",
    "EasySongGenerationReleaseModel": "Easy SongGeneration - Release Model",
    "EasySongGenerationGenerateMixed": "Easy SongGeneration - Generate Mixed",
    "EasySongGenerationGenerateVocal": "Easy SongGeneration - Generate Vocal",
    "EasySongGenerationGenerateBGM": "Easy SongGeneration - Generate BGM",
    "EasySongGenerationGenerateSeparate": "Easy SongGeneration - Generate Separate",
}
