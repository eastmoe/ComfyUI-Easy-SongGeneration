from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .config import (
    _MODEL_CACHE,
    _RUNTIME_LOCK,
    _dtype_from_choice,
    _normalize_quantization_mode,
    _path_signature,
    _quantization_cache_path,
    _torch_load_weights,
)
from .options import GenerationOptions
from .progress import _ProgressBridge, _SongGenProgress, _check_interrupted, _get_comfy_progress_module, model_management
from .quantization import (
    _apply_quantization_cache,
    _load_prefixed_state_dict_segmented,
    _load_state_dict_assign,
    _move_module_segmented,
    _replace_quantized_modules_from_names,
    _unpack_quantized_cache_state_dict,
)
from .runtime import (
    _add_songgeneration_paths,
    _infer_version,
    _register_resolvers,
    _resolve_existing_path,
    _resolve_prefixed_existing_path,
    _resolve_model_dir,
    _resolve_runtime_file,
    _runtime_roots,
    _write_temp_wav,
)

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
            cfg.audio_tokenizer_checkpoint = _resolve_prefixed_existing_path(
                str(cfg.audio_tokenizer_checkpoint), self.runtime_roots
            )
            if "audio_tokenizer_checkpoint_sep" in cfg.keys():
                cfg.audio_tokenizer_checkpoint_sep = _resolve_prefixed_existing_path(
                    str(cfg.audio_tokenizer_checkpoint_sep), self.runtime_roots
                )
            if "vae_config" in cfg.keys():
                cfg.vae_config = _resolve_existing_path(str(cfg.vae_config), self.runtime_roots)
            if "vae_model" in cfg.keys():
                cfg.vae_model = _resolve_existing_path(str(cfg.vae_model), self.runtime_roots)
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
            if not isinstance(self.auto_prompt, dict) or auto_type not in self.auto_prompt:
                raise ValueError(f"Unsupported auto_prompt_audio_type: {auto_type}")
            sg_generate = self._modules["sg_generate"]
            lang = sg_generate.check_language_by_text(options.lyrics)
            prompt_group = self.auto_prompt[auto_type]
            if not isinstance(prompt_group, dict):
                raise ValueError(f"Invalid auto prompt payload for type: {auto_type}")
            if lang not in prompt_group:
                lang = "en" if "en" in prompt_group else next(iter(prompt_group), "")
            pool = prompt_group.get(lang) or []
            if not pool:
                raise ValueError(f"No auto prompt tokens available for type={auto_type!r}, language={lang!r}.")
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


