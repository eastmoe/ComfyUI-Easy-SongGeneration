from __future__ import annotations

import json
from typing import Any

from .config import (
    AUTO_PROMPT_TYPES,
    CATEGORY,
    _DIFFUSION_DTYPE_CHOICES,
    SONGGEN_MODEL_TYPE,
    _DTYPE_CHOICES,
    _QUANTIZATION_CHOICES,
    _QUANTIZATION_TARGETS,
    _songgen_cache_root,
    _tr_mapping,
    _tr_names,
    _tr_text,
    _ui_text,
)
from .downloader import DOWNLOAD_SOURCES, SONGGEN_DOWNLOAD_CHOICES, download_songgeneration_assets
from .formatter import LANGUAGE_CHOICES, SECTION_CHOICES, format_lyrics_text, format_style_text
from .model import _load_model
from .options import GenerationOptions
from .runtime import _model_choices, _to_comfy_audio

def _base_inputs() -> dict[str, Any]:
    return {
        "songgen_model": (SONGGEN_MODEL_TYPE, _ui_text("ui.common.songgen_model", "模型", "SongGeneration 模型加载节点输出。")),
        "lyrics": (
            "STRING",
            _ui_text("ui.common.lyrics", "歌词", "SongGeneration 段落格式歌词，例如 [verse] ... ; [chorus] ...。", multiline=True),
        ),
        "descriptions": (
            "STRING",
            _ui_text("ui.common.descriptions", "描述", "风格、情绪、乐器、人声等逗号分隔提示词。", multiline=True, default="female, pop, energetic, piano, drum kit"),
        ),
        "seed": ("INT", _ui_text("ui.common.seed", "种子", "-1 使用当前时间。", default=-1, min=-1, max=2**31 - 1)),
        "duration": ("FLOAT", _ui_text("ui.common.duration", "时长", "0 使用模型 config.yaml 的 max_dur。", default=0.0, min=0.0, max=270.0, step=1.0)),
        "extend_stride": ("FLOAT", _ui_text("ui.common.extend_stride", "扩展步长", "长音频生成步长，通常保持 5。", default=5.0, min=1.0, max=60.0, step=1.0)),
        "temperature": ("FLOAT", _ui_text("ui.common.temperature", "温度", "0 使用原推理默认值。", default=0.0, min=0.0, max=2.0, step=0.05)),
        "cfg_coef": ("FLOAT", _ui_text("ui.common.cfg_coef", "CFG", "Classifier-free guidance 系数。", default=1.5, min=0.0, max=10.0, step=0.1)),
        "top_k": ("INT", _ui_text("ui.common.top_k", "Top K", "0 使用原推理默认值。", default=0, min=0, max=10000)),
        "top_p": ("FLOAT", _ui_text("ui.common.top_p", "Top P", "0 关闭 top-p。", default=0.0, min=0.0, max=1.0, step=0.01)),
        "use_sampling": ("BOOLEAN", _ui_text("ui.common.use_sampling", "采样", "关闭后使用 greedy decoding。", default=True)),
        "record_tokens": ("BOOLEAN", _ui_text("ui.common.record_tokens", "记录 Tokens", "保持与原推理脚本一致。", default=True)),
        "record_window": ("INT", _ui_text("ui.common.record_window", "Token 窗口", "Token recording window。", default=50, min=1, max=1000)),
        "chunk_size": ("INT", _ui_text("ui.common.chunk_size", "解码块大小", "Diffusion decoding chunk size。", default=128, min=16, max=1024, step=16)),
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
                "model": (_model_choices(), _ui_text("ui.load_model.model", "模型目录", "包含 config.yaml 和 model.pt 的模型子文件夹。")),
                "version": (["auto", "v2", "v1"], _ui_text("ui.load_model.version", "版本", "auto 会根据模型目录名推断 v1/v2。")),
                "runtime_root": (
                    "STRING",
                    _ui_text("ui.load_model.runtime_root", "运行时根目录", "auto 会搜索模型目录、ComfyUI/models/SongGeneration 和插件目录。", default="auto"),
                ),
                "gpu_id": ("INT", _ui_text("ui.load_model.gpu_id", "GPU ID", "-1 使用当前 CUDA 设备。", default=-1, min=-1, max=16)),
                "use_flash_attn": ("BOOLEAN", _ui_text("ui.load_model.use_flash_attn", "Flash Attention", "环境支持时可开启。", default=True)),
                "segmented_load": ("BOOLEAN", _ui_text("ui.load_model.segmented_load", "分段加载", "按模块分段加载/移动权重，减少加载时显存峰值。", default=True)),
                "quantization": (
                    _QUANTIZATION_CHOICES,
                    _ui_text("ui.load_model.quantization", "量化格式", "Linear 权重量化格式；none 表示不量化。缓存位于 ComfyUI/models/SongGeneration-cache。"),
                ),
                "quantization_target": (
                    _QUANTIZATION_TARGETS,
                    _ui_text("ui.load_model.quantization_target", "量化范围", "选择要量化的模块。VAE 量化可能影响音质或兼容性。"),
                ),
                "rebuild_quantization_cache": (
                    "BOOLEAN",
                    _ui_text("ui.load_model.rebuild_quantization_cache", "重建量化缓存", "忽略已有量化缓存并重新生成。", default=False),
                ),
                "llm_precision": (_DTYPE_CHOICES, _ui_text("ui.load_model.llm_precision", "LLM 精度", "LLM 推理/权重计算精度。", default="float16")),
                "diffusion_precision": (_DIFFUSION_DTYPE_CHOICES, _ui_text("ui.load_model.diffusion_precision", "Diffusion 精度", "音频 Diffusion 解码模型计算精度。", default="bfloat16")),
                "vae_precision": (_DTYPE_CHOICES, _ui_text("ui.load_model.vae_precision", "VAE 精度", "音频 VAE 编解码计算精度。", default="float32")),
                "reload_model": ("BOOLEAN", _ui_text("ui.load_model.reload_model", "重新加载", "忽略缓存并重新加载权重。", default=False)),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
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
        unique_id=None,
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
            owner_id=unique_id,
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


class SongGenerationDownloadModel:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = _tr_names("return_names.EasySongGenerationDownloadModel", ("status",))
    FUNCTION = "download"
    OUTPUT_NODE = True
    DESCRIPTION = _tr_text(
        "descriptions.EasySongGenerationDownloadModel",
        "Download SongGeneration runtime folders and selected model checkpoints to ComfyUI/models/SongGeneration.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (DOWNLOAD_SOURCES, _ui_text("ui.download_model.source", "下载源", "可选 hf-mirror.com 或 huggingface.co。")),
                "model": (
                    SONGGEN_DOWNLOAD_CHOICES,
                    _ui_text("ui.download_model.model", "模型目录", "common 和 third_party 始终下载；这里选择额外下载的 SongGeneration 模型。"),
                ),
                "revision": ("STRING", _ui_text("ui.download_model.revision", "分支/版本", "Hugging Face revision，通常保持 main。", default="main")),
                "overwrite_existing": ("BOOLEAN", _ui_text("ui.download_model.overwrite_existing", "覆盖已有文件", "关闭时会跳过大小一致的本地文件。", default=False)),
                "download_threads": ("INT", _ui_text("ui.download_model.download_threads", "下载线程", "huggingface_hub 并发下载线程数。", default=8, min=1, max=32)),
                "disable_ssl_verify": (
                    "BOOLEAN",
                    _ui_text("ui.download_model.disable_ssl_verify", "关闭 SSL 验证", "仅在证书链异常或代理拦截 HTTPS 时开启。", default=False),
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def download(self, source, model, revision, overwrite_existing, download_threads=8, disable_ssl_verify=False):
        status = download_songgeneration_assets(
            source=source,
            model_choice=model,
            revision=revision,
            overwrite_existing=bool(overwrite_existing),
            download_threads=int(download_threads),
            disable_ssl_verify=bool(disable_ssl_verify),
        )
        return (status,)


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
                "songgen_model": (SONGGEN_MODEL_TYPE, _ui_text("ui.release_model.songgen_model", "模型", "要释放的 SongGeneration 模型。")),
                "clear_cuda_cache": ("BOOLEAN", _ui_text("ui.release_model.clear_cuda_cache", "清理显存缓存", "释放后调用 torch.cuda.empty_cache。", default=True)),
            }
        }

    def release(self, songgen_model, clear_cuda_cache):
        return (songgen_model.release(clear_cuda_cache=bool(clear_cuda_cache)),)


class SongGenerationLyricsStyleFormatter:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = _tr_names("return_names.EasySongGenerationLyricsStyleFormatter", ("lyrics", "style"))
    FUNCTION = "format"
    DESCRIPTION = _tr_text(
        "descriptions.EasySongGenerationLyricsStyleFormatter",
        "Format SongGeneration lyrics and comma-separated style tags, with UI shortcuts for section and style tags.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": (
                    "STRING",
                    _ui_text(
                        "ui.formatter.lyrics",
                        "歌词编辑器",
                        "支持 [intro-short]、[verse]、[chorus] 等 SongGeneration 结构标签；节点会输出格式化后的纯文本歌词。",
                        multiline=True,
                        default="[intro-medium]\n[verse] \n[chorus] \n[inst-medium]\n[verse] \n[chorus] \n[outro-medium]",
                    ),
                ),
                "style": (
                    "STRING",
                    _ui_text(
                        "ui.formatter.style",
                        "风格标签",
                        "使用逗号分隔性别、曲风、情绪、乐器等标签；节点会输出格式化后的纯文本风格描述。",
                        multiline=True,
                        default="female, pop, energetic, piano, drum kit",
                    ),
                ),
                "lyric_language": (
                    LANGUAGE_CHOICES,
                    _ui_text("ui.formatter.lyric_language", "歌词语言", "auto 会根据是否包含中文决定英文末句是否补句点。"),
                ),
                "wrap_untagged_as": (
                    SECTION_CHOICES,
                    _ui_text(
                        "ui.formatter.wrap_untagged_as",
                        "无标签歌词包裹为",
                        "当歌词没有任何结构标签时，可自动包裹成 verse/chorus/bridge。",
                        default="verse",
                    ),
                ),
                "style_trailing_period": (
                    "BOOLEAN",
                    _ui_text("ui.formatter.style_trailing_period", "风格末尾句点", "按官方示例在风格标签末尾保留英文句点。", default=True),
                ),
            }
        }

    def format(self, lyrics, style, lyric_language="auto", wrap_untagged_as="verse", style_trailing_period=True):
        formatted_lyrics = format_lyrics_text(lyrics, lyric_language, wrap_untagged_as)
        formatted_style = format_style_text(style, bool(style_trailing_period))
        return (formatted_lyrics, formatted_style)


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
                "auto_prompt_audio_type": (AUTO_PROMPT_TYPES, _ui_text("ui.common.auto_prompt_audio_type", "自动参考风格", "None 表示不使用自动参考音频。")),
            },
            "optional": {
                "prompt_audio": ("AUDIO", _ui_text("ui.common.prompt_audio", "参考音频", "可选 ComfyUI AUDIO，会优先于自动参考风格。")),
                "prompt_audio_batch_index": (
                    "INT",
                    _ui_text("ui.common.prompt_audio_batch_index", "音频批次", "当 AUDIO 包含 batch 时选择其中一条。", default=0, min=0, max=4096),
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
                "auto_prompt_audio_type": (AUTO_PROMPT_TYPES, _ui_text("ui.common.auto_prompt_audio_type", "自动参考风格", "None 表示不使用自动参考音频。")),
            },
            "optional": {
                "prompt_audio": ("AUDIO", _ui_text("ui.common.prompt_audio", "参考音频", "可选 ComfyUI AUDIO，会优先于自动参考风格。")),
                "prompt_audio_batch_index": (
                    "INT",
                    _ui_text("ui.common.prompt_audio_batch_index", "音频批次", "当 AUDIO 包含 batch 时选择其中一条。", default=0, min=0, max=4096),
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
    "EasySongGenerationDownloadModel": SongGenerationDownloadModel,
    "EasySongGenerationLoadModel": SongGenerationLoadModel,
    "EasySongGenerationReleaseModel": SongGenerationReleaseModel,
    "EasySongGenerationLyricsStyleFormatter": SongGenerationLyricsStyleFormatter,
    "EasySongGenerationGenerateMixed": SongGenerationGenerateMixed,
    "EasySongGenerationGenerateVocal": SongGenerationGenerateVocal,
    "EasySongGenerationGenerateBGM": SongGenerationGenerateBGM,
    "EasySongGenerationGenerateSeparate": SongGenerationGenerateSeparate,
}

NODE_DISPLAY_NAME_MAPPINGS = _tr_mapping("node_display_names", {
    "EasySongGenerationDownloadModel": "Easy SongGeneration - Download Model",
    "EasySongGenerationLoadModel": "Easy SongGeneration - Load Model",
    "EasySongGenerationReleaseModel": "Easy SongGeneration - Release Model",
    "EasySongGenerationLyricsStyleFormatter": "Easy SongGeneration - Lyrics & Style Formatter",
    "EasySongGenerationGenerateMixed": "Easy SongGeneration - Generate Mixed",
    "EasySongGenerationGenerateVocal": "Easy SongGeneration - Generate Vocal",
    "EasySongGenerationGenerateBGM": "Easy SongGeneration - Generate BGM",
    "EasySongGenerationGenerateSeparate": "Easy SongGeneration - Generate Separate",
})
