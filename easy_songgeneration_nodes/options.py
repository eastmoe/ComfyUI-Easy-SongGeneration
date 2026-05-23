from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


