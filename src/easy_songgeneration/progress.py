from __future__ import annotations

import sys
import time

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

try:
    from comfy import model_management
except ImportError:
    model_management = None

try:
    from comfy.utils import ProgressBar as ComfyProgressBar
except ImportError:
    ComfyProgressBar = None

from .config import SONGGEN_DIR

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


