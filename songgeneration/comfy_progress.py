from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Callable, Iterator


ProgressCallback = Callable[[int, int, str | None], None]
InterruptCallback = Callable[[], None]

_STATE = threading.local()


def _progress_stack() -> list[ProgressCallback]:
    stack = getattr(_STATE, "progress_stack", None)
    if stack is None:
        stack = []
        _STATE.progress_stack = stack
    return stack


def _interrupt_stack() -> list[InterruptCallback]:
    stack = getattr(_STATE, "interrupt_stack", None)
    if stack is None:
        stack = []
        _STATE.interrupt_stack = stack
    return stack


@contextmanager
def progress_hooks(
    progress_callback: ProgressCallback | None = None,
    interrupt_callback: InterruptCallback | None = None,
) -> Iterator[None]:
    progress_stack = _progress_stack()
    interrupt_stack = _interrupt_stack()
    if progress_callback is not None:
        progress_stack.append(progress_callback)
    if interrupt_callback is not None:
        interrupt_stack.append(interrupt_callback)
    try:
        yield
    finally:
        if interrupt_callback is not None and interrupt_stack:
            interrupt_stack.pop()
        if progress_callback is not None and progress_stack:
            progress_stack.pop()


def report_progress(current: int, total: int, label: str | None = None) -> None:
    stack = _progress_stack()
    if stack:
        stack[-1](int(current), max(1, int(total)), label)


def check_interrupted() -> None:
    stack = _interrupt_stack()
    if stack:
        stack[-1]()
