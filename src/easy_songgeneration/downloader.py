from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path

from .config import _songgen_model_root
from .progress import _SongGenProgress

SONGGEN_REPO_ID = "eastmoe/SongGeneration"
DOWNLOAD_SOURCES = ["hf-mirror.com", "huggingface.co"]
REQUIRED_DOWNLOAD_DIRS = ("common", "third_party")
SONGGEN_DOWNLOAD_MODELS = (
    "SongGeneration-v2-large",
    "SongGeneration-base-full",
    "SongGeneration-base-new",
)
SONGGEN_DOWNLOAD_CHOICES = [*SONGGEN_DOWNLOAD_MODELS, "runtime-only", "all"]


def _endpoint(source: str) -> str:
    value = (source or DOWNLOAD_SOURCES[0]).strip().lower()
    if value in {"hf-mirror", "hf-mirror.com", "https://hf-mirror.com"}:
        return "https://hf-mirror.com"
    if value in {"huggingface", "huggingface.co", "https://huggingface.co"}:
        return "https://huggingface.co"
    raise ValueError(f"Unsupported download source: {source}")


def _selected_prefixes(model_choice: str) -> tuple[str, ...]:
    choice = (model_choice or SONGGEN_DOWNLOAD_MODELS[0]).strip()
    dirs = list(REQUIRED_DOWNLOAD_DIRS)
    if choice == "all":
        dirs.extend(SONGGEN_DOWNLOAD_MODELS)
    elif choice != "runtime-only":
        if choice not in SONGGEN_DOWNLOAD_MODELS:
            raise ValueError(f"Unsupported SongGeneration model choice: {model_choice}")
        dirs.append(choice)
    return tuple(f"{name.rstrip('/')}/" for name in dirs)


def _target_path(target_root: Path, relative: str) -> Path:
    target = target_root / relative
    root = target_root.resolve()
    resolved = target.resolve()
    if root != resolved and root not in resolved.parents:
        raise RuntimeError(f"Refusing to write outside SongGeneration model root: {relative}")
    return target


def _allow_patterns(model_choice: str) -> list[str]:
    return [f"{prefix}*" for prefix in _selected_prefixes(model_choice)]


def _snapshot_download(**kwargs):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for model downloads. "
            "ComfyUI's transformers dependency normally installs it."
        ) from exc

    parameters = inspect.signature(snapshot_download).parameters
    supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return snapshot_download(**supported_kwargs)


def _dry_run_supported() -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    return "dry_run" in inspect.signature(snapshot_download).parameters


def _cleanup_legacy_partials(target_root: Path, model_choice: str) -> int:
    removed = 0
    for prefix in _selected_prefixes(model_choice):
        directory = _target_path(target_root, prefix)
        if not directory.exists():
            continue
        for partial in directory.rglob("*.download"):
            _target_path(target_root, os.fspath(partial.relative_to(target_root)))
            partial.unlink(missing_ok=True)
            removed += 1
    return removed


@contextmanager
def _hf_ssl_context(disable_ssl_verify: bool):
    if not disable_ssl_verify:
        yield
        return

    try:
        import httpx
        from huggingface_hub import set_client_factory
        from huggingface_hub.utils._http import default_client_factory, hf_request_event_hook
    except Exception:
        pass
    else:

        def unverified_client_factory():
            return httpx.Client(
                verify=False,
                event_hooks={"request": [hf_request_event_hook]},
                follow_redirects=True,
                timeout=None,
            )

        set_client_factory(unverified_client_factory)
        try:
            yield
        finally:
            set_client_factory(default_client_factory)
        return

    try:
        import requests
        import urllib3
        try:
            from huggingface_hub import configure_http_backend
        except ImportError:
            from huggingface_hub.utils import configure_http_backend
    except Exception as exc:
        raise RuntimeError("This huggingface_hub version does not support disabling SSL verification.") from exc

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def unverified_backend_factory():
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=unverified_backend_factory)
    try:
        yield
    finally:
        configure_http_backend(backend_factory=lambda: requests.Session())


def download_songgeneration_assets(
    *,
    source: str,
    model_choice: str,
    revision: str,
    overwrite_existing: bool,
    download_threads: int = 8,
    disable_ssl_verify: bool = False,
) -> str:
    endpoint = _endpoint(source)
    revision = (revision or "main").strip() or "main"
    target_root = _songgen_model_root()
    target_root.mkdir(parents=True, exist_ok=True)
    patterns = _allow_patterns(model_choice)
    max_workers = max(1, int(download_threads or 1))
    removed_partials = _cleanup_legacy_partials(target_root, model_choice)

    dry_run_files = []
    downloaded = None
    skipped = None
    planned_bytes = None
    progress = _SongGenProgress(3, "准备 SongGeneration 模型下载")
    try:
        with _hf_ssl_context(bool(disable_ssl_verify)):
            if _dry_run_supported():
                dry_run_files = list(
                    _snapshot_download(
                        repo_id=SONGGEN_REPO_ID,
                        repo_type="model",
                        revision=revision,
                        local_dir=target_root,
                        endpoint=endpoint,
                        allow_patterns=patterns,
                        max_workers=max_workers,
                        force_download=bool(overwrite_existing),
                        user_agent="ComfyUI-Easy-SongGeneration",
                        dry_run=True,
                    )
                )
                if not dry_run_files:
                    raise RuntimeError(f"No matching files found in {SONGGEN_REPO_ID} for {model_choice}")
                downloaded = sum(1 for item in dry_run_files if getattr(item, "will_download", False))
                skipped = len(dry_run_files) - downloaded
                planned_bytes = sum(
                    int(getattr(item, "file_size", 0) or 0)
                    for item in dry_run_files
                    if getattr(item, "will_download", False)
                )
            progress.update_absolute(1, total=3, label="下载 SongGeneration 模型")
            _snapshot_download(
                repo_id=SONGGEN_REPO_ID,
                repo_type="model",
                revision=revision,
                local_dir=target_root,
                endpoint=endpoint,
                allow_patterns=patterns,
                max_workers=max_workers,
                force_download=bool(overwrite_existing),
                user_agent="ComfyUI-Easy-SongGeneration",
            )
            progress.update_absolute(2, total=3, label="校验 SongGeneration 模型下载")
        progress.finish("SongGeneration 模型下载完成")
    except Exception:
        progress.close()
        raise

    selected_dirs = [prefix.rstrip("/") for prefix in _selected_prefixes(model_choice)]
    info = {
        "repository": SONGGEN_REPO_ID,
        "source": endpoint,
        "revision": revision,
        "target": os.fspath(target_root),
        "directories": selected_dirs,
        "allow_patterns": patterns,
        "files": len(dry_run_files) if dry_run_files else None,
        "downloaded": downloaded,
        "skipped": skipped,
        "planned_bytes": planned_bytes,
        "download_threads": max_workers,
        "ssl_verification": not bool(disable_ssl_verify),
        "legacy_partial_files_removed": removed_partials,
    }
    return json.dumps(info, ensure_ascii=False, indent=2)
