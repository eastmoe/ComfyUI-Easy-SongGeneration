from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import shutil

from .config import _is_git_lfs_pointer, _songgen_model_root
from .progress import _SongGenProgress

SONGGEN_REPO_ID = "eastmoe/SongGeneration"
AUTO_PROMPT_REPO_ID = "tencent/SongGeneration"
AUTO_PROMPT_REPO_TYPE = "space"
AUTO_PROMPT_REVISION = "main"
AUTO_PROMPT_REMOTE_FILE = "tools/new_prompt.pt"
AUTO_PROMPT_LOCAL_FILES = ("tools/new_auto_prompt.pt", "tools/new_prompt.pt")
MIN_AUTO_PROMPT_BYTES = 1024
AUTO_PROMPT_ETAG_TIMEOUT_SECONDS = 10
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


def _hf_hub_download(**kwargs):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for model downloads. "
            "ComfyUI's transformers dependency normally installs it."
        ) from exc

    parameters = inspect.signature(hf_hub_download).parameters
    supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return hf_hub_download(**supported_kwargs)


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


def _dry_run_relative_path(item, target_root: Path) -> Path | None:
    local_path = getattr(item, "local_path", None)
    if local_path:
        path = Path(local_path)
        if path.is_absolute():
            try:
                return path.resolve().relative_to(target_root.resolve())
            except ValueError:
                pass
        else:
            return path

    repo_file = getattr(item, "repo_file", None)
    for owner in (item, repo_file):
        if owner is None:
            continue
        for attr in ("rfilename", "filename", "path"):
            value = getattr(owner, attr, None)
            if isinstance(value, str) and value:
                path = Path(value)
                if not path.is_absolute():
                    return path
    return None


def _dry_run_file_size(item) -> int | None:
    repo_file = getattr(item, "repo_file", None)
    for owner in (item, repo_file):
        if owner is None:
            continue
        for attr in ("file_size", "size"):
            value = getattr(owner, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    return None


def _remove_invalid_download_files(target_root: Path, dry_run_files: list) -> dict[str, int]:
    stats = {"lfs_pointers": 0, "size_mismatches": 0}
    for item in dry_run_files:
        relative = _dry_run_relative_path(item, target_root)
        if relative is None:
            continue
        path = _target_path(target_root, os.fspath(relative))
        if not path.is_file():
            continue
        expected_size = _dry_run_file_size(item)
        invalid_size = expected_size is not None and path.stat().st_size != expected_size
        if _is_git_lfs_pointer(path):
            path.unlink(missing_ok=True)
            stats["lfs_pointers"] += 1
        elif invalid_size:
            path.unlink(missing_ok=True)
            stats["size_mismatches"] += 1
    return stats


def _is_valid_weights_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size >= MIN_AUTO_PROMPT_BYTES
        and not _is_git_lfs_pointer(path)
    )


def _ensure_auto_prompt_asset(
    *,
    target_root: Path,
    endpoint: str,
    overwrite_existing: bool,
) -> dict[str, str]:
    canonical_path = _target_path(target_root, AUTO_PROMPT_LOCAL_FILES[0])
    if _is_valid_weights_file(canonical_path) and not overwrite_existing:
        return {"status": "skipped", "path": os.fspath(canonical_path)}

    fallback_path = _target_path(target_root, AUTO_PROMPT_LOCAL_FILES[1])
    if _is_valid_weights_file(fallback_path) and not overwrite_existing:
        if not _is_valid_weights_file(canonical_path):
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback_path, canonical_path)
            return {
                "status": "copied",
                "path": os.fspath(canonical_path),
                "source_path": os.fspath(fallback_path),
            }
        return {"status": "skipped", "path": os.fspath(canonical_path)}

    download_error = None
    downloaded_path = None
    downloaded_endpoint = None
    # Honor the source selected by the user. In particular, hf-mirror may redirect
    # Space files to huggingface.co; explicitly retrying the official endpoint after
    # that makes an unavailable optional asset stall the whole node for minutes.
    for candidate_endpoint in (endpoint,):
        try:
            downloaded_path = Path(
                _hf_hub_download(
                    repo_id=AUTO_PROMPT_REPO_ID,
                    repo_type=AUTO_PROMPT_REPO_TYPE,
                    revision=AUTO_PROMPT_REVISION,
                    filename=AUTO_PROMPT_REMOTE_FILE,
                    local_dir=target_root,
                    endpoint=candidate_endpoint,
                    force_download=bool(overwrite_existing),
                    etag_timeout=AUTO_PROMPT_ETAG_TIMEOUT_SECONDS,
                    user_agent="ComfyUI-Easy-SongGeneration",
                )
            )
            downloaded_endpoint = candidate_endpoint
            break
        except Exception as exc:
            download_error = exc

    if downloaded_path is None:
        raise RuntimeError(
            f"Failed to download SongGeneration auto prompt weights from {AUTO_PROMPT_REPO_ID}."
        ) from download_error

    if not _is_valid_weights_file(downloaded_path):
        raise RuntimeError(f"Downloaded auto prompt file is invalid or still a Git LFS pointer: {downloaded_path}")

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite_existing or not _is_valid_weights_file(canonical_path):
        shutil.copy2(downloaded_path, canonical_path)
    if not _is_valid_weights_file(canonical_path):
        raise RuntimeError(f"Auto prompt file is invalid or still a Git LFS pointer: {canonical_path}")
    return {
        "status": "downloaded",
        "path": os.fspath(canonical_path),
        "source_path": os.fspath(downloaded_path),
        "source": downloaded_endpoint or endpoint,
    }


def _verify_downloaded_files(target_root: Path, dry_run_files: list) -> dict[str, int]:
    checked = 0
    missing = []
    lfs_pointers = []
    size_mismatches = []
    for item in dry_run_files:
        relative = _dry_run_relative_path(item, target_root)
        if relative is None:
            continue
        path = _target_path(target_root, os.fspath(relative))
        checked += 1
        if not path.is_file():
            missing.append(os.fspath(relative))
            continue
        if _is_git_lfs_pointer(path):
            lfs_pointers.append(os.fspath(relative))
            continue
        expected_size = _dry_run_file_size(item)
        if expected_size is not None and path.stat().st_size != expected_size:
            size_mismatches.append(f"{relative} ({path.stat().st_size} != {expected_size})")

    invalid = [
        *(f"missing: {path}" for path in missing),
        *(f"Git LFS pointer: {path}" for path in lfs_pointers),
        *(f"size mismatch: {path}" for path in size_mismatches),
    ]
    if invalid:
        preview = "\n  - ".join(invalid[:10])
        if len(invalid) > 10:
            preview += f"\n  - ... and {len(invalid) - 10} more"
        raise RuntimeError(f"SongGeneration download integrity check failed:\n  - {preview}")
    return {
        "checked": checked,
        "missing": len(missing),
        "lfs_pointers": len(lfs_pointers),
        "size_mismatches": len(size_mismatches),
    }


def _verify_selected_local_files(target_root: Path, model_choice: str) -> dict[str, int]:
    checked = 0
    lfs_pointers = []
    for prefix in _selected_prefixes(model_choice):
        directory = _target_path(target_root, prefix)
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            checked += 1
            if _is_git_lfs_pointer(path):
                lfs_pointers.append(os.fspath(path.relative_to(target_root)))

    if lfs_pointers:
        preview = "\n  - ".join(lfs_pointers[:10])
        if len(lfs_pointers) > 10:
            preview += f"\n  - ... and {len(lfs_pointers) - 10} more"
        raise RuntimeError(f"SongGeneration download integrity check failed:\n  - Git LFS pointer: {preview}")
    return {"checked": checked, "missing": 0, "lfs_pointers": 0, "size_mismatches": 0}


def _verify_auto_prompt_asset(target_root: Path) -> dict[str, str]:
    checked = []
    for relative in AUTO_PROMPT_LOCAL_FILES:
        path = _target_path(target_root, relative)
        checked.append(os.fspath(path))
        if _is_valid_weights_file(path):
            return {"path": os.fspath(path), "status": "valid"}
    checked_text = "\n  - ".join(checked)
    raise RuntimeError(
        "SongGeneration auto prompt weights were not downloaded correctly. "
        f"Checked:\n  - {checked_text}"
    )


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
    removed_invalid_files = {"lfs_pointers": 0, "size_mismatches": 0}
    verification = {"checked": 0, "missing": 0, "lfs_pointers": 0, "size_mismatches": 0}
    auto_prompt = None
    auto_prompt_verification = None
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
                removed_invalid_files = _remove_invalid_download_files(target_root, dry_run_files)
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
            try:
                auto_prompt = _ensure_auto_prompt_asset(
                    target_root=target_root,
                    endpoint=endpoint,
                    overwrite_existing=bool(overwrite_existing),
                )
            except Exception as exc:
                # Auto-prompt weights are only needed for the optional automatic
                # style selector. Text descriptions and uploaded prompt audio can
                # use the downloaded model without them.
                auto_prompt = {
                    "status": "unavailable",
                    "repository": AUTO_PROMPT_REPO_ID,
                    "error": str(exc),
                }
                print(
                    "[Easy-SongGeneration] Optional auto prompt weights could not be downloaded; "
                    "text prompts and uploaded reference audio remain available. "
                    f"Reason: {exc}",
                    flush=True,
                )
            progress.update_absolute(2, total=3, label="校验 SongGeneration 模型下载")
            if dry_run_files:
                verification = _verify_downloaded_files(target_root, dry_run_files)
            else:
                verification = _verify_selected_local_files(target_root, model_choice)
            if auto_prompt.get("status") != "unavailable":
                auto_prompt_verification = _verify_auto_prompt_asset(target_root)
            else:
                auto_prompt_verification = {"status": "unavailable"}
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
        "invalid_files_removed": removed_invalid_files,
        "integrity_check": verification,
        "auto_prompt": auto_prompt,
        "auto_prompt_integrity_check": auto_prompt_verification,
    }
    return json.dumps(info, ensure_ascii=False, indent=2)
