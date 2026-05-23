from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

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


def _base_url(source: str) -> str:
    value = (source or DOWNLOAD_SOURCES[0]).strip().lower()
    if value in {"hf-mirror", "hf-mirror.com", "https://hf-mirror.com"}:
        return "https://hf-mirror.com"
    if value in {"huggingface", "huggingface.co", "https://huggingface.co"}:
        return "https://huggingface.co"
    raise ValueError(f"Unsupported download source: {source}")


def _headers() -> dict[str, str]:
    return {"User-Agent": "ComfyUI-Easy-SongGeneration"}


def _read_json(url: str) -> Any:
    request = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to read Hugging Face API ({exc.code}): {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to connect to Hugging Face API: {exc.reason}") from exc


def _repo_tree(base_url: str, revision: str) -> list[dict[str, Any]]:
    repo = urllib.parse.quote(SONGGEN_REPO_ID, safe="/")
    rev = urllib.parse.quote((revision or "main").strip() or "main", safe="")
    url = f"{base_url}/api/models/{repo}/tree/{rev}?recursive=1"
    payload = _read_json(url)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected repository tree response from {base_url}")
    return [item for item in payload if isinstance(item, dict)]


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


def _selected_files(tree: list[dict[str, Any]], model_choice: str) -> list[dict[str, Any]]:
    prefixes = _selected_prefixes(model_choice)
    files = []
    for item in tree:
        path = str(item.get("path") or "").strip("/")
        if item.get("type") == "file" and path.startswith(prefixes):
            files.append({**item, "path": path})
    return sorted(files, key=lambda item: item["path"].lower())


def _download_url(base_url: str, revision: str, path: str) -> str:
    repo = urllib.parse.quote(SONGGEN_REPO_ID, safe="/")
    rev = urllib.parse.quote((revision or "main").strip() or "main", safe="")
    quoted_path = urllib.parse.quote(path, safe="/")
    return f"{base_url}/{repo}/resolve/{rev}/{quoted_path}?download=true"


def _file_size(item: dict[str, Any]) -> int | None:
    size = item.get("size")
    if isinstance(size, int) and size >= 0:
        return size
    return None


def _is_current(path: Path, size: int | None) -> bool:
    if not path.is_file():
        return False
    if size is None:
        return path.stat().st_size > 0
    return path.stat().st_size == size


def _open_with_retries(url: str, *, offset: int = 0, retries: int = 3):
    headers = _headers()
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        request = urllib.request.Request(url, headers=headers)
        try:
            return urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as exc:
            if offset > 0 and exc.code == 416:
                raise
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def _download_one(
    *,
    base_url: str,
    revision: str,
    item: dict[str, Any],
    target_root: Path,
    overwrite_existing: bool,
    progress: _SongGenProgress,
    downloaded_bytes: int,
    total_bytes: int,
) -> tuple[str, int]:
    relative = item["path"]
    size = _file_size(item)
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and _is_current(target, size):
        return f"skip {relative}", downloaded_bytes + (size or 0)

    temp = target.with_name(f"{target.name}.download")
    if overwrite_existing and temp.exists():
        temp.unlink()

    offset = temp.stat().st_size if temp.exists() else 0
    if size is not None and offset > size:
        temp.unlink()
        offset = 0
    if size is not None and offset == size:
        temp.replace(target)
        return f"download {relative}", downloaded_bytes + size

    url = _download_url(base_url, revision, relative)
    mode = "ab" if offset > 0 else "wb"
    current = downloaded_bytes + offset
    progress.update_absolute(current, total=max(1, total_bytes), label=f"下载 {relative}")
    try:
        with _open_with_retries(url, offset=offset) as response:
            if offset > 0 and getattr(response, "status", None) == 200:
                temp.unlink(missing_ok=True)
                mode = "wb"
                current = downloaded_bytes
            with temp.open(mode) as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
                    current += len(chunk)
                    progress.update_absolute(current, total=max(1, total_bytes), label=f"下载 {relative}")
    except Exception:
        if temp.exists() and temp.stat().st_size == 0:
            temp.unlink(missing_ok=True)
        raise

    if size is not None and temp.stat().st_size != size:
        raise RuntimeError(f"Downloaded size mismatch for {relative}: expected {size}, got {temp.stat().st_size}")
    temp.replace(target)
    return f"download {relative}", downloaded_bytes + (size or target.stat().st_size)


def download_songgeneration_assets(
    *,
    source: str,
    model_choice: str,
    revision: str,
    overwrite_existing: bool,
) -> str:
    base_url = _base_url(source)
    target_root = _songgen_model_root()
    tree = _repo_tree(base_url, revision)
    files = _selected_files(tree, model_choice)
    if not files:
        raise RuntimeError(f"No matching files found in {SONGGEN_REPO_ID} for {model_choice}")

    planned_bytes = 0
    for item in files:
        size = _file_size(item)
        target = target_root / item["path"]
        if overwrite_existing or not _is_current(target, size):
            planned_bytes += size or 1

    progress = _SongGenProgress(max(1, planned_bytes), "下载 SongGeneration 模型")
    downloaded_bytes = 0
    downloaded = 0
    skipped = 0
    try:
        for item in files:
            size = _file_size(item)
            target = target_root / item["path"]
            if not overwrite_existing and _is_current(target, size):
                skipped += 1
                continue
            action, downloaded_bytes = _download_one(
                base_url=base_url,
                revision=revision,
                item=item,
                target_root=target_root,
                overwrite_existing=overwrite_existing,
                progress=progress,
                downloaded_bytes=downloaded_bytes,
                total_bytes=max(1, planned_bytes),
            )
            if action.startswith("download "):
                downloaded += 1
        progress.finish("SongGeneration 模型下载完成")
    except Exception:
        progress.close()
        raise

    selected_dirs = [prefix.rstrip("/") for prefix in _selected_prefixes(model_choice)]
    info = {
        "repository": SONGGEN_REPO_ID,
        "source": base_url,
        "revision": (revision or "main").strip() or "main",
        "target": os.fspath(target_root),
        "directories": selected_dirs,
        "files": len(files),
        "downloaded": downloaded,
        "skipped": skipped,
    }
    return json.dumps(info, ensure_ascii=False, indent=2)
