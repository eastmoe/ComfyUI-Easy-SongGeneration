# Project Structure

This repository follows the usual ComfyUI custom node layout:

- `__init__.py` exports `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, and the ComfyUI `WEB_DIRECTORY`.
- `nodes.py` is the thin ComfyUI entry point.
- `src/easy_songgeneration/` contains implementation code split by concern.
- `web/` contains frontend extensions, including the lyrics/style formatter shortcut buttons.
- `locales/<locale>/nodes.json` contains node strings, tooltips, display names, and return names.
- `requirements.txt` contains installable Python dependencies for ComfyUI Manager.
- `songgeneration/` contains the trimmed upstream SongGeneration runtime.

Model checkpoints and quantization caches are intentionally not stored in this
custom node directory. They belong under:

```text
ComfyUI/models/SongGeneration/
ComfyUI/models/SongGeneration-cache/
```
