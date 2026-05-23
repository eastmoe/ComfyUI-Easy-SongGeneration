# ComfyUI Easy SongGeneration

ComfyUI wrapper for the bundled `songgeneration` inference code. The plugin loads SongGeneration checkpoints from ComfyUI's model directory and exposes native ComfyUI `AUDIO` outputs.

## Model Layout

Put all model files under:

```text
ComfyUI/models/SongGeneration/
```

Recommended layout:

```text
ComfyUI/models/SongGeneration/
  SongGeneration-v2-large/
    config.yaml
    model.pt
  ckpt/
    ...
  third_party/
    demucs/
      ckpt/
        htdemucs.pth
        htdemucs.yaml
```

Subfolders are supported, so you can keep multiple checkpoints side by side:

```text
ComfyUI/models/SongGeneration/
  songgeneration_base/
    config.yaml
    model.pt
  songgeneration_large/
    config.yaml
    model.pt
  songgeneration_v2_large/
    config.yaml
    model.pt
```

The loader scans for folders containing both `config.yaml` and `model.pt`.

## Nodes

- `Easy SongGeneration - Load Model`: loads a checkpoint and returns `SONGGEN_MODEL`.
- `Easy SongGeneration - Release Model`: releases the loaded model and clears CUDA cache.
- `Easy SongGeneration - Generate Mixed`: generates a full mixed song.
- `Easy SongGeneration - Generate Vocal`: generates vocal-only audio.
- `Easy SongGeneration - Generate BGM`: generates accompaniment / pure music.
- `Easy SongGeneration - Generate Separate`: outputs mixed, vocal, and BGM tracks separately.

All generation nodes return ComfyUI native `AUDIO`, so they can connect to ComfyUI's `Preview Audio` or `Save Audio` nodes.

## Prompt Modes

Generation nodes support three prompt styles:

- No prompt: set `auto_prompt_audio_type` to `None` and leave `prompt_audio` disconnected.
- Auto prompt: choose one of `Pop`, `Latin`, `Rock`, `Electronic`, `Metal`, `Country`, `R&B/Soul`, `Ballad`, `Jazz`, `World`, `Hip-Hop`, `Funk`, `Soundtrack`, or `Auto`.
- Audio prompt: connect a ComfyUI `AUDIO` input to `prompt_audio`. This takes priority over `auto_prompt_audio_type`.

Lyrics should use the original SongGeneration section format, for example:

```text
[intro-short] ; [verse] Stars are waking in the rain. ; [chorus] We sing until the morning light. ; [outro-short]
```

## Runtime Files

The wrapper searches runtime assets in this order:

1. The selected checkpoint folder.
2. `ComfyUI/models/SongGeneration`.
3. This plugin's bundled `songgeneration` folder.

If your `config.yaml` references relative tokenizer paths such as `ckpt/...`, place those folders under `ComfyUI/models/SongGeneration` or next to the checkpoint folder.

For audio prompt support, Demucs files are expected at:

```text
ComfyUI/models/SongGeneration/third_party/demucs/ckpt/htdemucs.pth
ComfyUI/models/SongGeneration/third_party/demucs/ckpt/htdemucs.yaml
```

The bundled `songgeneration/tools/new_auto_prompt.pt` is used for auto prompts unless you provide another copy under the model root.

## Notes

- SongGeneration requires CUDA.
- The current ComfyUI loader keeps the LM and diffusion tokenizer resident so downstream generation nodes can reuse the loaded model.
- If Flash Attention is not installed or unsupported on your GPU, disable `Flash Attention` in the loader node.
