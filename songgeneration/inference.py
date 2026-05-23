import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for extra_path in (
    SCRIPT_DIR,
    SCRIPT_DIR / "codeclm" / "tokenizer",
    SCRIPT_DIR / "codeclm" / "tokenizer" / "Flow1dVAE",
):
    extra_path = str(extra_path)
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TRANSFORMERS_CACHE", str(SCRIPT_DIR / "third_party" / "hub"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

AUTO_PROMPT_TYPES = ['Pop', 'Latin', 'Rock', 'Electronic', 'Metal', 'Country', 'R&B/Soul', 'Ballad', 'Jazz', 'World', 'Hip-Hop', 'Funk', 'Soundtrack', 'Auto']
V1_MODEL_NAMES = {
    "songgeneration_base",
    "songgeneration_base_new",
    "songgeneration_base_full",
    "songgeneration_large",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SongGeneration inference CLI for single-request and JSONL batch generation."
    )
    parser.add_argument("--ckpt-path", "--ckpt_path", dest="ckpt_path", required=True,
                        help="Checkpoint directory containing config.yaml and model.pt.")
    parser.add_argument("--input-jsonl", "--input_jsonl", dest="input_jsonl",
                        help="Batch input JSONL. If omitted, --lyrics is required.")
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", default="outputs",
                        help="Directory for generated audios and result JSONL.")
    parser.add_argument("--output-audio", dest="output_audio",
                        help="Single-request output WAV path. For --generate-type separate, _vocal/_bgm files are also copied.")
    parser.add_argument("--output-jsonl", dest="output_jsonl",
                        help="Optional path to copy the result JSONL metadata.")

    parser.add_argument("--idx", default="song_001", help="Single-request output id.")
    parser.add_argument("--lyrics", "--gt-lyric", dest="lyrics",
                        help="Single-request lyrics string in SongGeneration section format.")
    parser.add_argument("--descriptions", "--description", dest="descriptions",
                        help="Single-request comma-separated text prompt tags.")
    parser.add_argument("--prompt-audio-path", "--prompt_audio_path", dest="prompt_audio_path",
                        help="Single-request reference audio path.")
    parser.add_argument("--auto-prompt-audio-type", "--auto_prompt_audio_type",
                        dest="auto_prompt_audio_type", choices=AUTO_PROMPT_TYPES,
                        help="Single-request built-in auto prompt style.")

    parser.add_argument("--generate-type", "--generate_type", dest="generate_type",
                        choices=["mixed", "vocal", "bgm", "separate"], default="mixed",
                        help="Output type: mixed song, vocal only, bgm only, or separated tracks.")
    parser.add_argument("--version", choices=["auto", "v1", "v2"], default="auto",
                        help="Model prompt formatting version. Auto infers it from checkpoint folder name.")
    parser.add_argument("--low-mem", "--low_mem", dest="low_mem", action="store_true",
                        help="Force low-memory inference.")
    parser.add_argument("--use-flash-attn", "--use_flash_attn", dest="use_flash_attn",
                        action="store_true", default=True,
                        help="Enable Flash Attention.")
    parser.add_argument("--not-use-flash-attn", "--not_use_flash_attn", "--no-flash-attn",
                        dest="use_flash_attn", action="store_false",
                        help="Disable Flash Attention.")
    parser.add_argument("--gpu-id", "--gpu_id", dest="gpu_id", type=int,
                        help="CUDA device id.")
    parser.add_argument("--seed", type=int,
                        help="Random seed. Defaults to current time.")

    parser.add_argument("--config-path", "--config_path", dest="config_path",
                        help="Override config.yaml path.")
    parser.add_argument("--model-path", "--model_path", dest="model_path",
                        help="Override model.pt path.")
    parser.add_argument("--audio-tokenizer-checkpoint", "--audio_tokenizer_checkpoint",
                        dest="audio_tokenizer_checkpoint",
                        help="Override cfg.audio_tokenizer_checkpoint.")
    parser.add_argument("--audio-tokenizer-checkpoint-sep", "--audio_tokenizer_checkpoint_sep",
                        dest="audio_tokenizer_checkpoint_sep",
                        help="Override cfg.audio_tokenizer_checkpoint_sep.")
    parser.add_argument("--demucs-model-path", "--demucs_model_path", dest="demucs_model_path",
                        help="Demucs model path for prompt audio separation.")
    parser.add_argument("--demucs-config-path", "--demucs_config_path", dest="demucs_config_path",
                        help="Demucs config path for prompt audio separation.")
    parser.add_argument("--auto-prompt-path", "--auto_prompt_path", dest="auto_prompt_path",
                        default=str(SCRIPT_DIR / "tools" / "new_auto_prompt.pt"),
                        help="Path to new_auto_prompt.pt.")

    parser.add_argument("--duration", type=float,
                        help="Generated duration in seconds. Defaults to config max_dur.")
    parser.add_argument("--extend-stride", "--extend_stride", dest="extend_stride",
                        type=float, default=5,
                        help="Generation extend stride in seconds.")
    parser.add_argument("--temperature", type=float,
                        help="Sampling temperature. Defaults follow the original script.")
    parser.add_argument("--cfg-coef", "--cfg_coef", dest="cfg_coef", type=float, default=1.5,
                        help="Classifier-free guidance coefficient.")
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int,
                        help="Top-k sampling value. Defaults follow the selected memory mode.")
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.0,
                        help="Top-p sampling value. 0 disables top-p.")
    parser.add_argument("--no-sampling", "--no_sampling", dest="use_sampling",
                        action="store_false",
                        help="Use greedy decoding instead of sampling.")
    parser.set_defaults(use_sampling=True)
    parser.add_argument("--record-tokens", "--record_tokens", dest="record_tokens",
                        action="store_true", default=True,
                        help="Enable token recording.")
    parser.add_argument("--no-record-tokens", "--no_record_tokens", dest="record_tokens",
                        action="store_false",
                        help="Disable token recording.")
    parser.add_argument("--record-window", "--record_window", dest="record_window",
                        type=int, default=50,
                        help="Token recording window.")
    parser.add_argument("--chunk-size", "--chunk_size", dest="chunk_size", type=int, default=128,
                        help="Diffusion decode chunk size.")
    parser.add_argument("--audio-format", "--audio_format", dest="audio_format",
                        choices=["wav"], default="wav",
                        help="Output audio format. Currently only PCM WAV is supported.")
    return parser.parse_args()


def validate_args(args):
    if args.input_jsonl and args.lyrics:
        raise SystemExit("error: use either --input-jsonl for batch mode or --lyrics for single-request mode, not both.")
    if not args.input_jsonl and not args.lyrics:
        raise SystemExit("error: provide --input-jsonl or --lyrics.")
    if args.prompt_audio_path and args.auto_prompt_audio_type:
        raise SystemExit("error: --prompt-audio-path and --auto-prompt-audio-type cannot be used together.")
    if args.input_jsonl and args.output_audio:
        raise SystemExit("error: --output-audio is only supported for single-request mode.")


def build_single_request_jsonl(args):
    item = {
        "idx": args.idx,
        "gt_lyric": args.lyrics,
    }
    if args.descriptions:
        item["descriptions"] = args.descriptions
    if args.prompt_audio_path:
        item["prompt_audio_path"] = args.prompt_audio_path
    if args.auto_prompt_audio_type:
        item["auto_prompt_audio_type"] = args.auto_prompt_audio_type

    safe_idx = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(args.idx))
    temp_dir = tempfile.mkdtemp(prefix="songgeneration_inference_")
    input_jsonl = os.path.join(temp_dir, f"{safe_idx}.jsonl")
    with open(input_jsonl, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return input_jsonl, temp_dir


def infer_version(args):
    if args.version != "auto":
        return args.version
    model_name = Path(args.ckpt_path).name.lower().replace("-", "_")
    if model_name in V1_MODEL_NAMES:
        return "v1"
    return "v2"


def should_use_lowmem(args, torch):
    if args.low_mem:
        return True
    model_name = Path(args.ckpt_path).name.lower().replace("-", "_")
    device = torch.cuda.current_device()
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    available_gb = (total - reserved) / 1024 / 1024 / 1024
    if model_name in {"songgeneration_base", "songgeneration_base_new", "songgeneration_base_full"}:
        return available_gb <= 24
    if model_name == "songgeneration_large":
        return available_gb <= 36
    if model_name == "songgeneration_v2_large":
        return available_gb <= 32
    return False


def result_jsonl_path(args):
    return Path(args.save_dir) / "jsonl" / f"{Path(args.input_jsonl).stem}.jsonl"


def copy_single_outputs(args):
    if not args.output_audio:
        return
    output_audio = Path(args.output_audio)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    generated = Path(args.save_dir) / "audios" / f"{args.idx}.wav"
    shutil.copy2(generated, output_audio)
    if args.generate_type == "separate":
        stem = output_audio.with_suffix("")
        shutil.copy2(generated.with_name(f"{args.idx}_vocal.wav"), stem.with_name(stem.name + "_vocal").with_suffix(".wav"))
        shutil.copy2(generated.with_name(f"{args.idx}_bgm.wav"), stem.with_name(stem.name + "_bgm").with_suffix(".wav"))


def main():
    args = parse_args()
    validate_args(args)

    import numpy as np
    import torch
    from generate import generate, generate_lowmem
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. SongGeneration inference requires a CUDA GPU.")
    if args.gpu_id is not None:
        torch.cuda.set_device(int(args.gpu_id))

    seed = args.seed if args.seed is not None else int(time.time())
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda: Path(args.ckpt_path).name)
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))

    temp_dir = None
    if not args.input_jsonl:
        args.input_jsonl, temp_dir = build_single_request_jsonl(args)

    try:
        version = infer_version(args)
        lowmem = should_use_lowmem(args, torch)
        print(f"SongGeneration inference: version={version}, low_mem={lowmem}, seed={seed}")
        if lowmem:
            generate_lowmem(args, version=version)
        else:
            generate(args, version=version)

        copy_single_outputs(args)
        if args.output_jsonl:
            output_jsonl = Path(args.output_jsonl)
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result_jsonl_path(args), output_jsonl)
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
