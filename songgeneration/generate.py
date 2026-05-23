import sys
import os
import argparse

import time
import json
import wave
import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf
from codeclm.models import builders
import gc
from codeclm.models import CodecLM
from third_party.demucs.models.pretrained import get_model_from_yaml
import re

auto_prompt_type = ['Pop', 'Latin', 'Rock', 'Electronic', 'Metal', 'Country', 'R&B/Soul', 'Ballad', 'Jazz', 'World', 'Hip-Hop', 'Funk', 'Soundtrack','Auto']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_arg(args, name, default=None):
    return getattr(args, name, default)


def runtime_roots(ckpt_dir):
    roots = [
        ckpt_dir,
        os.path.dirname(os.path.abspath(ckpt_dir)),
        SCRIPT_DIR,
        os.getcwd(),
    ]
    deduped = []
    seen = set()
    for root in roots:
        if not root:
            continue
        key = os.path.abspath(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def common_ckpt_alias(path):
    norm = path.replace("\\", "/")
    leading = "./" if norm.startswith("./") else ""
    stripped = norm[2:] if leading else norm
    aliases = {"ckpt": "common", "common": "ckpt"}
    for source, target in aliases.items():
        if stripped == source or stripped.startswith(source + "/"):
            return leading + target + stripped[len(source):]
    return None


def resolve_existing_path(value, roots):
    if not value:
        return value
    text = str(value).strip()
    if os.path.isabs(text):
        return text
    candidates = [text]
    alias = common_ckpt_alias(text)
    if alias:
        candidates.append(alias)
    for root in roots:
        for candidate in candidates:
            path = os.path.join(root, candidate)
            if os.path.exists(path):
                return path
    return text


def resolve_prefixed_existing_path(value, roots):
    if not value:
        return value
    text = str(value).strip()
    prefixes = ("Flow1dVAE1rvq_", "Flow1dVAESeparate_")
    for prefix in prefixes:
        if text.startswith(prefix):
            return prefix + resolve_existing_path(text[len(prefix):], roots)
    return resolve_existing_path(text, roots)


def resolve_config_paths(cfg, roots):
    cfg.audio_tokenizer_checkpoint = resolve_prefixed_existing_path(cfg.audio_tokenizer_checkpoint, roots)
    if "audio_tokenizer_checkpoint_sep" in cfg.keys():
        cfg.audio_tokenizer_checkpoint_sep = resolve_prefixed_existing_path(cfg.audio_tokenizer_checkpoint_sep, roots)
    if "vae_config" in cfg.keys():
        cfg.vae_config = resolve_existing_path(cfg.vae_config, roots)
    if "vae_model" in cfg.keys():
        cfg.vae_model = resolve_existing_path(cfg.vae_model, roots)
    return cfg


def prepare_inference_env(args, cfg):
    if get_arg(args, "gpu_id") is not None:
        torch.cuda.set_device(int(args.gpu_id))
    if get_arg(args, "audio_tokenizer_checkpoint"):
        cfg.audio_tokenizer_checkpoint = args.audio_tokenizer_checkpoint
    if get_arg(args, "audio_tokenizer_checkpoint_sep"):
        cfg.audio_tokenizer_checkpoint_sep = args.audio_tokenizer_checkpoint_sep
    return cfg


def resolve_generation_params(args, max_duration, low_mem=False):
    default_temp = 0.9 if low_mem else 0.8
    default_top_k = 50 if low_mem else 5000
    temperature = get_arg(args, "temperature", None)
    top_k = get_arg(args, "top_k", None)
    return {
        "duration": get_arg(args, "duration", None) or max_duration,
        "extend_stride": get_arg(args, "extend_stride", 5),
        "temperature": default_temp if temperature is None else temperature,
        "cfg_coef": get_arg(args, "cfg_coef", 1.5),
        "top_k": default_top_k if top_k is None else top_k,
        "top_p": get_arg(args, "top_p", 0.0),
        "use_sampling": get_arg(args, "use_sampling", True),
        "record_tokens": get_arg(args, "record_tokens", True),
        "record_window": get_arg(args, "record_window", 50),
    }

def check_language_by_text(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    english_pattern = re.compile(r'[a-zA-Z]')
    chinese_count = len(re.findall(chinese_pattern, text))
    english_count = len(re.findall(english_pattern, text))
    chinese_ratio = chinese_count / len(text)
    english_ratio = english_count / len(text)
    if chinese_ratio >= 0.2:
        return "zh"
    elif english_ratio >= 0.5:
        return "en"
    else:
        return "en"

def load_pcm_wav(f):
    """Minimal stdlib WAV fallback for environments without optional audio backends."""
    with wave.open(f, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        audio = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        audio = ((audio ^ 0x800000) - 0x800000).astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM WAV sample width: {sample_width}")

    audio = audio.reshape(-1, channels).T
    return torch.from_numpy(audio), sample_rate


def load_audio_48k(f):
    try:
        audio, sample_rate = torchaudio.load(f)
    except Exception:
        audio, sample_rate = load_pcm_wav(f)
    if sample_rate != 48000:
        audio = torchaudio.functional.resample(audio, sample_rate, 48000)
    if audio.shape[-1] >= 48000 * 10:
        audio = audio[..., :48000 * 10]
    return audio[:, 0:48000 * 10]


def save_pcm_wav(path, audio, sample_rate):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    audio = audio.detach().cpu().float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2:
        raise ValueError("WAV audio should have shape [C, T] or [T].")

    pcm = (audio.clamp(-1.0, 1.0).t().contiguous().numpy() * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(audio.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())

class Separator:
    def __init__(self, dm_model_path='third_party/demucs/ckpt/htdemucs.pth', dm_config_path='third_party/demucs/ckpt/htdemucs.yaml', gpu_id=0) -> None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.demucs_model = self.init_demucs_model(dm_model_path, dm_config_path)

    def init_demucs_model(self, model_path, config_path):
        model = get_model_from_yaml(config_path, model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def load_audio(self, f):
        return load_audio_48k(f)
    
    def run(self, audio_path, output_dir='tmp', ext=".wav"):
        os.makedirs(output_dir, exist_ok=True)
        name, _ = os.path.splitext(os.path.split(audio_path)[-1])
        output_paths = []

        for stem in self.demucs_model.sources:
            output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
            if os.path.exists(output_path):
                output_paths.append(output_path)
        if len(output_paths) == 1:  # 4
            vocal_path = output_paths[0]
        else:
            drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(audio_path, output_dir, device=self.device)
            for path in [drums_path, bass_path, other_path]:
                os.remove(path)
        full_audio = self.load_audio(audio_path)
        vocal_audio = self.load_audio(vocal_path)
        bgm_audio = full_audio - vocal_audio
        return full_audio, vocal_audio, bgm_audio


def parse_args():
    parser = argparse.ArgumentParser(description='Song Generation Script')
    
    # 必需参数
    parser.add_argument('--ckpt_path', type=str, required=True,
                      help='Path to the checkpoint directory containing config.yaml and model.pt')
    parser.add_argument('--input_jsonl', type=str, required=True,
                      help='Path to input JSONL file containing generation tasks')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save generated audio files and results')
    # 可选参数
    parser.add_argument('--generate_type', type=str, default='mixed',
                      help='Type of generation: "vocal" or "bgm" or "separate" or "mixed" (default: "mixed")')
    parser.add_argument('--use_flash_attn', action='store_true',
                      help='Whether to use flash attention (default: False)')
    parser.add_argument('--low_mem', action='store_true',
                      help='Whether to use low memory mode (default: False)')
    parser.add_argument('--config_path', type=str, default=None,
                      help='Optional path to config.yaml. Defaults to <ckpt_path>/config.yaml')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Optional path to model.pt. Defaults to <ckpt_path>/model.pt')
    parser.add_argument('--audio_tokenizer_checkpoint', type=str, default=None,
                      help='Override cfg.audio_tokenizer_checkpoint')
    parser.add_argument('--audio_tokenizer_checkpoint_sep', type=str, default=None,
                      help='Override cfg.audio_tokenizer_checkpoint_sep')
    parser.add_argument('--demucs_model_path', type=str, default=None,
                      help='Path to Demucs htdemucs.pth used for prompt audio separation')
    parser.add_argument('--demucs_config_path', type=str, default=None,
                      help='Path to Demucs htdemucs.yaml used for prompt audio separation')
    parser.add_argument('--auto_prompt_path', type=str, default=os.path.join(SCRIPT_DIR, 'tools', 'new_auto_prompt.pt'),
                      help='Path to auto prompt token file')
    parser.add_argument('--gpu_id', type=int, default=None,
                      help='CUDA device id to use')
    parser.add_argument('--duration', type=float, default=None,
                      help='Generated duration in seconds. Defaults to cfg.max_dur')
    parser.add_argument('--extend_stride', type=float, default=5,
                      help='Stride in seconds for generation params')
    parser.add_argument('--temperature', type=float, default=None,
                      help='Sampling temperature')
    parser.add_argument('--cfg_coef', type=float, default=1.5,
                      help='Classifier-free guidance coefficient')
    parser.add_argument('--top_k', type=int, default=None,
                      help='Top-k sampling value')
    parser.add_argument('--top_p', type=float, default=0.0,
                      help='Top-p sampling value. 0 disables top-p')
    parser.add_argument('--no_sampling', dest='use_sampling', action='store_false',
                      help='Disable sampling and use greedy decoding')
    parser.set_defaults(use_sampling=True)
    parser.add_argument('--record_tokens', dest='record_tokens', action='store_true',
                      help='Enable token recording during generation')
    parser.add_argument('--no_record_tokens', dest='record_tokens', action='store_false',
                      help='Disable token recording during generation')
    parser.set_defaults(record_tokens=True)
    parser.add_argument('--record_window', type=int, default=50,
                      help='Token recording window size')
    parser.add_argument('--chunk_size', type=int, default=128,
                      help='Chunk size for diffusion audio decoding')
    return parser.parse_args()

def generate(args, version = 'v1'):
    torch.set_num_threads(1)
    if get_arg(args, "gpu_id") is not None:
        torch.cuda.set_device(int(args.gpu_id))
    ckpt_dir = args.ckpt_path
    input_jsonl = args.input_jsonl
    save_dir = args.save_dir
    cfg_path = get_arg(args, "config_path") or os.path.join(ckpt_dir, 'config.yaml')
    ckpt_path = get_arg(args, "model_path") or os.path.join(ckpt_dir, 'model.pt')
    cfg = OmegaConf.load(cfg_path)
    cfg = prepare_inference_env(args, cfg)
    cfg = resolve_config_paths(cfg, runtime_roots(ckpt_dir))
    cfg.lm.use_flash_attn_2 = args.use_flash_attn
    print(f"use_flash_attn: {args.use_flash_attn}")
    cfg.mode = 'inference'
    max_duration = cfg.max_dur
    gen_type = args.generate_type
    

    demucs_model_path = get_arg(args, "demucs_model_path") or os.path.join(SCRIPT_DIR, 'third_party', 'demucs', 'ckpt', 'htdemucs.pth')
    demucs_config_path = get_arg(args, "demucs_config_path") or os.path.join(SCRIPT_DIR, 'third_party', 'demucs', 'ckpt', 'htdemucs.yaml')
    separator = Separator(demucs_model_path, demucs_config_path, gpu_id=get_arg(args, "gpu_id", 0) or 0)
    auto_prompt = torch.load(get_arg(args, "auto_prompt_path", os.path.join(SCRIPT_DIR, 'tools', 'new_auto_prompt.pt')))
    audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
    audio_tokenizer = audio_tokenizer.eval().cuda()
    with open(input_jsonl, "r") as fp:
        lines = fp.readlines()

        
    new_items = []
    for line in lines:
        item = json.loads(line)
        target_wav_name = f"{save_dir}/audios/{item['idx']}.wav"
        # get prompt audio
        if "prompt_audio_path" in item:
            assert os.path.exists(item['prompt_audio_path']), f"prompt_audio_path {item['prompt_audio_path']} not found"
            assert 'auto_prompt_audio_type' not in item, f"auto_prompt_audio_type and prompt_audio_path cannot be used together"
            with torch.no_grad():
                pmt_wav, vocal_wav, bgm_wav = separator.run(item['prompt_audio_path'])
            item['raw_pmt_wav'] = pmt_wav
            item['raw_vocal_wav'] = vocal_wav
            item['raw_bgm_wav'] = bgm_wav
            if pmt_wav.dim() == 2:
                pmt_wav = pmt_wav[None]
            if pmt_wav.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            pmt_wav = list(pmt_wav)
            if vocal_wav.dim() == 2:
                vocal_wav = vocal_wav[None]
            if vocal_wav.dim() != 3:
                raise ValueError("Vocal wavs should have a shape [B, C, T].")
            vocal_wav = list(vocal_wav)
            if bgm_wav.dim() == 2:
                bgm_wav = bgm_wav[None]
            if bgm_wav.dim() != 3:
                raise ValueError("BGM wavs should have a shape [B, C, T].")
            bgm_wav = list(bgm_wav)
            if type(pmt_wav) == list:
                pmt_wav = torch.stack(pmt_wav, dim=0)
            if type(vocal_wav) == list:
                vocal_wav = torch.stack(vocal_wav, dim=0)
            if type(bgm_wav) == list:
                bgm_wav = torch.stack(bgm_wav, dim=0)
            pmt_wav = pmt_wav
            vocal_wav = vocal_wav
            bgm_wav = bgm_wav
            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(pmt_wav.cuda())
            melody_is_wav = False
        elif "auto_prompt_audio_type" in item:
            assert item["auto_prompt_audio_type"] in auto_prompt_type, f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
            lang = check_language_by_text(item['gt_lyric'])
            prompt_token = auto_prompt[item["auto_prompt_audio_type"]][lang][np.random.randint(0, len(auto_prompt[item["auto_prompt_audio_type"]][lang]))]
            pmt_wav = prompt_token[:,[0],:]
            vocal_wav = prompt_token[:,[1],:]
            bgm_wav = prompt_token[:,[2],:]
            melody_is_wav = False
        else:
            pmt_wav = None
            vocal_wav = None
            bgm_wav = None
            melody_is_wav = True
        item['pmt_wav'] = pmt_wav
        item['vocal_wav'] = vocal_wav
        item['bgm_wav'] = bgm_wav
        item['melody_is_wav'] = melody_is_wav
        item["idx"] = f"{item['idx']}"
        item["wav_path"] = target_wav_name
        new_items.append(item)

    del audio_tokenizer
    del separator

    torch.cuda.empty_cache()

    if "audio_tokenizer_checkpoint_sep" in cfg.keys():
        seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg) 
    else:
        seperate_tokenizer = None
    
    if seperate_tokenizer is not None:
        seperate_tokenizer = seperate_tokenizer.eval().cuda()

    for item in new_items:
        if "prompt_audio_path" in item:
            with torch.no_grad():
                vocal_wav, bgm_wav = seperate_tokenizer.encode(item['vocal_wav'].cuda(), item['bgm_wav'].cuda())
            item['vocal_wav'] = vocal_wav
            item['bgm_wav'] = bgm_wav

    torch.cuda.empty_cache()
    audiolm = builders.get_lm_model(cfg, version=version)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
    audiolm.load_state_dict(audiolm_state_dict, strict=False)
    audiolm = audiolm.eval()
    audiolm = audiolm.cuda().to(torch.float16)

    model = CodecLM(name = "tmp",
        lm = audiolm,
        audiotokenizer = None,
        max_duration = max_duration,
        seperate_tokenizer = seperate_tokenizer,
    )

    gen_params = resolve_generation_params(args, max_duration, low_mem=False)
    model.set_generation_params(**gen_params)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "/audios", exist_ok=True)
    os.makedirs(save_dir + "/jsonl", exist_ok=True)

    for item in new_items:
        lyric = item["gt_lyric"]
        if version == 'v1':
            descriptions = item["descriptions"].lower() if "descriptions" in item else None
        else:
            if gen_type == 'bgm':
                descriptions = '[Musicality-very-high]' + ', ' + '[Pure-Music]' + ', ' + item["descriptions"].lower() if "descriptions" in item else '.'
            else:
                descriptions = item["descriptions"].lower() if "descriptions" in item else '.'
                descriptions = '[Musicality-very-high]' + ', ' + descriptions

        pmt_wav = item['pmt_wav']
        vocal_wav = item['vocal_wav']
        bgm_wav = item['bgm_wav']
        melody_is_wav = item['melody_is_wav']
        target_wav_name = f"{save_dir}/audios/{item['idx']}.wav"

        generate_inp = {
            'lyrics': [lyric.replace("  ", " ")] if gen_type != 'bgm' else '.',
            'descriptions': [descriptions],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }
        start_time = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = model.generate(**generate_inp, return_tokens=True)
        mid_time = time.time()

        with torch.no_grad():
            if 'raw_pmt_wav' in item:
                if gen_type == 'separate':
                    wav_seperate = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='mixed')
                    wav_vocal = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='vocal')
                    wav_bgm = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='bgm')
                elif gen_type == 'mixed':
                    wav_seperate = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type=gen_type)
                else:
                    wav_seperate = model.generate_audio(tokens, chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type=gen_type)
                del item['raw_pmt_wav']
                del item['raw_vocal_wav']
                del item['raw_bgm_wav']
            else:
                if gen_type == 'separate':
                    wav_vocal = model.generate_audio(tokens, chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='vocal')
                    wav_bgm = model.generate_audio(tokens, chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='bgm')
                    wav_seperate = model.generate_audio(tokens, chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='mixed')
                else:
                    wav_seperate = model.generate_audio(tokens, chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type=gen_type)
        del item['pmt_wav']
        del item['vocal_wav']
        del item['bgm_wav']
        del item['melody_is_wav']
        end_time = time.time()
        if gen_type == 'separate':
            save_pcm_wav(target_wav_name.replace('.wav', '_vocal.wav'), wav_vocal[0], cfg.sample_rate)
            save_pcm_wav(target_wav_name.replace('.wav', '_bgm.wav'), wav_bgm[0], cfg.sample_rate)
            save_pcm_wav(target_wav_name, wav_seperate[0], cfg.sample_rate)
        else:
            save_pcm_wav(target_wav_name, wav_seperate[0], cfg.sample_rate)

        print(f"process{item['idx']}, lm cost {mid_time - start_time}s, diffusion cost {end_time - mid_time}")
        item["idx"] = f"{item['idx']}"
        item["wav_path"] = target_wav_name
    
    src_jsonl_name = os.path.splitext(os.path.split(input_jsonl)[-1])[0]
    with open(f"{save_dir}/jsonl/{src_jsonl_name}.jsonl", "w", encoding='utf-8') as fw:
        for item in new_items:
            fw.writelines(json.dumps(item, ensure_ascii=False)+"\n")

def generate_lowmem(args, version = 'v1'):
    torch.set_num_threads(1)
    if get_arg(args, "gpu_id") is not None:
        torch.cuda.set_device(int(args.gpu_id))
    ckpt_dir = args.ckpt_path
    input_jsonl = args.input_jsonl
    save_dir = args.save_dir
    cfg_path = get_arg(args, "config_path") or os.path.join(ckpt_dir, 'config.yaml')
    ckpt_path = get_arg(args, "model_path") or os.path.join(ckpt_dir, 'model.pt')
    cfg = OmegaConf.load(cfg_path)
    cfg = prepare_inference_env(args, cfg)
    cfg = resolve_config_paths(cfg, runtime_roots(ckpt_dir))
    cfg.lm.use_flash_attn_2 = args.use_flash_attn
    print(f"use_flash_attn: {args.use_flash_attn}")
    cfg.mode = 'inference'
    max_duration = cfg.max_dur
    gen_type = args.generate_type
    chunk_size = 128
    use_audio_tokenizer = False
    with open(input_jsonl, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        item = json.loads(line)
        if "prompt_audio_path" in item:
            use_audio_tokenizer = True
            break
    if use_audio_tokenizer:
        demucs_model_path = get_arg(args, "demucs_model_path") or os.path.join(SCRIPT_DIR, 'third_party', 'demucs', 'ckpt', 'htdemucs.pth')
        demucs_config_path = get_arg(args, "demucs_config_path") or os.path.join(SCRIPT_DIR, 'third_party', 'demucs', 'ckpt', 'htdemucs.yaml')
        separator = Separator(demucs_model_path, demucs_config_path, gpu_id=get_arg(args, "gpu_id", 0) or 0)
        audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
        audio_tokenizer = audio_tokenizer.eval().cuda()
    auto_prompt = torch.load(get_arg(args, "auto_prompt_path", os.path.join(SCRIPT_DIR, 'tools', 'new_auto_prompt.pt')))
    new_items = []
    for line in lines:
        item = json.loads(line)
        target_wav_name = f"{save_dir}/audios/{item['idx']}.wav"
        # get prompt audio
        if "prompt_audio_path" in item:
            assert os.path.exists(item['prompt_audio_path']), f"prompt_audio_path {item['prompt_audio_path']} not found"
            assert 'auto_prompt_audio_type' not in item, f"auto_prompt_audio_type and prompt_audio_path cannot be used together"
            with torch.no_grad():
                pmt_wav, vocal_wav, bgm_wav = separator.run(item['prompt_audio_path'])
            item['raw_pmt_wav'] = pmt_wav
            item['raw_vocal_wav'] = vocal_wav
            item['raw_bgm_wav'] = bgm_wav
            if pmt_wav.dim() == 2:
                pmt_wav = pmt_wav[None]
            if pmt_wav.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            pmt_wav = list(pmt_wav)
            if vocal_wav.dim() == 2:
                vocal_wav = vocal_wav[None]
            if vocal_wav.dim() != 3:
                raise ValueError("Vocal wavs should have a shape [B, C, T].")
            vocal_wav = list(vocal_wav)
            if bgm_wav.dim() == 2:
                bgm_wav = bgm_wav[None]
            if bgm_wav.dim() != 3:
                raise ValueError("BGM wavs should have a shape [B, C, T].")
            bgm_wav = list(bgm_wav)
            if type(pmt_wav) == list:
                pmt_wav = torch.stack(pmt_wav, dim=0)
            if type(vocal_wav) == list:
                vocal_wav = torch.stack(vocal_wav, dim=0)
            if type(bgm_wav) == list:
                bgm_wav = torch.stack(bgm_wav, dim=0)
            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(pmt_wav.cuda())
            melody_is_wav = False
        elif "auto_prompt_audio_type" in item:
            assert item["auto_prompt_audio_type"] in auto_prompt_type, f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
            lang = check_language_by_text(item['gt_lyric'])
            prompt_token = auto_prompt[item["auto_prompt_audio_type"]][lang][np.random.randint(0, len(auto_prompt[item["auto_prompt_audio_type"]][lang]))]
            pmt_wav = prompt_token[:,[0],:]
            vocal_wav = prompt_token[:,[1],:]
            bgm_wav = prompt_token[:,[2],:]
            melody_is_wav = False
        else:
            pmt_wav = None
            vocal_wav = None
            bgm_wav = None
            melody_is_wav = True
        item['pmt_wav'] = pmt_wav
        item['vocal_wav'] = vocal_wav
        item['bgm_wav'] = bgm_wav
        item['melody_is_wav'] = melody_is_wav
        item["idx"] = f"{item['idx']}"
        item["wav_path"] = target_wav_name
        new_items.append(item)

    if use_audio_tokenizer:
        del audio_tokenizer
        del separator

    torch.cuda.empty_cache()
    
    if "audio_tokenizer_checkpoint_sep" in cfg.keys() and use_audio_tokenizer:
        seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg) 
    else:
        seperate_tokenizer = None
    
    if seperate_tokenizer is not None:
        seperate_tokenizer = seperate_tokenizer.eval().cuda()

    for item in new_items:
        if "prompt_audio_path" in item:
            with torch.no_grad():
                vocal_wav, bgm_wav = seperate_tokenizer.encode(item['vocal_wav'].cuda(), item['bgm_wav'].cuda())
            item['vocal_wav'] = vocal_wav
            item['bgm_wav'] = bgm_wav

    if use_audio_tokenizer:
        del seperate_tokenizer

    torch.cuda.empty_cache()

    # Define model or load pretrained model
    audiolm = builders.get_lm_model(cfg, version=version)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
    audiolm.load_state_dict(audiolm_state_dict, strict=False)
    audiolm = audiolm.eval()

    offload_audiolm = True if 'offload' in cfg.keys() and 'audiolm' in cfg.offload else False
    if offload_audiolm:
        from codeclm.utils.offload_profiler import OffloadProfiler, OffloadParamParse
        audiolm_offload_param = OffloadParamParse.parse_config(audiolm, cfg.offload.audiolm)
        audiolm_offload_param.show()
        offload_profiler = OffloadProfiler(device_index=0, **(audiolm_offload_param.init_param_dict()))
        offload_profiler.offload_layer(**(audiolm_offload_param.offload_layer_param_dict()))
        offload_profiler.clean_cache_wrapper(**(audiolm_offload_param.clean_cache_param_dict()))
    else:
        audiolm = audiolm.cuda().to(torch.float16)

    model = CodecLM(name = "tmp",
        lm = audiolm,
        audiotokenizer = None,
        max_duration = max_duration,
        seperate_tokenizer = None,
    )
    
    gen_params = resolve_generation_params(args, max_duration, low_mem=True)
    model.set_generation_params(**gen_params)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "/audios", exist_ok=True)
    os.makedirs(save_dir + "/jsonl", exist_ok=True)

    
    for item in new_items:
        lyric = item["gt_lyric"]
        if version == 'v1':
            descriptions = item["descriptions"].lower() if "descriptions" in item else None
        else:
            if gen_type == 'bgm':
                descriptions = '[Musicality-very-high]' + ', ' + '[Pure-Music]' + ', ' + item["descriptions"].lower() if "descriptions" in item else '.'
            else:
                descriptions = item["descriptions"].lower() if "descriptions" in item else '.'
                descriptions = '[Musicality-very-high]' + ', ' + descriptions
        pmt_wav = item['pmt_wav']
        vocal_wav = item['vocal_wav']
        bgm_wav = item['bgm_wav']
        melody_is_wav = item['melody_is_wav']
            
        generate_inp = {
            'lyrics': [lyric.replace("  ", " ")] if gen_type != 'bgm' else '.',
            'descriptions': [descriptions],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = model.generate(**generate_inp, return_tokens=True)
                if offload_audiolm:
                    offload_profiler.reset_empty_cache_mem_line()
        item['tokens'] = tokens
    if offload_audiolm:
        offload_profiler.stop()
        del offload_profiler
        del audiolm_offload_param
    del model
    audiolm = audiolm.cpu()
    del audiolm
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    seperate_tokenizer = builders.get_audio_tokenizer_model_cpu(cfg.audio_tokenizer_checkpoint_sep, cfg)
    device = "cuda:0"
    seperate_tokenizer.model.device = device
    seperate_tokenizer.model.vae = seperate_tokenizer.model.vae.to(device)
    seperate_tokenizer.model.model.device = torch.device(device)
    seperate_tokenizer = seperate_tokenizer.eval()

    # offload_wav_tokenizer_diffusion =  True if 'offload' in cfg.keys() and 'wav_tokenizer_diffusion' in cfg.offload else False
    offload_wav_tokenizer_diffusion =  False
    if offload_wav_tokenizer_diffusion:
        sep_offload_param = OffloadParamParse.parse_config(seperate_tokenizer, cfg.offload.wav_tokenizer_diffusion)
        sep_offload_param.show()
        sep_offload_profiler = OffloadProfiler(device_index=0, **(sep_offload_param.init_param_dict()))
        sep_offload_profiler.offload_layer(**(sep_offload_param.offload_layer_param_dict()))
        sep_offload_profiler.clean_cache_wrapper(**(sep_offload_param.clean_cache_param_dict()))
    else:
        seperate_tokenizer.model.model = seperate_tokenizer.model.model.to(device)

    model = CodecLM(name = "tmp",
        lm = None,
        audiotokenizer = None,
        max_duration = max_duration,
        seperate_tokenizer = seperate_tokenizer,
    )

    for item in new_items:
        with torch.no_grad():
            if 'raw_pmt_wav' in item:
                if gen_type == 'separate':
                    wav_seperate = model.generate_audio(item['tokens'], item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='mixed')
                    wav_vocal = model.generate_audio(item['tokens'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='vocal')
                    wav_bgm = model.generate_audio(item['tokens'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='bgm')
                elif gen_type == 'mixed':
                    wav_seperate = model.generate_audio(item['tokens'], item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type=gen_type)
                else:
                    wav_seperate = model.generate_audio(item['tokens'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type=gen_type)
                del item['raw_pmt_wav']
                del item['raw_vocal_wav']
                del item['raw_bgm_wav']
            else:
                if gen_type == 'separate':
                    wav_vocal = model.generate_audio(item['tokens'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='vocal')
                    wav_bgm = model.generate_audio(item['tokens'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='bgm')
                    wav_seperate = model.generate_audio(item['tokens'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type='mixed')
                else:
                    wav_seperate = model.generate_audio(item['tokens'], chunked=True, chunk_size=get_arg(args, "chunk_size", 128), gen_type=gen_type)
        if gen_type == 'separate':
            save_pcm_wav(item['wav_path'].replace('.wav', '_vocal.wav'), wav_vocal[0], cfg.sample_rate)
            save_pcm_wav(item['wav_path'].replace('.wav', '_bgm.wav'), wav_bgm[0], cfg.sample_rate)
            save_pcm_wav(item['wav_path'], wav_seperate[0], cfg.sample_rate)
        else:
            save_pcm_wav(item['wav_path'], wav_seperate[0], cfg.sample_rate)
        del item['tokens']
        del item['pmt_wav']
        del item['vocal_wav']
        del item['bgm_wav']
        del item['melody_is_wav']
        if offload_wav_tokenizer_diffusion:
            sep_offload_profiler.reset_empty_cache_mem_line()
    
    if offload_wav_tokenizer_diffusion:
        sep_offload_profiler.stop()
    torch.cuda.empty_cache()
    src_jsonl_name = os.path.splitext(os.path.split(input_jsonl)[-1])[0]
    with open(f"{save_dir}/jsonl/{src_jsonl_name}.jsonl", "w", encoding='utf-8') as fw:
        for item in new_items:
            fw.writelines(json.dumps(item, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda: os.path.splitext(os.path.basename(sys.argv[1]))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))
    np.random.seed(int(time.time()))
    # 解析命令行参数
    args = parse_args()
    if torch.cuda.is_available():
        if args.gpu_id is not None:
            torch.cuda.set_device(int(args.gpu_id))
        device = torch.cuda.current_device()
        reserved = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        res_mem = (total - reserved) / 1024 / 1024 / 1024
        print(f"reserved memory: {res_mem}GB")

        model_name = os.path.basename(os.path.normpath(args.ckpt_path)).lower().replace('-', '_')
        if model_name == 'songgeneration_base' or model_name == 'songgeneration_base_new' or model_name == 'songgeneration_base_full':
            if res_mem > 24 and not args.low_mem:
                print("use generate")
                generate(args)
            else:
                from codeclm.utils.offload_profiler import OffloadProfiler, OffloadParamParse
                print("use generate_lowmem")
                generate_lowmem(args)
        elif model_name == 'songgeneration_large':
            if res_mem > 36 and not args.low_mem:
                print("use generate")
                generate(args)
            else:                
                print("use generate_lowmem")   
                from codeclm.utils.offload_profiler import OffloadProfiler, OffloadParamParse
                generate_lowmem(args)
        elif model_name == 'songgeneration_v2_large':
            if res_mem > 32 and not args.low_mem:
                print("use generate")
                generate(args, version = 'v2')
            else:
                print("use generate_lowmem")
                from codeclm.utils.offload_profiler import OffloadProfiler, OffloadParamParse
                generate_lowmem(args, version = 'v2')
        else:
            if not args.low_mem:
                print('use generate')
                generate(args, version = 'v2')
            else:
                print('use generate_lowmem')
                from codeclm.utils.offload_profiler import OffloadProfiler, OffloadParamParse
                generate_lowmem(args, version = 'v2')
    else:
        print("CUDA is not available")
        exit()

