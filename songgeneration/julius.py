import torchaudio


def resample_frac(wav, old_sr, new_sr, *args, **kwargs):
    if old_sr == new_sr:
        return wav
    return torchaudio.functional.resample(wav, old_sr, new_sr)
