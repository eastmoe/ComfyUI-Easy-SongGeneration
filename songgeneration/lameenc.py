class Encoder:
    def __init__(self):
        raise RuntimeError(
            "lameenc is only required for Demucs MP3 export. "
            "ComfyUI-Easy-SongGeneration uses WAV/FLAC paths; install lameenc to export MP3."
        )
