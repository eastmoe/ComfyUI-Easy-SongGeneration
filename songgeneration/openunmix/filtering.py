def wiener(*args, **kwargs):
    raise RuntimeError(
        "openunmix is only required for Demucs non-CAC Wiener filtering. "
        "The bundled htdemucs runtime uses cac=True; install openunmix to use Wiener filtering."
    )
