from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import soundfile as sf


def ffmpeg_binary() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def to_wav(src: Path, dst: Path) -> Path:
    """Decode any input audio into a stereo 44.1 kHz PCM WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_binary(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Could not decode audio: {result.stderr.strip()[-500:]}")
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError("Audio conversion produced an empty file")
    return dst


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return round(info.frames / info.samplerate, 2)


def detect_bpm(path: Path) -> float | None:
    import librosa
    import numpy as np

    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        if y.size == 0:
            return None
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        value = float(np.atleast_1d(tempo)[0])
    except Exception:
        return None
    if not 30.0 <= value <= 300.0:
        return None
    return round(value, 1)
