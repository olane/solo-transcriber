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
        y, sr = librosa.load(str(path), sr=22050, mono=True, duration=180)
        if y.size == 0:
            return None
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        value = float(np.atleast_1d(tempo)[0])
    except Exception:
        return None
    if not 30.0 <= value <= 300.0:
        return None
    return round(value, 1)


def trim_wav(src: Path, dst: Path, start: float, end: float) -> Path:
    """Cut [start, end) seconds out of a WAV into a new stereo 44.1 kHz WAV."""
    if end <= start:
        raise ValueError("The section end must be after its start")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if start <= 0.01 and end >= wav_duration(src) - 0.01:
        shutil.copyfile(src, dst)
        return dst
    cmd = [
        ffmpeg_binary(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{end - start:.3f}",
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
        raise RuntimeError(f"Could not trim audio: {result.stderr.strip()[-500:]}")
    return dst


def make_preview(src: Path, dst: Path) -> Path:
    """High-quality MP3 used for auditioning while choosing the section."""
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
        "-b:a",
        "320k",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Could not build the waveform preview: {result.stderr.strip()[-500:]}")
    return dst
