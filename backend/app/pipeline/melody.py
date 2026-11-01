from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.ndimage import median_filter

HOP_LENGTH = 160
SAMPLE_RATE = 16000
MIN_NOTE_SECONDS = 0.06
PITCH_CONFIRM_FRAMES = 2
MERGE_GAP_SECONDS = 0.05


def _segment_notes(
    f0: np.ndarray,
    periodicity: np.ndarray,
    times: np.ndarray,
    threshold: float = 0.5,
) -> list[tuple[int, int, int]]:
    voiced = (periodicity > threshold) & (f0 > 0) & np.isfinite(f0)
    if voiced.sum() < 3:
        return []

    midi = 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-9) / 440.0)
    smoothed = midi.copy()
    smoothed[~voiced] = np.nan
    indices = np.arange(len(smoothed))
    good = ~np.isnan(smoothed)
    smoothed = np.interp(indices, indices[good], smoothed[good])
    smoothed = median_filter(smoothed, size=5)
    smoothed[~voiced] = np.nan
    rounded = np.round(smoothed)

    segments: list[list[int]] = []
    current: list[int] | None = None
    candidate: tuple[int, int] | None = None

    for index in range(len(rounded)):
        if not voiced[index] or np.isnan(rounded[index]):
            if current is not None:
                segments.append(current)
                current = None
            candidate = None
            continue

        pitch = int(rounded[index])
        if current is None:
            current = [index, index, pitch]
            candidate = None
        elif pitch == current[2]:
            current[1] = index
            candidate = None
        elif candidate is None or pitch != candidate[0]:
            candidate = (pitch, index)
        elif index - candidate[1] >= PITCH_CONFIRM_FRAMES:
            cut = candidate[1]
            previous = [current[0], cut - 1, current[2]]
            if previous[1] >= previous[0]:
                segments.append(previous)
            current = [cut, index, pitch]
            candidate = None
        else:
            current[1] = index
        if current is not None:
            current[1] = index

    if current is not None:
        segments.append(current)

    merged: list[list[int]] = []
    for segment in segments:
        if (
            merged
            and segment[2] == merged[-1][2]
            and times[segment[0]] - times[merged[-1][1]] < MERGE_GAP_SECONDS
        ):
            merged[-1][1] = segment[1]
        else:
            merged.append(segment)
    return [(a, b, p) for a, b, p in merged]


def transcribe_melody(
    audio_path: Path,
    fmin: float,
    fmax: float,
    model: str = "tiny",
) -> list[dict[str, Any]]:
    """Follow the dominant monophonic line with CREPE and segment it into notes."""
    import torch
    import torchcrepe

    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise RuntimeError("The selected stem is empty")

    tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    f0, periodicity = torchcrepe.predict(
        tensor,
        sr,
        hop_length=HOP_LENGTH,
        fmin=float(fmin),
        fmax=float(fmax),
        model=model,
        batch_size=1024,
        device="cpu",
        return_periodicity=True,
    )
    f0 = f0.squeeze(0).numpy()
    periodicity = periodicity.squeeze(0).numpy()
    times = np.arange(len(f0)) * HOP_LENGTH / sr
    segments = _segment_notes(f0, periodicity, times)

    rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=HOP_LENGTH)[0]
    loudness = float(np.percentile(rms, 90)) or 1.0
    notes: list[dict[str, Any]] = []
    for start_frame, end_frame, pitch in segments:
        start = float(times[start_frame])
        end = float(times[end_frame] + HOP_LENGTH / sr)
        if end - start < MIN_NOTE_SECONDS:
            continue
        level = float(np.mean(rms[start_frame : end_frame + 1]))
        velocity = int(np.clip(55 + 65 * level / loudness, 40, 120))
        notes.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "pitch": int(pitch),
                "velocity": velocity,
                "bend": None,
            }
        )

    if not notes:
        raise RuntimeError("No melody was detected in that stem")
    return notes
