from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.ndimage import median_filter

from .notes import NoteEvent

HOP_LENGTH = 160
SAMPLE_RATE = 16000
MIN_NOTE_SECONDS = 0.06
MERGE_GAP_SECONDS = 0.05
PITCH_CONFIRM_FRAMES = 2
PITCH_CENTER_FRAMES = 5
PITCH_CHANGE_LOOKAHEAD_FRAMES = 6
PITCH_TOLERANCE_SEMITONES = 0.75
ONSET_DELTA = 0.1
ONSET_WAIT_FRAMES = 2
ARTICULATION_WINDOW_FRAMES = 2
ARTICULATION_BASELINE_FRAMES = 8
ARTICULATION_DIP_RATIO = 0.92


def _confirmed_onsets(onsets: Any, periodicity: np.ndarray) -> np.ndarray:
    """Keep only onsets backed by a dip in CREPE's confidence.

    A broadband onset detector also fires on accompaniment bleed inside a
    sustained note. A real re-articulation briefly destabilises the pitch
    estimate, so require a local periodicity dip before splitting a note.
    Onsets that start a note from silence are unaffected: ``_segment_notes``
    starts those notes without needing the onset.
    """
    if onsets is None or len(periodicity) == 0:
        return np.array([], dtype=int)
    values = np.asarray(onsets)
    if values.size == 0:
        return np.array([], dtype=int)
    if values.dtype == bool:
        values = np.flatnonzero(values)
    frames = values.astype(int)
    frames = frames[(frames >= 0) & (frames < len(periodicity))]
    confirmed: list[int] = []
    for frame in frames:
        inner = periodicity[
            max(0, frame - ARTICULATION_WINDOW_FRAMES) : frame
            + ARTICULATION_WINDOW_FRAMES
            + 1
        ]
        surrounding = periodicity[
            max(0, frame - ARTICULATION_BASELINE_FRAMES) : frame
            + ARTICULATION_BASELINE_FRAMES
            + 1
        ]
        if inner.size == 0 or surrounding.size == 0:
            continue
        baseline = float(np.percentile(surrounding, 75))
        if baseline <= 0:
            continue
        if float(np.min(inner)) <= baseline * ARTICULATION_DIP_RATIO:
            confirmed.append(frame)
    return np.asarray(confirmed, dtype=int)


def _onset_mask(onsets: Any, length: int) -> np.ndarray:
    """Normalise frame indices or a boolean mask into a mask of ``length``."""
    mask = np.zeros(length, dtype=bool)
    if onsets is None:
        return mask
    values = np.asarray(onsets)
    if values.size == 0:
        return mask
    if values.dtype == bool:
        width = min(length, values.size)
        mask[:width] = values[:width]
        return mask
    indices = values.astype(int)
    indices = indices[(indices >= 0) & (indices < length)]
    mask[indices] = True
    return mask


def _pitch_center(values: np.ndarray, start: int, end: int, fallback: float) -> float:
    window = values[start:end]
    window = window[np.isfinite(window)]
    if window.size == 0:
        return fallback
    return float(np.median(window))


def _pitch_changes_soon(smoothed: np.ndarray, index: int, pitch: float) -> bool:
    """Detect a sustained pitch change just after an onset frame."""
    window = smoothed[index + 1 : index + PITCH_CHANGE_LOOKAHEAD_FRAMES + 1]
    window = window[np.isfinite(window)]
    run = 0
    for value in window:
        if abs(float(value) - pitch) > PITCH_TOLERANCE_SEMITONES:
            run += 1
            if run >= PITCH_CONFIRM_FRAMES:
                return True
        else:
            run = 0
    return False


def _segment_notes(
    f0: np.ndarray,
    periodicity: np.ndarray,
    times: np.ndarray,
    onsets: Any = None,
    threshold: float = 0.5,
) -> list[tuple[int, int, int]]:
    """Turn a frame-wise f0 curve into note segments.

    Onsets are first-class: a detected attack starts a new note even when the
    pitch is unchanged, so repeated notes survive. Pitch changes must persist
    for a few frames before they split a note, and the pitch of each segment is
    the median over its frames, which centres vibrato.
    """
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

    onset_mask = _onset_mask(onsets, len(smoothed))

    segments: list[list[float]] = []
    current: list[float] | None = None
    candidate: tuple[int, float] | None = None

    for index in range(len(smoothed)):
        if not voiced[index] or not np.isfinite(smoothed[index]):
            if current is not None:
                segments.append(current)
                current = None
            candidate = None
            continue

        pitch = float(smoothed[index])

        if current is None:
            center = _pitch_center(
                smoothed, index, index + PITCH_CENTER_FRAMES, pitch
            )
            current = [index, index, center]
            candidate = None
        elif onset_mask[index] and index > current[0]:
            if _pitch_changes_soon(smoothed, index, current[2]):
                # A slightly early onset right before a pitch change would
                # leave a tiny same-pitch fragment. Let the pitch change
                # create the boundary instead.
                current[1] = index
                candidate = None
            else:
                if index - 1 >= current[0]:
                    segments.append([current[0], index - 1, current[2]])
                center = _pitch_center(
                    smoothed, index, index + PITCH_CENTER_FRAMES, pitch
                )
                current = [index, index, center]
                candidate = None
        elif abs(pitch - current[2]) <= PITCH_TOLERANCE_SEMITONES:
            current[1] = index
            candidate = None
        elif (
            candidate is None
            or abs(pitch - candidate[1]) > PITCH_TOLERANCE_SEMITONES
        ):
            center = _pitch_center(
                smoothed, index, index + PITCH_CENTER_FRAMES, pitch
            )
            candidate = (index, center)
        elif index - candidate[0] >= PITCH_CONFIRM_FRAMES:
            cut = candidate[0]
            previous = [current[0], cut - 1, current[2]]
            if previous[1] >= previous[0]:
                segments.append(previous)
            current = [cut, index, candidate[1]]
            candidate = None
        else:
            current[1] = index

        if current is not None:
            current[1] = index

    if current is not None:
        segments.append(current)

    refined: list[tuple[int, int, int]] = []
    for start_frame, end_frame, _ in segments:
        start = int(start_frame)
        end = int(end_frame)
        values = smoothed[start : end + 1]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        pitch = int(np.round(np.median(values)))
        if 0 <= pitch <= 127:
            refined.append((start, end, pitch))

    merged: list[tuple[int, int, int]] = []
    for start, end, pitch in refined:
        previous_end = merged[-1][1] if merged else -1
        if (
            merged
            and pitch == merged[-1][2]
            and times[start] - times[previous_end] < MERGE_GAP_SECONDS
            and not onset_mask[previous_end + 1 : start + 1].any()
        ):
            merged[-1] = (merged[-1][0], end, pitch)
        else:
            merged.append((start, end, pitch))
    return merged


def _detect_onsets(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Detect note attacks inside the instrument's frequency range."""
    if audio.size == 0:
        return np.array([], dtype=int)
    envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        fmin=float(fmin),
        fmax=float(fmax),
    )
    if envelope.size == 0:
        return np.array([], dtype=int)
    scale = float(np.percentile(envelope, 99))
    if scale <= 0:
        return np.array([], dtype=int)
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope / scale,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        backtrack=True,
        wait=ONSET_WAIT_FRAMES,
        delta=ONSET_DELTA,
    )
    return np.asarray(frames, dtype=int)


def transcribe_melody(
    audio_path: Path,
    fmin: float,
    fmax: float,
    model: str = "tiny",
) -> list[NoteEvent]:
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
    onsets = _detect_onsets(audio, sr, HOP_LENGTH, fmin, fmax)
    onsets = _confirmed_onsets(onsets, periodicity)
    segments = _segment_notes(f0, periodicity, times, onsets)

    rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=HOP_LENGTH)[0]
    loudness = float(np.percentile(rms, 90)) or 1.0
    notes: list[NoteEvent] = []
    for start_frame, end_frame, pitch in segments:
        start = float(times[start_frame])
        end = float(times[end_frame] + HOP_LENGTH / sr)
        if end - start < MIN_NOTE_SECONDS:
            continue
        window = rms[start_frame : end_frame + 1]
        level = float(np.mean(window)) if window.size else 0.0
        velocity = int(np.clip(55 + 65 * level / loudness, 40, 120))
        notes.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "pitch": int(pitch),
                "velocity": velocity,
            }
        )

    if not notes:
        raise RuntimeError("No melody was detected in that stem")
    return notes
