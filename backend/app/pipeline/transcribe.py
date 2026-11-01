from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .. import config
from .audio import detect_bpm, wav_duration
from .melody import transcribe_melody
from .notes import sort_and_dedupe

MIN_NOTE_SECONDS = 0.04
NOTE_GAP_SECONDS = 0.01
ALTISSIMO_SEMITONES = 3
RANGE_BOTTOM_MARGIN = 2
OCTAVE_LEAP_SECONDS = 0.35

# Concert-pitch MIDI ranges for each sax, with a little room for altissimo.
INSTRUMENT_RANGES = {
    "soprano": (56, 88),
    "alto": (49, 81),
    "tenor": (44, 76),
    "bari": (37, 69),
    "concert": (37, 88),
}

MODES = ("melody", "basic_pitch")


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def frequency_bounds(instrument: str) -> tuple[float, float]:
    low, high = INSTRUMENT_RANGES[instrument]
    return midi_to_hz(low - RANGE_BOTTOM_MARGIN), midi_to_hz(
        high + ALTISSIMO_SEMITONES
    )


def enforce_instrument_range(
    notes: list[dict[str, Any]], instrument: str
) -> list[dict[str, Any]]:
    """Fold octave harmonics into the sax range and drop distant bleed."""
    low, high = INSTRUMENT_RANGES[instrument]
    floor = low - RANGE_BOTTOM_MARGIN
    ceiling = high + ALTISSIMO_SEMITONES
    ranged: list[dict[str, Any]] = []
    for note in notes:
        pitch = int(note["pitch"])
        while pitch > ceiling:
            pitch -= 12
        if pitch < floor:
            pitch += 12
        if floor <= pitch <= ceiling:
            item = dict(note)
            item["pitch"] = pitch
            ranged.append(item)
    return ranged


def repair_octave_jumps(
    notes: list[dict[str, Any]], instrument: str
) -> list[dict[str, Any]]:
    """Snap a short, isolated octave spike back to its neighbours' octave."""
    low, high = INSTRUMENT_RANGES[instrument]
    repaired = [dict(note) for note in notes]
    for index in range(1, len(repaired) - 1):
        current = repaired[index]
        previous = repaired[index - 1]
        following = repaired[index + 1]
        if abs(int(previous["pitch"]) - int(following["pitch"])) > 2:
            continue
        current_length = float(current["end"]) - float(current["start"])
        neighbouring_length = min(
            float(previous["end"]) - float(previous["start"]),
            float(following["end"]) - float(following["start"]),
        )
        if current_length > OCTAVE_LEAP_SECONDS:
            continue
        if current_length > 0.75 * neighbouring_length:
            continue
        reference = (float(previous["pitch"]) + float(following["pitch"])) / 2.0
        candidate = int(current["pitch"])
        while candidate - reference > 6:
            candidate -= 12
        while reference - candidate > 6:
            candidate += 12
        if candidate != int(current["pitch"]) and low <= candidate <= high:
            current["pitch"] = candidate
    return repaired


def monophonic_cleanup(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a single sax line: drop specks, dedupe attacks, trim overlaps."""
    cleaned = [
        note
        for note in notes
        if note["end"] - note["start"] >= MIN_NOTE_SECONDS
        and 0 <= int(note["pitch"]) <= 127
    ]
    deduped = sort_and_dedupe(cleaned)

    for current, following in zip(deduped, deduped[1:]):
        latest_end = following["start"] - NOTE_GAP_SECONDS
        if current["end"] > latest_end:
            current["end"] = max(current["start"] + MIN_NOTE_SECONDS, latest_end)

    return [note for note in deduped if note["end"] > note["start"]]


def note_events_to_notes(note_events) -> list[dict[str, Any]]:
    return [
        {
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "pitch": int(round(float(pitch))),
            "velocity": max(1, min(127, round(float(amplitude) * 127))),
            "bend": [int(value) for value in bend] if bend else None,
        }
        for start, end, pitch, amplitude, bend in note_events
    ]


def _mock_notes(duration: float) -> list[dict[str, Any]]:
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    notes: list[dict[str, Any]] = []
    start = 0.0
    index = 0
    limit = min(duration or 8.0, 30.0)
    while start < limit:
        pitch = scale[index % len(scale)]
        notes.append(
            {
                "start": round(start, 3),
                "end": round(min(start + 0.4, limit), 3),
                "pitch": pitch,
                "velocity": 88,
                "bend": None,
            }
        )
        start += 0.5
        index += 1
        if index % 8 == 0:
            start += 0.5
    return monophonic_cleanup(notes)


def _basic_pitch_notes(path: Path, instrument: str) -> list[dict[str, Any]]:
    if importlib.util.find_spec("basic_pitch") is None:
        raise RuntimeError(
            "Basic Pitch is not installed. Install the ML extras with: uv sync --extra ml"
        )

    try:
        from basic_pitch.inference import predict
    except ModuleNotFoundError as exc:
        if exc.name == "pkg_resources":
            raise RuntimeError(
                "Basic Pitch's resampy dependency needs setuptools. Run: "
                "uv pip install 'setuptools<81'"
            ) from exc
        raise RuntimeError(
            "Basic Pitch could not load an inference backend. On macOS install "
            "coremltools (uv pip install coremltools), or install basic-pitch[onnx]."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Basic Pitch could not load an inference backend. On macOS install "
            "coremltools (uv pip install coremltools), or install basic-pitch[onnx]."
        ) from exc

    fmin, fmax = frequency_bounds(instrument)
    _, _, note_events = predict(
        str(path),
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=60.0,
        minimum_frequency=fmin,
        maximum_frequency=fmax,
    )
    return monophonic_cleanup(note_events_to_notes(note_events))


def _melody_notes(path: Path, instrument: str) -> list[dict[str, Any]]:
    if importlib.util.find_spec("torchcrepe") is None:
        raise RuntimeError(
            "Melody mode needs torchcrepe. Install the ML extras with: uv sync --extra ml"
        )
    fmin, fmax = frequency_bounds(instrument)
    try:
        return transcribe_melody(path, fmin, fmax, model=config.CREPE_MODEL)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Melody transcription failed: {exc}") from exc


def transcribe_file(
    stem_wav: Path,
    mix_wav: Path,
    mock: bool = False,
    mode: str = "melody",
    instrument: str = "alto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if instrument not in INSTRUMENT_RANGES:
        raise ValueError(f"Unknown instrument: {instrument}")
    if mode not in MODES:
        raise ValueError(f"Unknown transcription mode: {mode}")

    duration = wav_duration(stem_wav)
    if mock:
        notes = _mock_notes(duration)
    elif mode == "melody":
        notes = _melody_notes(stem_wav, instrument)
    else:
        notes = _basic_pitch_notes(stem_wav, instrument)

    notes = monophonic_cleanup(notes)
    notes = enforce_instrument_range(notes, instrument)
    notes = repair_octave_jumps(notes, instrument)
    if not notes:
        raise RuntimeError("No notes were detected in that stem")

    bpm = detect_bpm(mix_wav) or 120.0
    return notes, {
        "bpm": bpm,
        "duration": duration,
        "mock": mock,
        "first_note": notes[0]["start"] if notes else 0.0,
        "instrument": instrument,
        "mode": mode,
    }
