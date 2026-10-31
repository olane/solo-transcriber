from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .audio import detect_bpm, wav_duration

MIN_NOTE_SECONDS = 0.04
SAME_ONSET_SECONDS = 0.001
NOTE_GAP_SECONDS = 0.01


def monophonic_cleanup(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a single sax line: drop specks, dedupe attacks, trim overlaps."""
    cleaned = [
        note
        for note in notes
        if note["end"] - note["start"] >= MIN_NOTE_SECONDS
        and 0 <= int(note["pitch"]) <= 127
    ]
    cleaned.sort(key=lambda note: (note["start"], -int(note["velocity"])))

    deduped: list[dict[str, Any]] = []
    for note in cleaned:
        if deduped and note["start"] - deduped[-1]["start"] < SAME_ONSET_SECONDS:
            continue
        deduped.append(note)

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
            "pitch": int(pitch),
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


def transcribe_file(
    stem_wav: Path, mix_wav: Path, mock: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    duration = wav_duration(stem_wav)

    if mock:
        notes = _mock_notes(duration)
        return notes, {
            "bpm": 120.0,
            "duration": duration,
            "mock": True,
            "first_note": notes[0]["start"] if notes else 0.0,
        }

    if importlib.util.find_spec("basic_pitch") is None:
        raise RuntimeError(
            "Basic Pitch is not installed. Install the ML extras with: uv sync --extra ml"
        )

    try:
        from basic_pitch.inference import predict
    except Exception as exc:
        raise RuntimeError(
            "Basic Pitch could not load an inference backend. On macOS install "
            "coremltools (uv pip install coremltools), or install basic-pitch[onnx]."
        ) from exc

    _, _, note_events = predict(
        str(stem_wav),
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=60.0,
    )

    notes = monophonic_cleanup(note_events_to_notes(note_events))
    if not notes:
        raise RuntimeError("No notes were detected in that stem")

    bpm = detect_bpm(mix_wav) or 120.0
    return notes, {
        "bpm": bpm,
        "duration": duration,
        "mock": False,
        "first_note": notes[0]["start"] if notes else 0.0,
    }
