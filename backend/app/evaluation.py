from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

EMPTY_SCORES = {
    "Precision": 0.0,
    "Recall": 0.0,
    "F-measure": 0.0,
    "Average_Overlap_Ratio": 0.0,
    "Precision_no_offset": 0.0,
    "Recall_no_offset": 0.0,
    "F-measure_no_offset": 0.0,
    "Average_Overlap_Ratio_no_offset": 0.0,
    "Onset_Precision": 0.0,
    "Onset_Recall": 0.0,
    "Onset_F-measure": 0.0,
    "Offset_Precision": 0.0,
    "Offset_Recall": 0.0,
    "Offset_F-measure": 0.0,
}


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def normalize_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for note in notes:
        try:
            start = float(note["start"])
            end = float(note["end"])
            pitch = float(note["pitch"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        cleaned.append({"start": start, "end": end, "pitch": pitch})
    cleaned.sort(key=lambda item: (item["start"], item["pitch"]))
    return cleaned


def _arrays(notes: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.array(
        [[note["start"], note["end"]] for note in notes], dtype=float
    ).reshape(-1, 2)
    pitches = np.array([midi_to_hz(note["pitch"]) for note in notes], dtype=float)
    return intervals, pitches


def evaluate_notes(
    reference: list[dict[str, Any]],
    estimate: list[dict[str, Any]],
    onset_tolerance: float = 0.05,
    pitch_tolerance_cents: float = 50.0,
) -> dict[str, float]:
    """Score estimated notes against a reference with mir_eval transcription metrics."""
    reference = normalize_notes(reference)
    estimate = normalize_notes(estimate)
    if not reference or not estimate:
        return dict(EMPTY_SCORES)

    import mir_eval

    ref_intervals, ref_pitches = _arrays(reference)
    est_intervals, est_pitches = _arrays(estimate)
    scores = mir_eval.transcription.evaluate(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance_cents,
        offset_ratio=0.2,
        offset_min_tolerance=0.05,
    )
    return {key: float(value) for key, value in scores.items()}


def _load_json_notes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("notes", [])
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a note list or an object with a 'notes' key")
    return normalize_notes(payload)


def _load_score_notes(path: Path) -> list[dict[str, Any]]:
    from music21 import chord, converter, note, tempo

    score = converter.parse(str(path))
    flat = score.flatten()
    events = getattr(flat, "secondsMap", None) or []
    notes: list[dict[str, Any]] = []
    for entry in events:
        element = entry.get("element")
        if isinstance(element, note.Note):
            pitches = [element.pitch.midi]
        elif isinstance(element, chord.Chord):
            pitches = [pitch.midi for pitch in element.pitches]
        else:
            continue
        start = float(entry.get("offsetSeconds", 0.0))
        end = start + float(entry.get("durationSeconds", 0.0))
        if end <= start:
            continue
        notes.append({"start": start, "end": end, "pitch": max(pitches)})
    if notes:
        return normalize_notes(notes)

    mark = flat.getContextByClass(tempo.MetronomeMark)
    bpm = float(mark.number) if mark and mark.number else 120.0
    seconds_per_quarter = 60.0 / bpm
    for element in flat.notes:
        if isinstance(element, note.Note):
            pitches = [element.pitch.midi]
        elif isinstance(element, chord.Chord):
            pitches = [pitch.midi for pitch in element.pitches]
        else:
            continue
        start = float(element.getOffsetInHierarchy(flat)) * seconds_per_quarter
        end = start + float(element.duration.quarterLength) * seconds_per_quarter
        if end <= start:
            continue
        notes.append({"start": start, "end": end, "pitch": max(pitches)})
    return normalize_notes(notes)


def load_reference_notes(path: Path) -> list[dict[str, Any]]:
    """Read reference note events from JSON, MIDI or a music21-readable score."""
    if path.suffix.lower() == ".json":
        return _load_json_notes(path)
    return _load_score_notes(path)
