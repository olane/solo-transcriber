from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pipeline.notes import NoteEvent

REFERENCE_SUFFIXES = (".mid", ".midi", ".musicxml", ".xml", ".mxl", ".json")

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
    "Octave_Error_Rate": 0.0,
    "Octave_Error_Count": 0.0,
    "Repeated_Note_Recall": 0.0,
    "Repeated_Note_Count": 0.0,
}

# A matched onset pair whose pitches differ by whole octaves is a classic
# monophonic-tracking failure that the pitch F-measure alone does not explain.
OCTAVE_SEMITONES = 12


def find_reference(audio: Path) -> Path | None:
    """Return the first supported reference score beside ``audio``, if any."""
    for suffix in REFERENCE_SUFFIXES:
        candidate = audio.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def normalize_notes(notes: list[NoteEvent]) -> list[NoteEvent]:
    cleaned: list[NoteEvent] = []
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


def _arrays(notes: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.array(
        [[note["start"], note["end"]] for note in notes], dtype=float
    ).reshape(-1, 2)
    pitches = np.array([midi_to_hz(note["pitch"]) for note in notes], dtype=float)
    return intervals, pitches


def _onset_match_pairs(
    reference: list[NoteEvent],
    estimate: list[NoteEvent],
    onset_tolerance: float,
) -> list[tuple[int, int]]:
    """Greedily pair reference and estimate notes by onset, ignoring pitch.

    mir_eval only matches notes whose pitch is already within tolerance, which
    hides octave mistakes. Pairing on onset alone lets the diagnostics below
    see what the pitch tracker got wrong.
    """
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for ref_index, ref_note in enumerate(reference):
        best_index = None
        best_delta = None
        for est_index, est_note in enumerate(estimate):
            if est_index in used:
                continue
            delta = abs(float(est_note["start"]) - float(ref_note["start"]))
            if delta > onset_tolerance:
                continue
            if best_delta is None or delta < best_delta:
                best_index, best_delta = est_index, delta
        if best_index is not None:
            used.add(best_index)
            pairs.append((ref_index, best_index))
    return pairs


def octave_error_stats(
    reference: list[NoteEvent],
    estimate: list[NoteEvent],
    onset_tolerance: float = 0.05,
) -> tuple[float, int, int]:
    """Return ``(rate, octave_errors, matched_onsets)`` for a note list pair."""
    reference = normalize_notes(reference)
    estimate = normalize_notes(estimate)
    pairs = _onset_match_pairs(reference, estimate, onset_tolerance)
    if not pairs:
        return 0.0, 0, 0
    errors = 0
    for ref_index, est_index in pairs:
        difference = abs(
            int(round(estimate[est_index]["pitch"]))
            - int(round(reference[ref_index]["pitch"]))
        )
        if difference and difference % OCTAVE_SEMITONES == 0:
            errors += 1
    return errors / len(pairs), errors, len(pairs)


def repeated_note_stats(
    reference: list[NoteEvent],
    estimate: list[NoteEvent],
    onset_tolerance: float = 0.05,
) -> tuple[float, int]:
    """Measure whether re-articulated notes survive segmentation.

    Returns ``(recall, repeated_notes)``. A reference attack that repeats the
    previous pitch counts as detected when an estimate with the same pitch
    starts within the onset tolerance. With no repeated notes, recall is 1.0.
    """
    reference = normalize_notes(reference)
    estimate = normalize_notes(estimate)
    repeated = [
        current
        for previous, current in zip(reference, reference[1:])
        if int(round(previous["pitch"])) == int(round(current["pitch"]))
        and float(current["start"]) > float(previous["start"])
    ]
    if not repeated:
        return 1.0, 0
    estimate_starts = [
        (int(round(item["pitch"])), float(item["start"])) for item in estimate
    ]
    hits = 0
    for note in repeated:
        pitch = int(round(note["pitch"]))
        start = float(note["start"])
        if any(
            est_pitch == pitch and abs(est_start - start) <= onset_tolerance
            for est_pitch, est_start in estimate_starts
        ):
            hits += 1
    return hits / len(repeated), len(repeated)


def evaluate_notes(
    reference: list[NoteEvent],
    estimate: list[NoteEvent],
    onset_tolerance: float = 0.05,
    pitch_tolerance_cents: float = 50.0,
) -> dict[str, float]:
    """Score estimated notes against a reference with mir_eval transcription metrics.

    In addition to the standard mir_eval metrics this reports octave mistakes
    and repeated-note recall, which the pitch F-measure can hide.
    """
    reference = normalize_notes(reference)
    estimate = normalize_notes(estimate)

    repeated_recall, repeated_count = repeated_note_stats(
        reference, estimate, onset_tolerance
    )
    octave_rate, octave_count, _ = octave_error_stats(
        reference, estimate, onset_tolerance
    )

    if not reference or not estimate:
        scores = dict(EMPTY_SCORES)
    else:
        import mir_eval

        ref_intervals, ref_pitches = _arrays(reference)
        est_intervals, est_pitches = _arrays(estimate)
        scores = {
            key: float(value)
            for key, value in mir_eval.transcription.evaluate(
                ref_intervals,
                ref_pitches,
                est_intervals,
                est_pitches,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=pitch_tolerance_cents,
                offset_ratio=0.2,
                offset_min_tolerance=0.05,
            ).items()
        }
    scores["Octave_Error_Rate"] = octave_rate
    scores["Octave_Error_Count"] = float(octave_count)
    scores["Repeated_Note_Recall"] = repeated_recall
    scores["Repeated_Note_Count"] = float(repeated_count)
    return scores


def render_estimate(
    notes: list[NoteEvent],
    bpm: float,
    instrument: str,
    time_signature: str = "4/4",
    bar_offset: float | None = None,
    fmt: str = "midi",
) -> list[NoteEvent]:
    """Render notes through the notation pipeline and read them back.

    This is the end-to-end view a user actually exports: quantisation, ties,
    bar placement and rests are all applied. Scores obtained this way can
    disagree with the raw note events, which is exactly what we want to catch.

    Two details make the result comparable with a reference in audio time:

    - MIDI is used by default because it is read back at sounding pitch. A
      MusicXML part for a transposing sax carries written pitch.
    - The score starts at the barline before the first note, so the origin is
      added back to map rendered times onto the input timeline.
    """
    import tempfile

    from .pipeline.notation import normalize_notes, render_score

    normalized = normalize_notes(notes)
    if not normalized:
        return []
    first_note = float(normalized[0]["start"])
    origin = first_note if bar_offset is None else min(float(bar_offset), first_note)

    suffix = ".mid" if fmt == "midi" else ".musicxml"
    with tempfile.TemporaryDirectory(prefix="st-eval-") as directory:
        path = Path(directory) / f"estimate{suffix}"
        render_score(
            normalized,
            bpm,
            instrument,
            time_signature,
            "estimate",
            fmt,
            path,
            bar_offset=bar_offset,
        )
        rendered = load_reference_notes(path)
    for item in rendered:
        item["start"] = float(item["start"]) + origin
        item["end"] = float(item["end"]) + origin
    return rendered


def _load_json_notes(path: Path) -> list[NoteEvent]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("notes", [])
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a note list or an object with a 'notes' key")
    return normalize_notes(payload)


def _load_score_notes(path: Path) -> list[NoteEvent]:
    from music21 import chord, converter, note, tempo

    score = converter.parse(str(path))
    flat = score.flatten()
    events = getattr(flat, "secondsMap", None) or []
    notes: list[NoteEvent] = []
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


def load_reference_notes(path: Path) -> list[NoteEvent]:
    """Read reference note events from JSON, MIDI or a music21-readable score."""
    if path.suffix.lower() == ".json":
        return _load_json_notes(path)
    return _load_score_notes(path)
