import json

import numpy as np
import pytest

from app.pipeline.transcribe import (
    enforce_instrument_range,
    monophonic_cleanup,
    note_events_to_notes,
    repair_octave_jumps,
)


def note(start, end, pitch):
    return {"start": start, "end": end, "pitch": pitch, "velocity": 80, "bend": None}


def test_note_events_are_json_safe():
    events = [
        (0.0, 0.5, 60, 0.8, [np.int64(1), np.int64(2)]),
        (0.6, 1.0, 62, 0.5, None),
    ]
    notes = note_events_to_notes(events)
    assert notes[0]["bend"] == [1, 2]
    assert notes[1]["bend"] is None
    json.dumps(notes)


def test_monophonic_cleanup_keeps_one_line():
    notes = [
        {"start": 0.0, "end": 2.0, "pitch": 60, "velocity": 40, "bend": None},
        {"start": 0.2, "end": 0.6, "pitch": 64, "velocity": 100, "bend": None},
        {"start": 0.7, "end": 1.2, "pitch": 65, "velocity": 90, "bend": None},
        {"start": 3.0, "end": 3.02, "pitch": 67, "velocity": 90, "bend": None},
    ]
    cleaned = monophonic_cleanup(notes)
    assert [note["pitch"] for note in cleaned] == [60, 64, 65]
    for current, following in zip(cleaned, cleaned[1:]):
        assert current["end"] <= following["start"]


def test_note_events_round_fractional_pitch():
    notes = note_events_to_notes([(0.0, 0.5, 60.6, 0.8, None)])

    assert notes[0]["pitch"] == 61


def test_range_folding_fixes_harmonics_and_drops_bleed():
    notes = [
        note(0.0, 0.1, 93),
        note(0.2, 0.3, 37),
        note(0.4, 0.5, 30),
        note(0.6, 0.7, 60),
    ]

    ranged = enforce_instrument_range(notes, "alto")

    assert [item["pitch"] for item in ranged] == [81, 49, 60]


def test_repair_octave_jumps_snaps_isolated_spike():
    notes = [
        note(0.0, 0.2, 60),
        note(0.2, 0.3, 72),
        note(0.3, 0.5, 62),
    ]

    repaired = repair_octave_jumps(notes, "alto")

    assert [item["pitch"] for item in repaired] == [60, 60, 62]


def test_repair_octave_jumps_leaves_sustained_or_unrelated_leaps():
    sustained = [
        note(0.0, 0.2, 60),
        note(0.2, 0.8, 72),
        note(0.8, 1.0, 62),
    ]
    unrelated = [
        note(0.0, 0.2, 60),
        note(0.2, 0.3, 72),
        note(0.3, 0.5, 70),
    ]

    assert [
        item["pitch"] for item in repair_octave_jumps(sustained, "alto")
    ] == [60, 72, 62]
    assert [
        item["pitch"] for item in repair_octave_jumps(unrelated, "alto")
    ] == [60, 72, 70]


def test_repair_octave_jumps_keeps_octave_trills():
    trill = [
        note(0.0, 0.1, 60),
        note(0.1, 0.2, 72),
        note(0.2, 0.3, 60),
    ]

    assert [
        item["pitch"] for item in repair_octave_jumps(trill, "alto")
    ] == [60, 72, 60]


def test_rhythm_metadata_maps_a_whole_track_downbeat_into_the_section(tmp_path):
    from pathlib import Path

    from app.pipeline.beats import BeatGrid
    from app.pipeline.transcribe import _rhythm_metadata

    grid = BeatGrid(
        bpm=120.0,
        time_signature="4/4",
        beats=tuple(index * 0.5 for index in range(60)),
    )

    meta = _rhythm_metadata(
        Path("unused.wav"),
        first_note_seconds=1.0,
        section_start=10.2,
        beat_grid=grid,
        mock=False,
    )

    assert meta["bpm"] == 120.0
    assert meta["time_signature"] == "4/4"
    assert meta["bar_offset"] == pytest.approx(-0.2)
