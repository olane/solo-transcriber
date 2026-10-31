import json

import numpy as np

from app.pipeline.transcribe import monophonic_cleanup, note_events_to_notes


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
