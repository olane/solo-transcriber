import json

import pytest

from app.evaluation import evaluate_notes, load_reference_notes

NOTES = [
    {"start": 0.0, "end": 0.5, "pitch": 60},
    {"start": 0.5, "end": 1.0, "pitch": 62},
    {"start": 1.0, "end": 1.5, "pitch": 64},
]


def test_perfect_notes_score_one():
    pytest.importorskip("mir_eval")

    scores = evaluate_notes(NOTES, NOTES)

    assert scores["Onset_F-measure"] == pytest.approx(1.0)
    assert scores["F-measure_no_offset"] == pytest.approx(1.0)
    assert scores["F-measure"] == pytest.approx(1.0)


def test_wrong_pitch_keeps_onsets_but_fails_pitch():
    pytest.importorskip("mir_eval")
    estimate = [dict(note, pitch=note["pitch"] + 1) for note in NOTES]

    scores = evaluate_notes(NOTES, estimate)

    assert scores["Onset_F-measure"] == pytest.approx(1.0)
    assert scores["F-measure_no_offset"] == pytest.approx(0.0)


def test_empty_estimate_scores_zero():
    scores = evaluate_notes(NOTES, [])

    assert scores["F-measure"] == pytest.approx(0.0)
    assert scores["Onset_F-measure"] == pytest.approx(0.0)


def test_load_json_reference(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"notes": NOTES}))

    loaded = load_reference_notes(path)

    assert [note["pitch"] for note in loaded] == [60, 62, 64]


def test_load_midi_reference(tmp_path):
    from music21 import meter, note, stream

    part = stream.Part()
    part.append(meter.TimeSignature("4/4"))
    for pitch, offset in ((60, 0.0), (62, 1.0), (64, 2.0)):
        item = note.Note(pitch)
        item.quarterLength = 1.0
        part.insert(offset, item)
    score = stream.Score([part])
    path = tmp_path / "reference.mid"
    score.write("midi", fp=str(path))

    loaded = load_reference_notes(path)

    assert [item["pitch"] for item in loaded] == [60, 62, 64]
    assert loaded[1]["start"] == pytest.approx(0.5, abs=0.02)
    assert loaded[2]["start"] == pytest.approx(1.0, abs=0.02)
