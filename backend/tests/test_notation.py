import re

import pytest
from music21 import converter, tempo
from music21.note import Rest

from app.pipeline.notation import (
    _quantization_divisors,
    build_score,
    normalize_notes,
    refine_tempo,
)


NOTES = [
    {"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 90},
    {"start": 0.5, "end": 1.0, "pitch": 62, "velocity": 90},
    {"start": 1.0, "end": 1.5, "pitch": 64, "velocity": 90},
    {"start": 1.5, "end": 2.0, "pitch": 65, "velocity": 90},
]


def test_alto_part_is_transposed_and_concert_midi(tmp_path):
    xml_path = tmp_path / "alto.musicxml"
    midi_path = tmp_path / "alto.mid"

    score = build_score(NOTES, 120, "alto", "4/4", "Test")
    score.write("musicxml", fp=str(xml_path))
    score.write("midi", fp=str(midi_path))

    xml = xml_path.read_text()
    assert "<chromatic>-9</chromatic>" in xml
    assert "<fifths>3</fifths>" in xml
    assert "<sign>G</sign>" in xml
    assert "<sign>F</sign>" not in xml

    written = [n.nameWithOctave for n in converter.parse(str(xml_path)).recurse().notes]
    assert written == ["A4", "B4", "C#5", "D5"]

    sounding = [n.nameWithOctave for n in converter.parse(str(midi_path)).recurse().notes]
    assert sounding == ["C4", "D4", "E4", "F4"]


def test_tenor_and_bari_transpositions(tmp_path):
    cases = {
        "tenor": ("<chromatic>-2</chromatic>", "D5"),
        "bari": ("<chromatic>-9</chromatic>", "A5"),
    }
    for name, (chromatic_tag, first_written) in cases.items():
        path = tmp_path / f"{name}.musicxml"
        build_score(NOTES, 120, name, "4/4", "Test").write("musicxml", fp=str(path))
        xml = path.read_text()
        assert chromatic_tag in xml
        assert "<octave-change>-1</octave-change>" in xml
        written = [n.nameWithOctave for n in converter.parse(str(path)).recurse().notes]
        assert written[0] == first_written


def test_concert_score_has_no_transposition(tmp_path):
    path = tmp_path / "concert.musicxml"
    build_score(NOTES, 120, "concert", "3/4", "Test").write("musicxml", fp=str(path))
    xml = path.read_text()
    assert "<transpose>" not in xml
    assert "<beats>3</beats>" in xml


def test_key_signature_follows_transposition(tmp_path):
    b_flat_major = [
        {"start": 0.0, "end": 0.5, "pitch": 70, "velocity": 80},
        {"start": 0.5, "end": 1.0, "pitch": 72, "velocity": 80},
        {"start": 1.0, "end": 1.5, "pitch": 74, "velocity": 80},
        {"start": 1.5, "end": 2.0, "pitch": 75, "velocity": 80},
        {"start": 2.0, "end": 3.0, "pitch": 77, "velocity": 80},
    ]
    alto = tmp_path / "alto-keys.musicxml"
    concert = tmp_path / "concert-keys.musicxml"
    build_score(b_flat_major, 120, "alto", "4/4", "Test").write("musicxml", fp=str(alto))
    build_score(b_flat_major, 120, "concert", "4/4", "Test").write("musicxml", fp=str(concert))

    alto_fifths = re.search(r"<fifths>(-?\d+)</fifths>", alto.read_text())
    concert_fifths = re.search(r"<fifths>(-?\d+)</fifths>", concert.read_text())
    assert alto_fifths and alto_fifths.group(1) == "1"
    assert concert_fifths and concert_fifths.group(1) == "-2"


def test_measures_are_complete_and_crossings_are_tied():
    # At 60 BPM one second is one quarter note, so the first note runs half a
    # beat past the first 4/4 barline.
    notes = [
        {"start": 0.0, "end": 4.5, "pitch": 60, "velocity": 80},
        {"start": 5.0, "end": 6.0, "pitch": 62, "velocity": 80},
    ]
    part = build_score(notes, 60, "concert", "4/4", "Test").parts[0]
    measures = part.getElementsByClass("Measure")

    assert [m.duration.quarterLength for m in measures] == [4.0, 4.0]

    ties = [n.tie.type for n in part.recurse().notes if n.tie is not None]
    assert "start" in ties
    assert "stop" in ties


def test_final_measure_is_padded_with_rests():
    notes = [{"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 80}]
    part = build_score(notes, 60, "concert", "3/4", "Test").parts[0]
    measures = part.getElementsByClass("Measure")

    assert len(measures) == 1
    assert measures[0].duration.quarterLength == 3.0
    assert isinstance(measures[0].notesAndRests[-1], Rest)


def test_normalize_notes_clips_overlaps_and_duplicates():
    notes = [
        {"start": 0.0, "end": 1.0, "pitch": 60, "velocity": 40},
        {"start": 0.0, "end": 1.0, "pitch": 64, "velocity": 100},
        {"start": 0.5, "end": 1.0, "pitch": 62, "velocity": 90},
        {"start": 2.0, "end": 2.0, "pitch": 60, "velocity": 90},
    ]
    cleaned = normalize_notes(notes)
    assert len(cleaned) == 2
    assert cleaned[0]["pitch"] == 64
    assert cleaned[0]["end"] <= 0.5
    assert cleaned[1]["pitch"] == 62


def test_normalize_notes_fills_tiny_legato_gaps():
    notes = [
        {"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 80},
        {"start": 0.54, "end": 1.0, "pitch": 62, "velocity": 80},
    ]

    cleaned = normalize_notes(notes)

    assert cleaned[0]["end"] == 0.54


def test_perturbed_legato_run_engraves_as_eighths():
    bpm = 107.7
    jitter = (0.0, 0.01, -0.01, 0.005, -0.005, 0.01, -0.01, 0.0)
    notes = []
    for index, offset in enumerate(jitter):
        start = 0.45 + index * 0.28 + offset
        notes.append(
            {
                "start": start,
                "end": start + 0.24,
                "pitch": 61 + index,
                "velocity": 90,
            }
        )

    part = build_score(notes, bpm, "concert", "4/4", "Legato").parts[0]
    events = list(part.recurse().notes)

    assert [round(float(event.offset), 3) for event in events] == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
    ]
    assert all(
        round(float(event.duration.quarterLength), 3) == 0.5
        for event in events
    )
    assert not any(
        isinstance(event, Rest) for event in part.recurse().notesAndRests
    )


def test_triplet_run_uses_the_triplet_grid():
    bpm = 120
    step = 60.0 / bpm / 3.0
    notes = [
        {
            "start": 0.2 + index * step,
            "end": 0.2 + (index + 1) * step,
            "pitch": 60 + index,
            "velocity": 90,
        }
        for index in range(6)
    ]

    part = build_score(notes, bpm, "concert", "4/4", "Triplets").parts[0]
    events = list(part.recurse().notes)

    assert [
        round(float(event.duration.quarterLength), 3) for event in events
    ] == [0.333] * 6


def test_extreme_keys_stay_within_seven_accidentals(tmp_path):
    # Concert B major plus alto/tenor transposition would naively need 8+ sharps.
    b_major = [
        {"start": 0.0, "end": 0.5, "pitch": 71, "velocity": 80},
        {"start": 0.5, "end": 1.0, "pitch": 73, "velocity": 80},
        {"start": 1.0, "end": 1.5, "pitch": 75, "velocity": 80},
        {"start": 1.5, "end": 2.0, "pitch": 76, "velocity": 80},
        {"start": 2.0, "end": 2.5, "pitch": 78, "velocity": 80},
        {"start": 2.5, "end": 3.0, "pitch": 80, "velocity": 80},
        {"start": 3.0, "end": 3.5, "pitch": 82, "velocity": 80},
        {"start": 3.5, "end": 4.0, "pitch": 83, "velocity": 80},
    ]
    for sax in ("alto", "tenor", "bari", "soprano", "concert"):
        xml_path = tmp_path / f"{sax}-extreme.musicxml"
        midi_path = tmp_path / f"{sax}-extreme.mid"
        score = build_score(b_major, 120, sax, "4/4", "Extreme")
        score.write("musicxml", fp=str(xml_path))
        score.write("midi", fp=str(midi_path))

        fifths = int(re.search(r"<fifths>(-?\d+)</fifths>", xml_path.read_text()).group(1))
        assert -7 <= fifths <= 7

        sounding = [n.pitch.midi for n in converter.parse(str(midi_path)).recurse().notes]
        assert sounding == [71, 73, 75, 76, 78, 80, 82, 83]


def test_bar_offset_puts_the_first_note_inside_the_bar_with_a_rest():
    # At 120 BPM, 0.5 s is one quarter note. The bar starts on the downbeat,
    # so the first note is beat 2 and the bar opens with a rest.
    notes = [
        {"start": 0.5, "end": 1.0, "pitch": 60, "velocity": 80},
        {"start": 1.0, "end": 1.5, "pitch": 62, "velocity": 80},
    ]
    part = build_score(
        notes, 120, "concert", "4/4", "Aligned", bar_offset=0.0
    ).parts[0]
    events = list(part.recurse().notesAndRests)

    assert round(float(events[0].offset), 3) == 0.0
    assert isinstance(events[0], Rest)
    assert round(float(events[1].offset), 3) == 1.0
    assert part.getElementsByClass("Measure")[0].duration.quarterLength == 4.0


def test_bar_offset_never_skips_a_pickup_note():
    notes = [
        {"start": 0.9, "end": 1.0, "pitch": 60, "velocity": 80},
        {"start": 1.0, "end": 1.5, "pitch": 62, "velocity": 80},
    ]
    part = build_score(
        notes, 120, "concert", "4/4", "Pickup", bar_offset=1.0
    ).parts[0]
    first = next(part.recurse().notes)

    assert round(float(first.offset), 3) == 0.0


def _quarter_note(start):
    return {"start": start, "end": start + 0.1, "pitch": 60, "velocity": 80}


def test_refine_tempo_recovers_a_small_beat_tracking_error():
    notes = [_quarter_note(index * 0.5) for index in range(16)]

    refined = refine_tempo(notes, 118.5, 0.0)

    assert refined == pytest.approx(120.0, abs=0.6)


def test_refine_tempo_keeps_an_already_aligned_tempo():
    notes = [_quarter_note(index * 0.5) for index in range(16)]

    assert refine_tempo(notes, 120.0, 0.0) == pytest.approx(120.0)


def test_build_score_uses_the_refined_tempo():
    notes = [_quarter_note(index * 0.5) for index in range(16)]

    part = build_score(notes, 118.5, "concert", "4/4", "Refined").parts[0]
    mark = list(part.recurse().getElementsByClass(tempo.MetronomeMark))[0]

    assert mark.number == pytest.approx(120.0, abs=0.6)


def test_mixed_straight_and_triplet_onsets_allow_tuplets():
    notes = [_quarter_note(index * 0.5) for index in range(8)]
    notes += [_quarter_note(4.0 + index / 3.0) for index in range(3)]

    assert _quantization_divisors(notes, 1.0, 0.0) == (4, 3)


def test_a_mostly_straight_line_keeps_the_straight_grid():
    notes = [_quarter_note(index * 0.5 + 0.004) for index in range(12)]

    assert _quantization_divisors(notes, 1.0, 0.0) == (4,)
