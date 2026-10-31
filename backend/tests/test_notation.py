import re

from music21 import converter

from app.pipeline.notation import build_score, normalize_notes


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
