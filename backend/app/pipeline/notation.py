from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from music21 import instrument, key, metadata, meter, note, pitch, stream, tempo

INSTRUMENTS = {
    "concert": instrument.Saxophone,
    "soprano": instrument.SopranoSaxophone,
    "alto": instrument.AltoSaxophone,
    "tenor": instrument.TenorSaxophone,
    "bari": instrument.BaritoneSaxophone,
}

TIME_SIGNATURES = ("4/4", "3/4", "2/4", "6/8")

# Key-signature shift (in fifths) when written pitch is transposed up from concert.
KEY_SHIFTS = {"soprano": 2, "alto": 3, "tenor": 2, "bari": 3, "concert": 0}

MIN_QUARTER_LENGTH = 0.05


def _respell_part(part, target_key) -> None:
    """Rewrite diatonic note spellings into the target key's enharmonic spelling."""
    names = {p.pitchClass: p.name for p in target_key.getScale().getPitches()}
    for event in part.recurse().notes:
        if not isinstance(event, note.Note):
            continue
        name = names.get(event.pitch.pitchClass)
        if name is None:
            continue
        midi = event.pitch.midi
        octave = 4 + (midi - pitch.Pitch(f"{name}4").midi) // 12
        event.pitch = pitch.Pitch(f"{name}{octave}")


def _legal_key(part, analyzed_key, instrument_key: str):
    """Keep the written key within seven accidentals by respelling enharmonically."""
    shift = KEY_SHIFTS[instrument_key]
    written = analyzed_key.sharps + shift
    candidates = [written]
    for delta in (-12, 12):
        if -7 <= written + delta <= 7:
            candidates.append(written + delta)
    chosen = min(candidates, key=lambda value: (abs(value), value != written))
    if chosen - shift == analyzed_key.sharps:
        return analyzed_key
    target = key.KeySignature(chosen - shift).asKey(analyzed_key.mode or "major")
    _respell_part(part, target)
    return target


def normalize_notes(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = [
        dict(note)
        for note in notes
        if note.get("end", 0) > note.get("start", 0)
        and 0 <= int(note.get("pitch", -1)) <= 127
    ]
    cleaned.sort(key=lambda item: (float(item["start"]), -int(item.get("velocity", 80))))
    result: list[dict[str, Any]] = []
    for item in cleaned:
        if result and abs(float(item["start"]) - float(result[-1]["start"])) < 0.001:
            continue
        result.append(item)
    for current, following in zip(result, result[1:]):
        if float(current["end"]) > float(following["start"]):
            current["end"] = max(
                float(current["start"]) + MIN_QUARTER_LENGTH / 4,
                float(following["start"]) - 0.001,
            )
    return [item for item in result if float(item["end"]) > float(item["start"])]


def build_score(
    notes: list[dict[str, Any]],
    bpm: float,
    instrument_key: str = "alto",
    time_signature: str = "4/4",
    title: str = "Transcription",
):
    if instrument_key not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument: {instrument_key}")
    if time_signature not in TIME_SIGNATURES:
        raise ValueError(f"Unknown time signature: {time_signature}")
    if not 20.0 <= bpm <= 400.0:
        raise ValueError("Tempo must be between 20 and 400 BPM")

    normalized = normalize_notes(notes)
    if not normalized:
        raise ValueError("There are no notes to engrave")

    part = stream.Part()
    part.insert(0, INSTRUMENTS[instrument_key]())
    part.insert(0, tempo.MetronomeMark(number=round(float(bpm), 1)))
    part.insert(0, meter.TimeSignature(time_signature))

    seconds_per_quarter = 60.0 / float(bpm)
    origin = float(normalized[0]["start"])
    for item in normalized:
        event = note.Note(int(item["pitch"]))
        event.velocity = max(1, min(127, int(item.get("velocity", 80))))
        event.duration.quarterLength = max(
            MIN_QUARTER_LENGTH,
            (float(item["end"]) - float(item["start"])) / seconds_per_quarter,
        )
        offset = (float(item["start"]) - origin) / seconds_per_quarter
        part.insert(max(0.0, offset), event)

    # Notes are stored at concert pitch. music21 writes the transposed part and
    # the matching <transpose> element, so MuseScore can toggle concert pitch.
    part.atSoundingPitch = True

    try:
        key_signature = part.analyze("key")
    except Exception:
        key_signature = key.Key("C")
    key_signature = _legal_key(part, key_signature, instrument_key)
    part.insert(0, key_signature)

    part.quantize(
        quarterLengthDivisors=(4, 3),
        processOffsets=True,
        processDurations=True,
        inPlace=True,
    )
    part.makeMeasures(inPlace=True)

    score = stream.Score([part])
    score.metadata = metadata.Metadata(title=title)
    return score


def safe_title(title: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "", title).strip().replace(" ", "-")
    return (cleaned or "transcription")[:60]


def render_score(
    notes: list[dict[str, Any]],
    bpm: float,
    instrument_key: str,
    time_signature: str,
    title: str,
    fmt: str,
    out_path: Path,
) -> Path:
    score = build_score(notes, bpm, instrument_key, time_signature, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    score.write(fmt, fp=str(out_path))
    if not out_path.exists():
        raise RuntimeError(f"Music21 did not write {out_path}")
    return out_path
