from __future__ import annotations

from typing import NotRequired, TypedDict

SAME_ONSET_SECONDS = 0.001


class NoteEvent(TypedDict):
    start: float
    end: float
    pitch: float
    velocity: NotRequired[int]


def sort_and_dedupe(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Order note events by onset and keep the strongest attack per onset."""
    ordered = sorted(
        notes,
        key=lambda note: (
            float(note["start"]),
            -int(note.get("velocity", 80)),
        ),
    )
    deduped: list[NoteEvent] = []
    for note in ordered:
        if (
            deduped
            and abs(float(note["start"]) - float(deduped[-1]["start"]))
            < SAME_ONSET_SECONDS
        ):
            continue
        deduped.append(note)
    return deduped
