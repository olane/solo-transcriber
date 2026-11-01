from __future__ import annotations

from typing import Any

SAME_ONSET_SECONDS = 0.001


def sort_and_dedupe(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order note events by onset and keep the strongest attack per onset."""
    ordered = sorted(
        notes,
        key=lambda note: (
            float(note["start"]),
            -int(note.get("velocity", 80)),
        ),
    )
    deduped: list[dict[str, Any]] = []
    for note in ordered:
        if (
            deduped
            and abs(float(note["start"]) - float(deduped[-1]["start"]))
            < SAME_ONSET_SECONDS
        ):
            continue
        deduped.append(note)
    return deduped
