#!/usr/bin/env python3
"""Measure transcription accuracy against reference scores.

The dataset directory should contain isolated sax stems and a reference score
with the same name, for example ``take1.wav`` + ``take1.mid``. JSON sidecars
with ``{"notes": [{"start": .., "end": .., "pitch": ..}]}`` are also accepted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.evaluation import evaluate_notes, find_reference, load_reference_notes  # noqa: E402
from app.pipeline.transcribe import (  # noqa: E402
    INSTRUMENT_RANGES,
    MODES,
    transcribe_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Directory of isolated sax stems plus matching reference scores",
    )
    parser.add_argument("--mode", choices=MODES, default="melody")
    parser.add_argument(
        "--instrument", choices=tuple(INSTRUMENT_RANGES), default="alto"
    )
    parser.add_argument(
        "--pattern", default="*.wav", help="Glob for the audio files (default: *.wav)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic notes instead of the models (plumbing smoke test)",
    )
    parser.add_argument("--json", type=Path, help="Write metrics to this JSON file")
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        parser.error(f"{args.dataset} is not a directory")
    audio_files = sorted(args.dataset.glob(args.pattern))
    if not audio_files:
        parser.error(f"No files matching {args.pattern!r} in {args.dataset}")

    rows: list[dict[str, object]] = []
    failures = 0
    for audio in audio_files:
        reference_path = find_reference(audio)
        if reference_path is None:
            print(f"skip {audio.name}: no reference score")
            continue
        try:
            reference = load_reference_notes(reference_path)
        except Exception as exc:
            failures += 1
            print(f"fail {audio.name}: could not read {reference_path.name}: {exc}")
            continue
        try:
            notes, _ = transcribe_file(
                audio,
                audio,
                mock=args.mock,
                mode=args.mode,
                instrument=args.instrument,
            )
        except Exception as exc:
            failures += 1
            print(f"fail {audio.name}: {exc}")
            continue

        scores = evaluate_notes(reference, notes)
        row: dict[str, object] = {
            "file": audio.name,
            "reference_notes": len(reference),
            "estimated_notes": len(notes),
            **scores,
        }
        rows.append(row)
        print(
            f"{audio.name}: ref={len(reference)} est={len(notes)} "
            f"onset_F={scores.get('Onset_F-measure', 0.0):.3f} "
            f"pitch_F={scores.get('F-measure_no_offset', 0.0):.3f} "
            f"full_F={scores.get('F-measure', 0.0):.3f}"
        )

    if not rows:
        print("No files were evaluated.")
        return 1

    summary = {
        "files": len(rows),
        "failures": failures,
        "mean_onset_F": sum(float(row.get("Onset_F-measure", 0.0)) for row in rows)
        / len(rows),
        "mean_pitch_F": sum(float(row.get("F-measure_no_offset", 0.0)) for row in rows)
        / len(rows),
        "mean_full_F": sum(float(row.get("F-measure", 0.0)) for row in rows)
        / len(rows),
    }
    print(
        f"\n{summary['files']} file(s): "
        f"onset F={summary['mean_onset_F']:.3f} "
        f"pitch F={summary['mean_pitch_F']:.3f} "
        f"full F={summary['mean_full_F']:.3f}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "results": rows}, indent=2))
        print(f"Wrote {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
