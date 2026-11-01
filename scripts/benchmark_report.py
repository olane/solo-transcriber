#!/usr/bin/env python3
"""Run a benchmark directory through the pipeline and build an HTML report.

The report embeds audio players, reference vs. transcribed piano rolls, note
tables, metrics and links to MusicXML/MIDI files.
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.transcribe import INSTRUMENT_RANGES, MODES  # noqa: E402
from app.report import build_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark",
        type=Path,
        help="Directory of <name>.wav stems plus matching reference scores",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Report path (default: <benchmark>/report.html)",
    )
    parser.add_argument(
        "--instrument", choices=tuple(INSTRUMENT_RANGES), default="alto"
    )
    parser.add_argument("--mode", choices=MODES, default="melody")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic notes instead of the model (plumbing smoke test)",
    )
    parser.add_argument("--limit", type=int, help="Only report the first N takes")
    parser.add_argument(
        "--title", default="Transcription benchmark", help="Report title"
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the report in a browser"
    )
    args = parser.parse_args(argv)

    if not args.benchmark.is_dir():
        parser.error(f"{args.benchmark} is not a directory")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    try:
        report = build_report(
            args.benchmark,
            out_path=args.out,
            instrument=args.instrument,
            mode=args.mode,
            mock=args.mock,
            limit=args.limit,
            title=args.title,
        )
    except Exception as exc:
        print(f"Could not build the report: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {report}")
    print("Open the MusicXML files in MuseScore for a readable score.")
    if args.open:
        webbrowser.open(report.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
