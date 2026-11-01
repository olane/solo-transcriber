#!/usr/bin/env python3
"""Generate synthetic sax stems and exact note references for benchmarking.

Run the output through ``scripts/evaluate.py`` to establish a regression
baseline for segmentation and range repairs. This is not a substitute for
recordings of real playing; it is a fast, deterministic smoke suite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.transcribe import INSTRUMENT_RANGES  # noqa: E402
from app.synthetic import (  # noqa: E402
    SAMPLE_RATE,
    build_scenarios,
    write_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        help="Directory to write <scenario>.wav and <scenario>.json pairs",
    )
    parser.add_argument(
        "--instrument", choices=tuple(INSTRUMENT_RANGES), default="alto"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE)
    parser.add_argument(
        "--only", nargs="*", help="Generate only the named scenarios"
    )
    parser.add_argument(
        "--list", action="store_true", help="List scenario names and exit"
    )
    args = parser.parse_args(argv)

    if args.list:
        for scenario in build_scenarios(args.instrument):
            print(f"{scenario.name}: {scenario.description}")
        return 0
    if args.out is None:
        parser.error("an output directory is required unless --list is used")

    try:
        manifest = write_benchmark(
            args.out,
            instrument=args.instrument,
            seed=args.seed,
            only=args.only,
            sr=args.sr,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Wrote {len(manifest['scenarios'])} scenario(s) to {args.out}")
    for entry in manifest["scenarios"]:
        print(f"  {entry['name']}: {entry['notes']} notes")
    print(f"\nScore it with: uv run python scripts/evaluate.py {args.out} --instrument {args.instrument}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
