#!/usr/bin/env python3
"""Generate the synthetic benchmark, build the HTML report, and open it."""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.transcribe import INSTRUMENT_RANGES, MODES  # noqa: E402
from app.report import build_report  # noqa: E402
from app.synthetic import write_benchmark  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/synthetic"),
        help="Benchmark directory (default: benchmark/synthetic)",
    )
    parser.add_argument(
        "--instrument", choices=tuple(INSTRUMENT_RANGES), default="alto"
    )
    parser.add_argument("--mode", choices=MODES, default="melody")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Generate only these scenarios (default: all)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, help="Only report the first N takes")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic notes instead of the model (plumbing smoke test)",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open the report in a browser"
    )
    parser.add_argument(
        "--title", default="Synthetic benchmark", help="Report title"
    )
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    try:
        manifest = write_benchmark(
            args.out,
            instrument=args.instrument,
            seed=args.seed,
            only=args.only,
        )
        report = build_report(
            args.out,
            instrument=args.instrument,
            mode=args.mode,
            mock=args.mock,
            limit=args.limit,
            title=args.title,
        )
    except Exception as exc:
        print(f"Could not build the report: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {len(manifest['scenarios'])} scenario(s) in {args.out}")
    print(f"Wrote {report}")
    if not args.no_open:
        webbrowser.open(report.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
