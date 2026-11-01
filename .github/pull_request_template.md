## Summary

<!-- What changed and why. -->

## Test

- [ ] `uv run pytest`

## Benchmark

<!--
Required when this can change note events or the engraved score (anything under
backend/app/pipeline/, backend/app/evaluation.py, backend/app/synthetic.py,
backend/app/report.py, or scripts/evaluate.py).

  uv sync --extra ml --extra dev
  uv run python scripts/make_synthetic_benchmark.py .benchmark/synthetic --instrument alto
  uv run python scripts/evaluate.py .benchmark/synthetic --instrument alto --mode melody --rendered
  uv run python scripts/evaluate.py .benchmark/synthetic --instrument alto --mode melody

Paste both summary lines and any per-scenario change at or above 0.02. If the
change cannot be measured this way (for example separation, which the harness
bypasses by feeding isolated stems), write "n/a" and say why. These numbers are
a regression tripwire, not a release gate.
-->

| mode | onset | pitch | full |
| --- | --- | --- | --- |
| baseline | | | |
| this PR | | | |

Notes:
