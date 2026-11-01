# Repository guidance

## Synthetic benchmark for logic changes

The synthetic suite is the fastest regression tripwire for changes that can
alter note events or the engraved score. It is deterministic but not a
predictor of real-world accuracy.

Run it before and after any change under `backend/app/pipeline/`,
`backend/app/evaluation.py`, `backend/app/synthetic.py`,
`backend/app/report.py`, or `scripts/evaluate.py`:

```bash
uv sync --extra ml --extra dev
uv run python scripts/make_synthetic_benchmark.py .benchmark/synthetic --instrument alto
uv run python scripts/evaluate.py .benchmark/synthetic --instrument alto --mode melody --rendered
uv run python scripts/evaluate.py .benchmark/synthetic --instrument alto --mode melody
```

If the ML extras are too heavy for the task, `--mock` still exercises the
plumbing but does not measure the models.

Put the two summary lines, plus any per-scenario change at or above 0.02, in
the PR description under `## Benchmark`. Note explicitly when a change cannot
be measured this way: separation is not exercised because the evaluator feeds
isolated stems straight to the transcriber.

Do not treat the synthetic numbers as a release gate. A few corrected real
transcriptions under `benchmark/` are the authority; add them as soon as
possible and prefer them when the synthetic and real results disagree.

## Tests

```bash
uv run --extra dev pytest
```

Run the full suite before opening a PR.
