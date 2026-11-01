# Solo Transcriber

Play along with real recordings. Solo Transcriber lets you pick the solo section
of a track, separates it into stems, and turns the stem carrying the saxophone
into a transposed part you can read in MuseScore or import into a DAW.

## How it works

```
audio file
   |
   v
decode + waveform preview
   |
   v
choose the solo section (start / end)
   |
   v
trim  ->  Demucs (htdemucs_6s)  ->  vocals / drums / bass / guitar / piano / other
   |                                  (the sax is usually in "other")
   v
stem audition + selection + your sax type
   |
   v
CREPE melody tracking (default) or Basic Pitch
   |
   v
note events in the sax's concert range
   |
   v
music21  ->  quantized score, key + tempo, MusicXML / MIDI
   |
   v
browser preview (OSMD) + MuseScore export
```

Notes are stored once at concert pitch. Instrument, tempo and time signature are
applied at engraving time, so switching between alto, tenor, bari, soprano or
concert pitch does not re-run the models.

## Requirements

- macOS on Apple Silicon (developed for an M1 Max, 64 GB)
- Python 3.11 (uv installs it automatically)
- Node 20+ and npm
- ffmpeg (`brew install ffmpeg`)

## Setup

```bash
brew install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --extra ml --extra dev
(cd frontend && npm install)
```

## Run

```bash
./scripts/dev.sh
```

Open http://localhost:5173. The API docs are at http://localhost:8000/docs.

The first separation downloads the Demucs weights (a few hundred MB). On Apple
Silicon the backend tries the Metal (`mps`) device first and falls back to CPU.
Basic Pitch uses its bundled CoreML model on macOS.

### Demo mode

No models, no problem. This runs the full upload → stems → score flow with
synthetic notes, useful for UI work:

```bash
ST_MOCK=1 uv run uvicorn app.main:app --app-dir backend --reload --port 8000
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `ST_DATA_DIR` | `./data` | Uploads, stems, scores, notes |
| `ST_MOCK` | off | Use the fake separation/transcription path |
| `ST_DEMUCS_MODEL` | `htdemucs_6s` | Any Demucs model name |
| `ST_DEVICE` | `auto` | `auto`, `mps`, `cuda` or `cpu` |
| `ST_CREPE_MODEL` | `tiny` | `tiny` (fast) or `full` (slower, more accurate) |
| `ST_MAX_UPLOAD_MB` | `300` | Upload limit |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/jobs` | Upload audio, decode it, build the waveform preview |
| `GET` | `/api/jobs` | List saved jobs, newest first |
| `GET` | `/api/jobs/{id}` | Poll status, section, stems, transcription |
| `DELETE` | `/api/jobs/{id}` | Forget a job and delete its saved files |
| `POST` | `/api/jobs/{id}/separate` | Trim to `{start, end}` seconds and separate stems |
| `POST` | `/api/jobs/{id}/transcribe` | Transcribe `other`, another stem, or `original` |
| `GET` | `/api/jobs/{id}/score` | Render `musicxml` or `midi` for an instrument |

The transcribe body takes `stem`, `instrument` (`alto|tenor|bari|soprano|concert`)
and `mode` (`melody|basic_pitch`). Score query parameters: `instrument`,
`time_signature=4/4|3/4|2/4|6/8`, `bpm`, `format=musicxml|midi`.

## Accuracy and limits

- Separation is the weak link. Sax lines often share the `other` stem with
  guitars, keys or synths, so bleed is normal. Audition everything; a solo
  recording can be transcribed directly with "original mix".
- Select only the solo section first. It makes separation much faster and
  stops long intros, outros and other solos from confusing the transcription.
- Auditioning uses the converted WAV directly for normal tracks, so the section
  picker is lossless. Only files over ~10 minutes get a 320 kbps MP3 audition
  copy. Separation and transcription always use the WAV.
- **Melody mode** (default) uses CREPE to follow the dominant single line and
  limits detection to your sax's concert range. Note attacks start a new note
  even when the pitch is unchanged, so repeated notes are preserved, and the
  pitch of each note is the median over its frames, which centres vibrato.
- **Basic Pitch mode** tries to transcribe every pitched note in the stem. Use it
  when the sax is buried, or as an A/B comparison.
- Octave harmonics outside the sax range are folded back into range, and short
  isolated octave spikes are snapped to their neighbours. Fast runs, growls and
  wide vibrato still produce wrong rhythms or notes. Clean up in MuseScore; the
  half/double tempo buttons can fix a doubling grid.
- Tempo, beats and the bar phase come from beat tracking on the whole uploaded
  track, not the sax stem. The first bar is placed on a detected downbeat, so a
  solo that enters mid-bar is engraved with the right rests instead of starting
  at the first note. The time signature is inferred from the beat accents
  (4/4, 3/4, 2/4 or 6/8) and remains editable. Swing, rubato and mid-track tempo
  changes may still need a manual BPM.
- Pitch bends are not engraved.
- Jobs are saved under `data/jobs/<id>/` (audio, stems, notes and a `job.json`
  with the section and choices). They are restored on startup and listed on the
  start screen, so you can reopen a test file without uploading and separating
  it again. Use "Forget" to delete a saved job, or remove the directory.

## Troubleshooting

`ModuleNotFoundError: No module named 'pkg_resources'` means the venv predates
the `setuptools<81` pin that Basic Pitch's `resampy` dependency needs. Fix it
without rebuilding the environment:

```bash
uv pip install 'setuptools<81'
```

Then restart the server. A fresh `uv sync --extra ml --extra dev` installs it
automatically.

`Melody mode needs torchcrepe` means the venv predates melody mode. Run:

```bash
uv sync --extra ml --extra dev
```

and restart.

## Project layout

```
backend/app/
  main.py              FastAPI routes and static audio serving
  jobs.py              job persistence, background worker, score cache
  pipeline/
    audio.py           ffmpeg decode, duration, BPM fallback
    beats.py           whole-track tempo, beat grid and meter detection
    separate.py        Demucs stem separation
    melody.py          CREPE tracking, onset-aware note segmentation
    transcribe.py      melody / Basic Pitch dispatch, range and octave repair
    notation.py        music21 score building and transposition
    notes.py           shared note ordering and duplicate removal
  evaluation.py        mir_eval reference scoring for benchmark runs
  synthetic.py         synthetic sax stems with exact note references
  report.py            self-contained HTML benchmark report
  report_template.html page template for the benchmark report
backend/tests/         notation, melody, evaluation and mock pipeline tests
frontend/src/          React UI, OSMD score viewer
scripts/dev.sh         backend + frontend in one terminal
scripts/evaluate.py    transcription accuracy benchmark
scripts/make_synthetic_benchmark.py
                       generate a repeatable regression dataset
scripts/benchmark_report.py
                       build the visual HTML report
scripts/synthetic_report.py
                       generate, score, and open the report in one command
```

## Tests

```bash
uv run --extra dev pytest
```

The API test runs in demo mode and exercises upload, separation, transcription
and score export end to end. The melody tests use synthetic f0 curves to cover
repeated notes, vibrato, onset splits and octave jumps without loading a model.

## Evaluating transcription

Put isolated sax stems next to reference scores with the same name (`take1.wav`
plus `take1.mid`, `take1.musicxml` or a `take1.json` note list) and run. To make
a reference from your own repertoire, run the app once, fix the part in
MuseScore, and export it as MIDI beside the source audio:

```bash
uv run python scripts/evaluate.py path/to/dataset --instrument alto --mode melody
```

It prints onset, pitch and full-note F-measures plus an octave-error rate and
repeated-note recall per file and a mean summary, and `--json metrics.json`
writes the same data for tracking changes. The octave rate counts matched
onsets whose pitch is off by whole octaves; the repeat recall checks whether
re-articulated notes survive segmentation. Both are hidden by the pitch
F-measure alone.

By default the script scores the raw note events. Add `--rendered` to render
the transcription through the notation pipeline and score the quantised score
a user would actually export. Quantisation, ties, bar placement and rests only
show up in that view, so run both when tuning engraving. Rendered estimates are
read back at concert pitch and mapped onto the audio timeline, so references
stay in concert pitch even when the instrument transposes.

When a change can affect notes or engraving, run both modes before and after
and put the summary lines in the PR description (the pull request template has
a `## Benchmark` section). The `Synthetic benchmark` GitHub Actions job also
runs the suite on PRs that touch the pipeline and writes the summary to the job
summary; it is informational, not a required check. Neither replaces real
recordings: the synthetic suite is a regression tripwire, and separation
changes are not exercised because the harness feeds isolated stems straight to
the transcriber.

No recordings yet? Generate a deterministic smoke suite covering repeated notes,
vibrato, chromatic runs, fast sixteenths, octave trills, scoops, bleed, wide
leaps, low and high registers, staccato, syncopation, scales, triplets, falls
and grace notes:

```bash
uv run python scripts/make_synthetic_benchmark.py benchmark/synthetic --instrument alto
uv run python scripts/evaluate.py benchmark/synthetic --instrument alto --mode melody
```

The synthetic scores are useful for comparing changes and catching regressions,
but they are not a prediction of real-world accuracy. Add a few corrected real
transcriptions to the dataset as soon as you can.

To generate the synthetic suite, score it, build the report and open it in one
command:

```bash
uv run python scripts/synthetic_report.py --instrument alto --mode melody
```

It writes `benchmark/synthetic/report.html` with an audio player per take,
reference and transcription piano rolls that follow the playhead, rendered OSMD
notation (the same library the app uses), note tables, F-measures and links to
MusicXML/MIDI for MuseScore. Use `--no-open` to skip opening the browser and
`--mock` only to smoke test the report plumbing without loading a model. The
lower-level command is still available:

```bash
uv run python scripts/benchmark_report.py benchmark/synthetic \
  --instrument alto --mode melody --open
```

The report finds reference scores beside each stem using the same `.mid`,
`.musicxml`, `.xml`, `.mxl` and `.json` lookup as `evaluate.py`.

## Roadmap

- In-app playback aligned to the score
- Half/double-time and swing quantization
- Manual note editing before export
- PDF export (MuseScore CLI or Verovio)
- A sax-focused separator fine-tune
- Packaged desktop build (Tauri wrapping this web app)
```
