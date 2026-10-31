# Solo Transcriber

Play along with real recordings. Solo Transcriber separates a track into stems,
lets you pick the one carrying the saxophone, and turns that stem into a
transposed part you can read in MuseScore or import into a DAW.

## How it works

```
audio file
   |
   v
Demucs (htdemucs_6s)  ->  vocals / drums / bass / guitar / piano / other
   |                         (the sax is usually in "other")
   v
stem audition + selection
   |
   v
Basic Pitch  ->  note events (concert pitch)
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
| `ST_MAX_UPLOAD_MB` | `300` | Upload limit |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/jobs` | Upload audio and start separation |
| `GET` | `/api/jobs/{id}` | Poll status, stems, transcription |
| `POST` | `/api/jobs/{id}/transcribe` | Transcribe `other`, another stem, or `original` |
| `GET` | `/api/jobs/{id}/score` | Render `musicxml` or `midi` for an instrument |

Score query parameters: `instrument=alto|tenor|bari|soprano|concert`,
`time_signature=4/4|3/4|2/4|6/8`, `bpm`, `format=musicxml|midi`.

## Accuracy and limits

- Separation is the weak link. Sax lines often share the `other` stem with
  guitars, keys or synths, so bleed is normal. Audition everything; a solo
  recording can be transcribed directly with "original mix".
- Basic Pitch is instrument-agnostic and monophonic-friendly, but fast runs,
  growls and vibrato produce wrong rhythms or octaves. Clean up in MuseScore.
- Tempo comes from beat tracking on the mix; swing and rubato may need a manual
  BPM. Time signature defaults to 4/4.
- Pitch bends are detected but not engraved.
- The job store is in memory; restarting the server forgets jobs (files stay in
  `data/`).

## Project layout

```
backend/app/
  main.py              FastAPI routes and static audio serving
  jobs.py              in-memory jobs, background worker, score cache
  pipeline/
    audio.py           ffmpeg decode, duration, BPM
    separate.py        Demucs stem separation
    transcribe.py      Basic Pitch note events, mono cleanup
    notation.py        music21 score building and transposition
backend/tests/         notation + mock pipeline tests
frontend/src/          React UI, OSMD score viewer
scripts/dev.sh         backend + frontend in one terminal
```

## Tests

```bash
uv run --extra dev pytest
```

The API test runs in demo mode and exercises upload, separation, transcription
and score export end to end.

## Roadmap

- In-app playback aligned to the score
- Half/double-time and swing quantization
- Manual note editing before export
- PDF export (MuseScore CLI or Verovio)
- A sax-focused separator fine-tune
- Packaged desktop build (Tauri wrapping this web app)
```
