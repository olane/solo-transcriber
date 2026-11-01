from __future__ import annotations

import html
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evaluation import evaluate_notes, load_reference_notes
from .pipeline.audio import wav_duration
from .pipeline.notation import render_score, safe_title
from .pipeline.transcribe import transcribe_file

NOTE_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
TABLE_LIMIT = 200
OSMD_CDN = (
    "https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@2.2.1/"
    "build/opensheetmusicdisplay.min.js"
)


@dataclass
class ScenarioResult:
    name: str
    audio_path: Path
    duration: float
    bpm: float | None = None
    reference: list[dict[str, Any]] = field(default_factory=list)
    predicted: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    notes_path: Path | None = None
    musicxml_path: Path | None = None
    midi_path: Path | None = None
    reference_musicxml_path: Path | None = None
    predicted_xml: str | None = None
    reference_xml: str | None = None
    error: str | None = None


def _note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _metric_class(value: float) -> str:
    if value >= 0.8:
        return "good"
    if value >= 0.5:
        return "warn"
    return "bad"


def _relative(path: Path | None, report_dir: Path) -> str | None:
    if path is None:
        return None
    return Path(os.path.relpath(path, report_dir)).as_posix()


def _read_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text()
    except OSError:
        return None


def _osmd_source(report_dir: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    local = (
        root
        / "frontend"
        / "node_modules"
        / "opensheetmusicdisplay"
        / "build"
        / "opensheetmusicdisplay.min.js"
    )
    if local.exists():
        target = report_dir / "opensheetmusicdisplay.min.js"
        try:
            if (
                not target.exists()
                or local.stat().st_mtime > target.stat().st_mtime
            ):
                shutil.copyfile(local, target)
            return target.name
        except OSError:
            pass
    return OSMD_CDN


def _pitch_bounds(reference: list[dict], predicted: list[dict]) -> tuple[int, int]:
    pitches = [int(note["pitch"]) for note in reference + predicted]
    if not pitches:
        return 48, 84
    return max(0, min(pitches) - 2), min(127, max(pitches) + 2)


def _render_notes(
    notes: list[dict[str, Any]],
    bpm: float,
    instrument: str,
    title: str,
    out_dir: Path,
    stem: str,
    time_signature: str = "4/4",
    bar_offset: float | None = None,
) -> tuple[Path | None, Path | None, Path | None]:
    notes_path = out_dir / f"{stem}.notes.json"
    notes_path.write_text(
        json.dumps(
            {
                "source": title,
                "bpm": round(float(bpm), 3),
                "time_signature": time_signature,
                "bar_offset": bar_offset,
                "notes": notes,
            },
            indent=2,
        )
    )
    musicxml_path = out_dir / f"{stem}.musicxml"
    midi_path = out_dir / f"{stem}.mid"
    try:
        render_score(
            notes,
            bpm,
            instrument,
            time_signature,
            title,
            "musicxml",
            musicxml_path,
            bar_offset=bar_offset,
        )
        render_score(
            notes,
            bpm,
            instrument,
            time_signature,
            title,
            "midi",
            midi_path,
            bar_offset=bar_offset,
        )
    except Exception:
        return notes_path, None, None
    return notes_path, musicxml_path, midi_path


def collect_results(
    benchmark_dir: Path | str,
    instrument: str = "alto",
    mode: str = "melody",
    mock: bool = False,
    limit: int | None = None,
) -> list[ScenarioResult]:
    benchmark = Path(benchmark_dir)
    audio_files = sorted(benchmark.glob("*.wav"))
    if not audio_files:
        raise ValueError(f"No .wav files in {benchmark}")
    if limit is not None:
        audio_files = audio_files[:limit]

    predictions = benchmark / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    results: list[ScenarioResult] = []

    for audio in audio_files:
        name = audio.stem
        duration = wav_duration(audio)
        reference_path = audio.with_suffix(".json")
        reference: list[dict[str, Any]] = []
        if reference_path.exists():
            try:
                reference = load_reference_notes(reference_path)
            except Exception as exc:
                results.append(
                    ScenarioResult(
                        name=name,
                        audio_path=audio,
                        duration=duration,
                        error=f"Could not read reference: {exc}",
                    )
                )
                continue

        try:
            predicted, meta = transcribe_file(
                audio,
                audio,
                mock=mock,
                mode=mode,
                instrument=instrument,
            )
        except Exception as exc:
            results.append(
                ScenarioResult(
                    name=name,
                    audio_path=audio,
                    duration=duration,
                    reference=reference,
                    error=str(exc),
                )
            )
            continue

        bpm = float(meta["bpm"])
        time_signature = str(meta.get("time_signature", "4/4"))
        bar_offset = meta.get("bar_offset")
        stem = safe_title(name)
        notes_path, musicxml_path, midi_path = _render_notes(
            predicted,
            bpm,
            instrument,
            name,
            predictions,
            stem,
            time_signature=time_signature,
            bar_offset=bar_offset,
        )
        reference_musicxml = None
        if reference:
            try:
                reference_path_out = predictions / f"{stem}.reference.musicxml"
                render_score(
                    reference,
                    bpm,
                    instrument,
                    time_signature,
                    f"{name} reference",
                    "musicxml",
                    reference_path_out,
                )
                reference_musicxml = reference_path_out
            except Exception:
                reference_musicxml = None

        results.append(
            ScenarioResult(
                name=name,
                audio_path=audio,
                duration=duration,
                bpm=bpm,
                reference=reference,
                predicted=predicted,
                metrics=evaluate_notes(reference, predicted),
                notes_path=notes_path,
                musicxml_path=musicxml_path,
                midi_path=midi_path,
                reference_musicxml_path=reference_musicxml,
                predicted_xml=_read_text(musicxml_path),
                reference_xml=_read_text(reference_musicxml),
            )
        )
    return results


def _note_table(notes: list[dict[str, Any]]) -> str:
    if not notes:
        return '<p class="muted">No notes.</p>'
    rows = []
    for note in notes[:TABLE_LIMIT]:
        rows.append(
            "<tr>"
            f"<td>{float(note['start']):.3f}</td>"
            f"<td>{float(note['end']):.3f}</td>"
            f"<td>{_note_name(int(note['pitch']))}</td>"
            f"<td>{int(note['pitch'])}</td>"
            "</tr>"
        )
    more = ""
    if len(notes) > TABLE_LIMIT:
        more = (
            f'<p class="muted">Showing {TABLE_LIMIT} of {len(notes)} notes.</p>'
        )
    return (
        '<table><thead><tr><th>Start</th><th>End</th><th>Note</th>'
        f"<th>MIDI</th></tr></thead><tbody>{''.join(rows)}</tbody></table>{more}"
    )


def _metrics_html(metrics: dict[str, float], has_reference: bool) -> str:
    if not has_reference:
        return '<span class="pill muted">No reference</span>'
    items = (
        ("Onset", metrics.get("Onset_F-measure", 0.0)),
        ("Pitch", metrics.get("F-measure_no_offset", 0.0)),
        ("Full", metrics.get("F-measure", 0.0)),
    )
    return "".join(
        f'<span class="pill {_metric_class(value)}">{label} {value:.3f}</span>'
        for label, value in items
    )


def _links_html(
    result: ScenarioResult, report_dir: Path
) -> str:
    links = []
    for label, path in (
        ("Reference MusicXML", result.reference_musicxml_path),
        ("Transcription MusicXML", result.musicxml_path),
        ("Transcription MIDI", result.midi_path),
        ("Notes JSON", result.notes_path),
    ):
        url = _relative(path, report_dir)
        if url:
            links.append(f'<a href="{html.escape(url, quote=True)}">{label}</a>')
    if not links:
        return ""
    return f'<div class="links">{"".join(links)}</div>'


def _card_html(result: ScenarioResult, index: int, report_dir: Path) -> str:
    audio_url = _relative(result.audio_path, report_dir) or ""
    metrics = _metrics_html(result.metrics, bool(result.reference))
    error = (
        f'<div class="error">{html.escape(result.error)}</div>'
        if result.error
        else ""
    )
    bpm = f"{result.bpm:.1f} BPM" if result.bpm is not None else ""
    note_counts = (
        f"{len(result.reference)} reference / {len(result.predicted)} transcribed"
    )
    if result.predicted_xml:
        default_kind = "predicted"
    elif result.reference_xml:
        default_kind = "reference"
    else:
        default_kind = "predicted"
    predicted_active = " active" if default_kind == "predicted" else ""
    reference_active = " active" if default_kind == "reference" else ""
    predicted_disabled = "" if result.predicted_xml else "disabled"
    reference_disabled = "" if result.reference_xml else "disabled"
    return f"""
    <article class="card" id="scenario-{index}">
      <header class="card-head">
        <div>
          <h2>{html.escape(result.name)}</h2>
          <p class="muted">{note_counts}{f" · {bpm}" if bpm else ""}</p>
        </div>
        <div class="metrics">{metrics}</div>
      </header>
      {error}
      <audio controls preload="metadata" data-index="{index}"
             src="{html.escape(audio_url, quote=True)}"></audio>
      <section class="notation">
        <div class="notation-head">
          <h3>Score</h3>
          <div class="score-tabs">
            <button type="button" data-index="{index}" data-score-kind="predicted"
                    class="score-tab{predicted_active}" {predicted_disabled}>Transcription</button>
            <button type="button" data-index="{index}" data-score-kind="reference"
                    class="score-tab{reference_active}" {reference_disabled}>Reference</button>
          </div>
        </div>
        <div class="score-shell" data-score-index="{index}">
          <p class="score-placeholder">Loading score…</p>
        </div>
      </section>
      <div class="rolls">
        <div>
          <h3>Reference</h3>
          <canvas class="roll" data-kind="reference" data-index="{index}"
                  aria-label="{html.escape(result.name)} reference piano roll"></canvas>
        </div>
        <div>
          <h3>Transcription</h3>
          <canvas class="roll" data-kind="predicted" data-index="{index}"
                  aria-label="{html.escape(result.name)} transcription piano roll"></canvas>
        </div>
      </div>
      <details>
        <summary>Note events</summary>
        <div class="tables">
          <div><h4>Reference</h4>{_note_table(result.reference)}</div>
          <div><h4>Transcription</h4>{_note_table(result.predicted)}</div>
        </div>
      </details>
      {_links_html(result, report_dir)}
    </article>
    """


def _nav_html(results: list[ScenarioResult]) -> str:
    links = []
    for index, result in enumerate(results):
        score = result.metrics.get("F-measure", 0.0)
        badge = "—" if not result.reference else f"{score:.2f}"
        links.append(
            f'<a href="#scenario-{index}"><span>{html.escape(result.name)}</span>'
            f'<b class="{_metric_class(score) if result.reference else "muted"}">{badge}</b></a>'
        )
    return "".join(links)


def _summary_html(results: list[ScenarioResult]) -> str:
    scored = [result for result in results if result.reference and not result.error]
    if not scored:
        return '<span class="pill muted">No scored takes</span>'
    onset = sum(result.metrics.get("Onset_F-measure", 0.0) for result in scored) / len(
        scored
    )
    pitch = sum(
        result.metrics.get("F-measure_no_offset", 0.0) for result in scored
    ) / len(scored)
    full = sum(result.metrics.get("F-measure", 0.0) for result in scored) / len(scored)
    errors = sum(1 for result in results if result.error)
    error_pill = (
        f'<span class="pill bad">{errors} error(s)</span>' if errors else ""
    )
    return "".join(
        (
            f'<span class="pill {_metric_class(onset)}">Onset F {onset:.3f}</span>',
            f'<span class="pill {_metric_class(pitch)}">Pitch F {pitch:.3f}</span>',
            f'<span class="pill {_metric_class(full)}">Full F {full:.3f}</span>',
            error_pill,
        )
    )


def _payload(results: list[ScenarioResult]) -> str:
    payload = []
    for index, result in enumerate(results):
        low, high = _pitch_bounds(result.reference, result.predicted)
        payload.append(
            {
                "index": index,
                "duration": result.duration,
                "pitchMin": low,
                "pitchMax": high,
                "reference": result.reference,
                "predicted": result.predicted,
                "referenceXml": result.reference_xml,
                "predictedXml": result.predicted_xml,
            }
        )
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0b0e13;
  --panel: #141922;
  --panel-2: #1a202b;
  --line: #263041;
  --text: #e7ebf2;
  --muted: #96a2b5;
  --accent: #5b8cff;
  --predicted: #ff9f43;
  --good: #2dd4a7;
  --warn: #f5c451;
  --bad: #ef6461;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1240px; margin: 0 auto; padding: 28px 20px 80px; }
header.top { margin-bottom: 22px; }
.eyebrow { color: var(--accent); font-size: 12px; letter-spacing: .16em; text-transform: uppercase; }
h1 { margin: 6px 0 6px; font-size: 30px; }
h2 { font-size: 20px; margin: 0; }
h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin: 0 0 8px; }
h4 { font-size: 12px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin: 0 0 8px; }
.muted { color: var(--muted); }
.layout { display: grid; grid-template-columns: 230px minmax(0, 1fr); gap: 24px; align-items: start; }
nav { position: sticky; top: 20px; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 8px; }
nav a { display: flex; justify-content: space-between; gap: 10px; padding: 8px 10px; border-radius: 7px; color: var(--text); font-size: 13px; }
nav a:hover { background: var(--panel-2); text-decoration: none; }
nav b { font-weight: 600; }
nav b.good { color: var(--good); } nav b.warn { color: var(--warn); } nav b.bad { color: var(--bad); } nav b.muted { color: var(--muted); }
.cards { display: grid; gap: 18px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px; scroll-margin-top: 16px; }
.card-head { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 12px; }
.metrics { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
.pill { display: inline-flex; align-items: center; padding: 4px 9px; border-radius: 999px; font-size: 12px; font-variant-numeric: tabular-nums; background: var(--panel-2); border: 1px solid var(--line); }
.pill.good { color: var(--good); border-color: rgba(45,212,167,.35); }
.pill.warn { color: var(--warn); border-color: rgba(245,196,81,.35); }
.pill.bad { color: var(--bad); border-color: rgba(239,100,97,.35); }
.pill.muted { color: var(--muted); }
audio { width: 100%; margin: 4px 0 14px; }
.notation { margin: 0 0 14px; }
.notation-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 8px; }
.notation-head h3 { margin: 0; }
.score-tabs { display: flex; gap: 6px; }
.score-tabs button { background: var(--panel-2); color: var(--muted); border: 1px solid var(--line); border-radius: 7px; padding: 5px 10px; font-size: 12px; cursor: pointer; }
.score-tabs button:hover { color: var(--text); }
.score-tabs button.active { color: var(--text); border-color: var(--accent); }
.score-tabs button:disabled { opacity: .4; cursor: default; }
.score-shell { background: #fbfaf7; border-radius: 10px; padding: 16px 14px; min-height: 180px; overflow-x: auto; }
.score-shell .score { color: #111; }
.score-placeholder { color: #6b7280; font-size: 13px; padding: 16px 4px; margin: 0; }
.rolls { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
canvas.roll { width: 100%; height: 180px; display: block; background: #0a0d12; border: 1px solid var(--line); border-radius: 8px; }
.error { margin: 0 0 12px; padding: 10px 12px; border: 1px solid rgba(239,100,97,.35); background: rgba(239,100,97,.08); color: var(--bad); border-radius: 8px; font-size: 13px; }
details { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 12px; }
summary { cursor: pointer; color: var(--muted); font-size: 13px; }
.tables { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 500; }
.links { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-top: 14px; font-size: 13px; }
footer { margin-top: 28px; color: var(--muted); font-size: 12px; }
@media (max-width: 860px) {
  .layout { grid-template-columns: 1fr; }
  nav { position: static; display: flex; overflow-x: auto; gap: 4px; }
  nav a { white-space: nowrap; }
  .rolls, .tables { grid-template-columns: 1fr; }
  .card-head { flex-direction: column; }
  .metrics { justify-content: flex-start; }
}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="eyebrow">Solo Transcriber</div>
    <h1>__TITLE__</h1>
    <p class="muted">__SUBTITLE__</p>
    <div class="metrics" style="justify-content:flex-start;margin-top:12px">__SUMMARY__</div>
  </header>
  <div class="layout">
    <nav aria-label="Takes">__NAV__</nav>
    <main class="cards">__CARDS__</main>
  </div>
  <footer>
    Blue is the reference; orange is the transcription. Play a take to follow
    the playhead over both piano rolls. Select a score tab to switch between the
    reference and transcribed notation.
  </footer>
</div>
<script src="__OSMD__"></script>
<script id="report-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('report-data').textContent);
const OSMD = (window.opensheetmusicdisplay && window.opensheetmusicdisplay.OpenSheetMusicDisplay)
  || window.OpenSheetMusicDisplay;
const COLORS = {
  reference: '#5b8cff',
  predicted: '#ff9f43',
  grid: 'rgba(255,255,255,.07)',
  text: 'rgba(150,162,181,.9)',
  playhead: '#e7ebf2'
};

function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}

function drawRoll(canvas, notes, duration, pitchMin, pitchMax, color, playhead) {
  const { ctx, w, h } = setup(canvas);
  ctx.clearRect(0, 0, w, h);
  const padL = 34, padR = 10, padT = 10, padB = 20;
  const plotW = Math.max(1, w - padL - padR);
  const plotH = Math.max(1, h - padT - padB);
  const span = Math.max(1, pitchMax - pitchMin);

  ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.fillStyle = COLORS.text;
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let pitch = Math.ceil(pitchMin / 12) * 12; pitch <= pitchMax; pitch += 12) {
    const y = padT + (pitchMax - pitch) / span * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    const name = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][(pitch % 12 + 12) % 12];
    ctx.fillText(name + (Math.floor(pitch / 12) - 1), 4, y + 3);
  }

  if (!notes.length) {
    ctx.fillStyle = COLORS.text;
    ctx.fillText('No notes', padL + 8, padT + 18);
    return;
  }

  const bar = Math.max(2, plotH / (span + 1) * 0.8);
  ctx.fillStyle = color;
  for (const note of notes) {
    const x = padL + Math.max(0, Math.min(1, note.start / duration)) * plotW;
    const x2 = padL + Math.max(0, Math.min(1, note.end / duration)) * plotW;
    const y = padT + (pitchMax - note.pitch) / span * plotH - bar / 2;
    ctx.fillRect(x, y, Math.max(2, x2 - x), bar);
  }

  if (playhead !== null && playhead !== undefined) {
    const x = padL + Math.max(0, Math.min(1, playhead / duration)) * plotW;
    ctx.strokeStyle = COLORS.playhead;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
  }
}

function renderScenario(index, playhead) {
  const item = DATA[index];
  if (!item) return;
  const canvases = document.querySelectorAll('canvas[data-index="' + index + '"]');
  canvases.forEach((canvas) => {
    const kind = canvas.dataset.kind;
    const color = kind === 'reference' ? COLORS.reference : COLORS.predicted;
    drawRoll(
      canvas,
      kind === 'reference' ? item.reference : item.predicted,
      item.duration,
      item.pitchMin,
      item.pitchMax,
      color,
      playhead
    );
  });
}

const scoreStates = new Map();

function renderScore(index, kind) {
  const item = DATA[index];
  const shell = document.querySelector(
    '.score-shell[data-score-index="' + index + '"]'
  );
  if (!item || !shell) return;

  document.querySelectorAll(
    'button[data-score-kind][data-index="' + index + '"]'
  ).forEach((button) => {
    button.classList.toggle('active', button.dataset.scoreKind === kind);
  });

  if (!OSMD) {
    shell.innerHTML = '<p class="score-placeholder">Score rendering unavailable: OSMD did not load.</p>';
    return;
  }

  const xml = kind === 'reference' ? item.referenceXml : item.predictedXml;
  if (!xml) {
    const existing = scoreStates.get(index);
    if (existing) {
      try { existing.osmd.clear(); } catch (error) {}
      scoreStates.delete(index);
    }
    shell.innerHTML = '<p class="score-placeholder">No ' + kind + ' score.</p>';
    return;
  }

  let state = scoreStates.get(index);
  if (!state) {
    shell.innerHTML = '';
    const target = document.createElement('div');
    target.className = 'score';
    shell.appendChild(target);
    state = {
      osmd: new OSMD(target, { autoResize: true, drawingParameters: 'compacttight' }),
      kind: null,
      rendered: false
    };
    scoreStates.set(index, state);
  }
  if (state.kind === kind && state.rendered) return;

  state.kind = kind;
  state.rendered = false;
  try { state.osmd.clear(); } catch (error) {}
  state.osmd.load(xml)
    .then(() => {
      state.osmd.render();
      state.rendered = true;
    })
    .catch(() => {
      scoreStates.delete(index);
      shell.innerHTML = '<p class="score-placeholder">Could not render the ' + kind + ' score.</p>';
    });
}

function loadDefaultScore(index) {
  const item = DATA[index];
  if (!item) return;
  renderScore(index, item.predictedXml ? 'predicted' : 'reference');
}

document.querySelectorAll('button[data-score-kind]').forEach((button) => {
  button.addEventListener('click', () => {
    renderScore(Number(button.dataset.index), button.dataset.scoreKind);
  });
});

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const index = Number(entry.target.dataset.scoreIndex);
      if (!scoreStates.has(index)) loadDefaultScore(index);
      observer.unobserve(entry.target);
    });
  }, { rootMargin: '240px' });
  document.querySelectorAll('.score-shell').forEach((shell) => observer.observe(shell));
} else {
  DATA.forEach((item) => loadDefaultScore(item.index));
}

const audios = Array.from(document.querySelectorAll('audio'));
let activeAudio = null;
let activeIndex = -1;
let frame = 0;

function tick() {
  if (!activeAudio) return;
  renderScenario(activeIndex, activeAudio.currentTime);
  frame = requestAnimationFrame(tick);
}

audios.forEach((audio, index) => {
  audio.addEventListener('play', () => {
    audios.forEach((other) => { if (other !== audio) other.pause(); });
    activeAudio = audio;
    activeIndex = index;
    cancelAnimationFrame(frame);
    tick();
  });
  const stop = () => {
    if (activeAudio === audio) {
      activeAudio = null;
      activeIndex = -1;
      cancelAnimationFrame(frame);
      renderScenario(index, null);
    }
  };
  audio.addEventListener('pause', stop);
  audio.addEventListener('ended', stop);
  audio.addEventListener('seeked', () => renderScenario(index, audio.currentTime));
});

window.addEventListener('resize', () => {
  DATA.forEach((item) => renderScenario(item.index, null));
});

DATA.forEach((item) => renderScenario(item.index, null));
</script>
</body>
</html>
"""


def render_report(
    results: list[ScenarioResult],
    out_path: Path | str,
    title: str,
    subtitle: str,
) -> Path:
    out = Path(out_path)
    report_dir = out.parent
    out.parent.mkdir(parents=True, exist_ok=True)
    osmd_src = _osmd_source(report_dir)
    cards = "".join(
        _card_html(result, index, report_dir) for index, result in enumerate(results)
    )
    page = (
        TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__SUBTITLE__", html.escape(subtitle))
        .replace("__SUMMARY__", _summary_html(results))
        .replace("__NAV__", _nav_html(results))
        .replace("__CARDS__", cards)
        .replace("__DATA__", _payload(results))
        .replace("__OSMD__", html.escape(osmd_src, quote=True))
    )
    out.write_text(page)
    return out


def build_report(
    benchmark_dir: Path | str,
    out_path: Path | str | None = None,
    instrument: str = "alto",
    mode: str = "melody",
    mock: bool = False,
    limit: int | None = None,
    title: str = "Transcription benchmark",
) -> Path:
    benchmark = Path(benchmark_dir)
    results = collect_results(
        benchmark,
        instrument=instrument,
        mode=mode,
        mock=mock,
        limit=limit,
    )
    out = Path(out_path) if out_path is not None else benchmark / "report.html"
    subtitle = (
        f"{len(results)} take(s) · {instrument} · {mode} mode"
        + (" · demo/mock" if mock else "")
    )
    return render_report(results, out, title, subtitle)
