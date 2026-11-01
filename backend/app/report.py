from __future__ import annotations

import html
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from .evaluation import evaluate_notes, find_reference, load_reference_notes
from .pipeline.audio import wav_duration
from .pipeline.notation import render_score, safe_title
from .pipeline.notes import NoteEvent
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
TEMPLATE_PATH = Path(__file__).with_name("report_template.html")


@dataclass
class ScenarioResult:
    name: str
    audio_path: Path
    duration: float
    bpm: float | None = None
    reference: list[NoteEvent] = field(default_factory=list)
    predicted: list[NoteEvent] = field(default_factory=list)
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


def _pitch_bounds(
    reference: list[NoteEvent], predicted: list[NoteEvent]
) -> tuple[int, int]:
    pitches = [int(note["pitch"]) for note in reference + predicted]
    if not pitches:
        return 48, 84
    return max(0, min(pitches) - 2), min(127, max(pitches) + 2)


def _render_notes(
    notes: list[NoteEvent],
    bpm: float,
    instrument: str,
    title: str,
    out_dir: Path,
    stem: str,
) -> tuple[Path | None, Path | None, Path | None]:
    notes_path = out_dir / f"{stem}.notes.json"
    notes_path.write_text(
        json.dumps(
            {
                "source": title,
                "bpm": round(float(bpm), 3),
                "notes": notes,
            },
            indent=2,
        )
    )
    musicxml_path = out_dir / f"{stem}.musicxml"
    midi_path = out_dir / f"{stem}.mid"
    try:
        render_score(notes, bpm, instrument, "4/4", title, "musicxml", musicxml_path)
        render_score(notes, bpm, instrument, "4/4", title, "midi", midi_path)
    except Exception:
        return notes_path, None, None
    return notes_path, musicxml_path, midi_path


def run_benchmark(
    benchmark_dir: Path | str,
    instrument: str = "alto",
    mode: str = "melody",
    mock: bool = False,
    limit: int | None = None,
) -> list[ScenarioResult]:
    """Transcribe and score a benchmark, writing artifacts under predictions/."""
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
        reference_path = find_reference(audio)
        reference: list[NoteEvent] = []
        if reference_path is not None:
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
        stem = safe_title(name)
        notes_path, musicxml_path, midi_path = _render_notes(
            predicted, bpm, instrument, name, predictions, stem
        )
        reference_musicxml = None
        if reference:
            try:
                reference_path_out = predictions / f"{stem}.reference.musicxml"
                render_score(
                    reference,
                    bpm,
                    instrument,
                    "4/4",
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


def _note_table(notes: list[NoteEvent]) -> str:
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
    page = Template(TEMPLATE_PATH.read_text()).substitute(
        title=html.escape(title),
        subtitle=html.escape(subtitle),
        summary=_summary_html(results),
        nav=_nav_html(results),
        cards=cards,
        data=_payload(results),
        osmd=html.escape(osmd_src, quote=True),
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
    results = run_benchmark(
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
