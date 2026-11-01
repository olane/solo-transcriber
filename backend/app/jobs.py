from __future__ import annotations

import json
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .pipeline.audio import make_preview, to_wav, trim_wav, wav_duration
from .pipeline.notation import render_score, safe_title
from .pipeline.separate import SUGGESTED_STEM, separate_stems
from .pipeline.transcribe import transcribe_file
from .schemas import JobOut, SectionOut, StemOut, TranscriptionOut

SCORE_CACHE_VERSION = "2"

# Stream the lossless source to the browser for normal tracks; only long files
# get a compressed audition copy.
PREVIEW_THRESHOLD_BYTES = 100 * 1024 * 1024


@dataclass
class Job:
    id: str
    filename: str
    mock: bool
    dir: Path
    status: str = "queued"
    stage: str = "Queued"
    progress: int = 0
    error: str | None = None
    duration: float = 0.0
    section_start: float | None = None
    section_end: float | None = None
    stems: list[str] = field(default_factory=list)
    selected_stem: str | None = None
    bpm: float | None = None
    note_count: int = 0
    transcription_duration: float = 0.0
    transcription_first_note: float = 0.0
    transcription_instrument: str = "alto"
    transcription_mode: str = "melody"
    transcription_mock: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def source_wav(self) -> Path:
        return self.dir / "source.wav"

    @property
    def section_wav(self) -> Path:
        return self.dir / "section.wav"

    @property
    def mix_wav(self) -> Path:
        return self.section_wav if self.section_wav.exists() else self.source_wav

    @property
    def notes_path(self) -> Path:
        return self.dir / "notes.json"

    def stem_path(self, stem: str) -> Path:
        if stem == "original":
            return self.mix_wav
        return self.dir / "stems" / f"{stem}.wav"


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="solo-job")

    def create(self, filename: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_dir = config.DATA_DIR / "jobs" / job_id
        for sub in ("stems", "scores"):
            (job_dir / sub).mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, filename=filename, mock=config.MOCK, dir=job_dir)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def snapshot(self, job: Job) -> JobOut:
        with job.lock:
            stems = [
                StemOut(
                    name=name,
                    url=f"/media/jobs/{job.id}/stems/{name}.wav",
                    suggested=name == SUGGESTED_STEM,
                )
                for name in job.stems
            ]
            section = None
            if job.section_start is not None and job.section_end is not None:
                section = SectionOut(start=job.section_start, end=job.section_end)
            preview_url = None
            original_url = None
            if job.source_wav.exists():
                preview = job.dir / "preview.mp3"
                preview_url = (
                    f"/media/jobs/{job.id}/preview.mp3"
                    if preview.exists()
                    else f"/media/jobs/{job.id}/source.wav"
                )
                original_url = (
                    f"/media/jobs/{job.id}/section.wav"
                    if job.section_wav.exists()
                    else f"/media/jobs/{job.id}/source.wav"
                )
            transcription = None
            if job.bpm is not None:
                transcription = TranscriptionOut(
                    bpm=job.bpm,
                    note_count=job.note_count,
                    duration=job.transcription_duration,
                    first_note=job.transcription_first_note,
                    instrument=job.transcription_instrument,
                    mode=job.transcription_mode,
                    mock=job.transcription_mock,
                )
            return JobOut(
                id=job.id,
                status=job.status,
                stage=job.stage,
                progress=job.progress,
                error=job.error,
                filename=job.filename,
                duration=job.duration,
                stems=stems,
                selected_stem=job.selected_stem,
                section=section,
                preview_url=preview_url,
                original_url=original_url,
                transcription=transcription,
                mock=job.mock,
            )

    def start_preparation(self, job: Job, upload: Path) -> None:
        with job.lock:
            job.status = "preparing"
            job.stage = "Preparing the audio"
            job.progress = 5
        self._executor.submit(self._prepare_worker, job, upload)

    def start_section_separation(self, job: Job, start: float, end: float) -> None:
        with job.lock:
            job.status = "separating"
            job.stage = "Separating the section"
            job.progress = 15
            job.section_start = start
            job.section_end = end
            job.stems = []
            job.selected_stem = None
            job.bpm = None
            job.note_count = 0
            job.transcription_duration = 0.0
            job.transcription_first_note = 0.0
            job.transcription_instrument = "alto"
            job.transcription_mode = "melody"
            job.transcription_mock = False
            job.error = None
        self._executor.submit(self._separate_worker, job, start, end)

    def start_transcription(
        self, job: Job, stem: str, instrument: str, mode: str
    ) -> None:
        with job.lock:
            job.status = "transcribing"
            job.stage = "Transcribing the sax line"
            job.progress = 65
            job.selected_stem = stem
            job.transcription_instrument = instrument
            job.transcription_mode = mode
            job.bpm = None
            job.note_count = 0
            job.transcription_duration = 0.0
            job.transcription_first_note = 0.0
            job.transcription_mock = False
            job.error = None
        self._executor.submit(
            self._transcribe_worker, job, stem, instrument, mode
        )

    def render(
        self,
        job: Job,
        instrument_key: str,
        time_signature: str,
        bpm: float,
        fmt: str,
    ) -> tuple[Path, str]:
        with job.lock:
            if job.bpm is None or not job.notes_path.exists():
                raise ValueError("This job has no transcription yet")
            title = Path(job.filename).stem
        safe = safe_title(
            f"v{SCORE_CACHE_VERSION}-{title}-{instrument_key}-{time_signature}-{float(bpm):.1f}"
        )
        out = job.dir / "scores" / f"{safe}.{fmt}"
        if not out.exists():
            payload = json.loads(job.notes_path.read_text())
            render_score(
                payload["notes"],
                bpm,
                instrument_key,
                time_signature,
                title,
                fmt,
                out,
            )
        download_name = f"{safe_title(title)}-{instrument_key}.{fmt}"
        return out, download_name

    def _fail(self, job: Job, exc: Exception) -> None:
        traceback.print_exc()
        with job.lock:
            job.status = "error"
            job.stage = "Something went wrong"
            job.error = str(exc) or exc.__class__.__name__
            job.progress = min(job.progress, 99)

    def _prepare_worker(self, job: Job, upload: Path) -> None:
        try:
            to_wav(upload, job.source_wav)
            duration = wav_duration(job.source_wav)
            if duration < 2.0:
                raise RuntimeError("The track must be at least 2 seconds long")
            if job.source_wav.stat().st_size > PREVIEW_THRESHOLD_BYTES:
                try:
                    make_preview(job.source_wav, job.dir / "preview.mp3")
                except Exception:
                    traceback.print_exc()
            with job.lock:
                job.duration = duration
                job.status = "awaiting_section"
                job.stage = "Choose the solo section"
                job.progress = 10
            upload.unlink(missing_ok=True)
        except Exception as exc:
            self._fail(job, exc)

    def _clear_outputs(self, job: Job) -> None:
        for name in ("stems", "scores"):
            path = job.dir / name
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
        job.notes_path.unlink(missing_ok=True)
        job.section_wav.unlink(missing_ok=True)

    def _clear_scores(self, job: Job) -> None:
        scores = job.dir / "scores"
        shutil.rmtree(scores, ignore_errors=True)
        scores.mkdir(parents=True, exist_ok=True)

    def _separate_worker(self, job: Job, start: float, end: float) -> None:
        try:
            self._clear_outputs(job)
            section = trim_wav(job.source_wav, job.section_wav, start, end)
            stems = separate_stems(section, job.dir, mock=job.mock)
            with job.lock:
                job.stems = stems
                job.status = "awaiting_stem"
                job.stage = "Pick the sax stem"
                job.progress = 55
        except Exception as exc:
            self._fail(job, exc)

    def _transcribe_worker(
        self, job: Job, stem: str, instrument: str, mode: str
    ) -> None:
        try:
            self._clear_scores(job)
            notes, meta = transcribe_file(
                job.stem_path(stem),
                job.mix_wav,
                mock=job.mock,
                mode=mode,
                instrument=instrument,
            )
            payload: dict[str, Any] = {
                "stem": stem,
                "bpm": meta["bpm"],
                "duration": meta["duration"],
                "mock": meta["mock"],
                "first_note": meta.get("first_note", 0.0),
                "instrument": instrument,
                "mode": mode,
                "section_start": job.section_start,
                "section_end": job.section_end,
                "filename": job.filename,
                "notes": notes,
            }
            job.notes_path.write_text(json.dumps(payload))
            with job.lock:
                job.bpm = float(meta["bpm"])
                job.note_count = len(notes)
                job.transcription_duration = float(meta["duration"])
                job.transcription_first_note = float(meta.get("first_note", 0.0))
                job.transcription_instrument = instrument
                job.transcription_mode = mode
                job.transcription_mock = bool(meta["mock"])
                job.status = "done"
                job.stage = "Ready to play"
                job.progress = 100
                job.error = None
        except Exception as exc:
            self._fail(job, exc)


manager = JobManager()
