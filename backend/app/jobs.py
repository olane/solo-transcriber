from __future__ import annotations

import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .pipeline.audio import to_wav, wav_duration
from .pipeline.notation import render_score, safe_title
from .pipeline.separate import SUGGESTED_STEM, separate_stems
from .pipeline.transcribe import transcribe_file
from .schemas import JobOut, StemOut, TranscriptionOut

SCORE_CACHE_VERSION = "2"


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
    stems: list[str] = field(default_factory=list)
    selected_stem: str | None = None
    bpm: float | None = None
    note_count: int = 0
    transcription_duration: float = 0.0
    transcription_first_note: float = 0.0
    transcription_mock: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def source_wav(self) -> Path:
        return self.dir / "source.wav"

    @property
    def notes_path(self) -> Path:
        return self.dir / "notes.json"

    def stem_path(self, stem: str) -> Path:
        if stem == "original":
            return self.source_wav
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
            transcription = None
            if job.bpm is not None:
                transcription = TranscriptionOut(
                    bpm=job.bpm,
                    note_count=job.note_count,
                    duration=job.transcription_duration,
                    first_note=job.transcription_first_note,
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
                transcription=transcription,
                mock=job.mock,
            )

    def start_separation(self, job: Job, upload: Path) -> None:
        with job.lock:
            job.status = "separating"
            job.stage = "Separating the mix"
            job.progress = 15
        self._executor.submit(self._separate_worker, job, upload)

    def start_transcription(self, job: Job, stem: str) -> None:
        with job.lock:
            job.status = "transcribing"
            job.stage = "Transcribing the sax line"
            job.progress = 65
            job.selected_stem = stem
            job.error = None
        self._executor.submit(self._transcribe_worker, job, stem)

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

    def _separate_worker(self, job: Job, upload: Path) -> None:
        try:
            to_wav(upload, job.source_wav)
            duration = wav_duration(job.source_wav)
            stems = separate_stems(job.source_wav, job.dir, mock=job.mock)
            with job.lock:
                job.duration = duration
                job.stems = stems
                job.status = "awaiting_stem"
                job.stage = "Pick the sax stem"
                job.progress = 55
            upload.unlink(missing_ok=True)
        except Exception as exc:
            self._fail(job, exc)

    def _transcribe_worker(self, job: Job, stem: str) -> None:
        try:
            notes, meta = transcribe_file(
                job.stem_path(stem), job.source_wav, mock=job.mock
            )
            payload: dict[str, Any] = {
                "stem": stem,
                "bpm": meta["bpm"],
                "duration": meta["duration"],
                "mock": meta["mock"],
                "first_note": meta.get("first_note", 0.0),
                "filename": job.filename,
                "notes": notes,
            }
            job.notes_path.write_text(json.dumps(payload))
            with job.lock:
                job.bpm = float(meta["bpm"])
                job.note_count = len(notes)
                job.transcription_duration = float(meta["duration"])
                job.transcription_first_note = float(meta.get("first_note", 0.0))
                job.transcription_mock = bool(meta["mock"])
                job.status = "done"
                job.stage = "Ready to play"
                job.progress = 100
                job.error = None
        except Exception as exc:
            self._fail(job, exc)


manager = JobManager()
