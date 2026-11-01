from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from . import config
from .pipeline.audio import make_preview, to_wav, trim_wav, wav_duration
from .pipeline.notation import render_score, safe_title
from .pipeline.separate import STEM_ORDER, SUGGESTED_STEM, separate_stems
from .pipeline.transcribe import transcribe_file
from .schemas import JobOut, JobStatus, SectionOut, StemOut, TranscriptionOut

SCORE_CACHE_VERSION = "2"
RUNNING_STATUSES: frozenset[JobStatus] = frozenset(
    {"queued", "preparing", "separating", "transcribing"}
)
RUNTIME_JOB_FIELDS = frozenset({"dir", "lock"})

# Stream the lossless source to the browser for normal tracks; only long files
# get a compressed audition copy.
PREVIEW_THRESHOLD_BYTES = 100 * 1024 * 1024


@dataclass
class Job:
    id: str
    filename: str
    mock: bool
    dir: Path
    status: JobStatus = "queued"
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
    created_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def metadata_path(self) -> Path:
        return self.dir / "job.json"

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

    def metadata(self) -> dict[str, Any]:
        """Return the JSON-serialisable fields, excluding runtime-only state."""
        payload: dict[str, Any] = {}
        for item in fields(self):
            if item.name in RUNTIME_JOB_FIELDS:
                continue
            value = getattr(self, item.name)
            payload[item.name] = list(value) if isinstance(value, list) else value
        return payload

    def set_phase(self, status: JobStatus, stage: str, progress: int) -> None:
        self.status = status
        self.stage = stage
        self.progress = progress


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

    def remove(self, job: Job) -> None:
        with self._lock:
            self._jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    def list_saved(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def load(self) -> None:
        jobs_dir = config.DATA_DIR / "jobs"
        if not jobs_dir.is_dir():
            return
        restored: list[Job] = []
        for job_dir in sorted(jobs_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            try:
                job = self._restore(job_dir)
            except Exception:
                traceback.print_exc()
                continue
            if job is not None:
                restored.append(job)
        with self._lock:
            for job in restored:
                self._jobs.setdefault(job.id, job)

    def _restore(self, job_dir: Path) -> Job | None:
        source = job_dir / "source.wav"
        if not source.exists():
            return None
        metadata = _read_json(job_dir / "job.json") or {}
        notes = _read_json(job_dir / "notes.json") or {}

        created = _optional_float(metadata.get("created_at"))
        job = Job(
            id=str(metadata.get("id") or job_dir.name),
            filename=str(metadata.get("filename") or notes.get("filename") or job_dir.name),
            mock=bool(metadata.get("mock", config.MOCK)),
            dir=job_dir,
            created_at=created if created is not None else job_dir.stat().st_mtime,
        )
        job.duration = _to_float(metadata.get("duration"))
        if job.duration <= 0:
            try:
                job.duration = wav_duration(source)
            except Exception:
                job.duration = 0.0
        job.section_start = _optional_float(metadata.get("section_start"))
        job.section_end = _optional_float(metadata.get("section_end"))
        if job.section_start is None:
            job.section_start = _optional_float(notes.get("section_start"))
        if job.section_end is None:
            job.section_end = _optional_float(notes.get("section_end"))
        disk_stems = _stems_on_disk(job_dir)
        persisted_stems = [
            str(name) for name in metadata.get("stems") or [] if str(name) in disk_stems
        ]
        job.stems = persisted_stems + [
            name for name in disk_stems if name not in persisted_stems
        ]
        job.selected_stem = metadata.get("selected_stem") or notes.get("stem")
        job.transcription_instrument = str(
            metadata.get("transcription_instrument") or notes.get("instrument") or "alto"
        )
        job.transcription_mode = str(
            metadata.get("transcription_mode") or notes.get("mode") or "melody"
        )
        job.transcription_mock = bool(
            metadata.get("transcription_mock", notes.get("mock", False))
        )
        job.transcription_duration = _to_float(
            metadata.get("transcription_duration", notes.get("duration"))
        )
        job.transcription_first_note = _to_float(
            metadata.get("transcription_first_note", notes.get("first_note"))
        )
        job.bpm = _optional_float(metadata.get("bpm", notes.get("bpm")))
        job.note_count = int(metadata.get("note_count") or len(notes.get("notes") or []))

        if str(metadata.get("status")) == "error":
            job.set_phase(
                "error",
                str(metadata.get("stage") or "Something went wrong"),
                int(metadata.get("progress") or 0),
            )
            job.error = metadata.get("error")
            return job

        if job.bpm is not None and (job_dir / "notes.json").exists():
            job.set_phase("done", "Ready to play", 100)
        elif (
            job.section_start is not None
            and job.section_end is not None
            and job.stems
        ):
            job.set_phase("awaiting_stem", "Pick the sax stem", 55)
        else:
            job.set_phase("awaiting_section", "Choose the solo section", 10)
            job.bpm = None
            job.note_count = 0
            job.transcription_duration = 0.0
            job.transcription_first_note = 0.0
            job.transcription_mock = False
        return job

    def persist(self, job: Job) -> None:
        try:
            with job.lock:
                payload: dict[str, Any] = {
                    "version": 1,
                    **job.metadata(),
                }
            path = job.metadata_path
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, indent=2))
            temp.replace(path)
        except Exception:
            traceback.print_exc()

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
                created_at=job.created_at,
            )

    def start_preparation(self, job: Job, upload: Path) -> None:
        with job.lock:
            job.set_phase("preparing", "Preparing the audio", 5)
        self.persist(job)
        self._executor.submit(self._prepare_worker, job, upload)

    def start_section_separation(self, job: Job, start: float, end: float) -> None:
        with job.lock:
            job.set_phase("separating", "Separating the section", 15)
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
        self.persist(job)
        self._executor.submit(self._separate_worker, job, start, end)

    def start_transcription(
        self, job: Job, stem: str, instrument: str, mode: str
    ) -> None:
        with job.lock:
            job.set_phase("transcribing", "Transcribing the sax line", 65)
            job.selected_stem = stem
            job.transcription_instrument = instrument
            job.transcription_mode = mode
            job.bpm = None
            job.note_count = 0
            job.transcription_duration = 0.0
            job.transcription_first_note = 0.0
            job.transcription_mock = False
            job.error = None
        self.persist(job)
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
            job.set_phase(
                "error", "Something went wrong", min(job.progress, 99)
            )
            job.error = str(exc) or exc.__class__.__name__
        self.persist(job)

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
                job.set_phase("awaiting_section", "Choose the solo section", 10)
            self.persist(job)
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
                job.set_phase("awaiting_stem", "Pick the sax stem", 55)
            self.persist(job)
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
                job.set_phase("done", "Ready to play", 100)
                job.error = None
            self.persist(job)
        except Exception as exc:
            self._fail(job, exc)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stems_on_disk(job_dir: Path) -> list[str]:
    stems_dir = job_dir / "stems"
    if not stems_dir.is_dir():
        return []
    found = {path.stem for path in stems_dir.glob("*.wav")}
    ordered = [name for name in STEM_ORDER if name in found]
    ordered += sorted(found - set(ordered))
    return ordered


manager = JobManager()
