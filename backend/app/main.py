from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .jobs import manager
from .pipeline.notation import TIME_SIGNATURES
from .schemas import Instrument, JobOut, ScoreFormat, SectionRequest, TranscribeRequest

ALLOWED_SUFFIXES = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".aiff",
    ".aif",
    ".wma",
    ".mp4",
    ".webm",
}

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
manager.load()

app = FastAPI(title="Solo Transcriber", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(config.DATA_DIR)), name="media")


def get_job_or_404(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "mock": config.MOCK, "data_dir": str(config.DATA_DIR)}


@app.post("/api/jobs", response_model=JobOut)
async def create_job(file: UploadFile = File(...)) -> JobOut:
    filename = Path(file.filename or "track").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {suffix or 'none'}")

    job = manager.create(filename)
    upload = job.dir / f"upload{suffix}"
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with upload.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is larger than {config.MAX_UPLOAD_MB} MB",
                    )
                handle.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty")
    except HTTPException:
        manager.delete(job.id)
        shutil.rmtree(job.dir, ignore_errors=True)
        raise
    except Exception as exc:
        manager.delete(job.id)
        shutil.rmtree(job.dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    manager.start_preparation(job, upload)
    return manager.snapshot(job)


@app.get("/api/jobs", response_model=list[JobOut])
def list_jobs() -> list[JobOut]:
    return [manager.snapshot(job) for job in manager.list_saved()]


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def read_job(job_id: str) -> JobOut:
    return manager.snapshot(get_job_or_404(job_id))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    manager.remove(get_job_or_404(job_id))
    return {"ok": True}


@app.post("/api/jobs/{job_id}/separate", response_model=JobOut)
def separate(job_id: str, request: SectionRequest) -> JobOut:
    job = get_job_or_404(job_id)
    with job.lock:
        status = job.status
        duration = job.duration
    if status in {"queued", "preparing", "separating", "transcribing"}:
        raise HTTPException(status_code=409, detail="This job is still running")
    if duration <= 0:
        raise HTTPException(status_code=409, detail="The audio is not ready yet")

    start = max(0.0, float(request.start))
    end = duration if request.end is None else min(duration, float(request.end))
    if end - start < 2.0:
        raise HTTPException(status_code=400, detail="Choose at least 2 seconds of audio")
    if start >= end:
        raise HTTPException(status_code=400, detail="The section end must be after its start")

    manager.start_section_separation(job, start, end)
    return manager.snapshot(job)


@app.post("/api/jobs/{job_id}/transcribe", response_model=JobOut)
def transcribe(job_id: str, request: TranscribeRequest) -> JobOut:
    job = get_job_or_404(job_id)
    with job.lock:
        status = job.status
        stems = list(job.stems)
    if status in {"queued", "preparing", "separating", "transcribing"}:
        raise HTTPException(status_code=409, detail="This job is still running")
    if status == "awaiting_section":
        raise HTTPException(status_code=409, detail="Choose a section first")
    if request.stem != "original" and request.stem not in stems:
        raise HTTPException(status_code=400, detail="Unknown stem")
    if not job.mix_wav.exists():
        raise HTTPException(status_code=409, detail="The audio has not been prepared yet")
    manager.start_transcription(job, request.stem, request.instrument, request.mode)
    return manager.snapshot(job)


@app.get("/api/jobs/{job_id}/score")
def read_score(
    job_id: str,
    instrument: Instrument = Query("alto"),
    time_signature: str = Query("4/4"),
    bpm: float | None = Query(None, ge=20, le=400),
    format: ScoreFormat = Query("musicxml"),
) -> FileResponse:
    job = get_job_or_404(job_id)
    if time_signature not in TIME_SIGNATURES:
        raise HTTPException(status_code=400, detail="Unsupported time signature")
    with job.lock:
        if job.status != "done" or job.bpm is None:
            raise HTTPException(status_code=409, detail="This job has no finished transcription")
        effective_bpm = float(bpm if bpm is not None else job.bpm)

    try:
        path, download_name = manager.render(
            job, instrument, time_signature, effective_bpm, format
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    media_type = (
        "application/vnd.recordare.musicxml+xml"
        if format == "musicxml"
        else "audio/midi"
    )
    return FileResponse(path, media_type=media_type, filename=download_name)

FRONTEND_DIST = config.ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
