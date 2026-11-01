from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Instrument = Literal["concert", "soprano", "alto", "tenor", "bari"]
ScoreFormat = Literal["musicxml", "midi"]
TranscriptionMode = Literal["melody", "basic_pitch"]
JobStatus = Literal[
    "queued",
    "preparing",
    "awaiting_section",
    "separating",
    "awaiting_stem",
    "transcribing",
    "done",
    "error",
]


class StemOut(BaseModel):
    name: str
    url: str
    suggested: bool = False


class SectionOut(BaseModel):
    start: float
    end: float


class SectionRequest(BaseModel):
    start: float = 0.0
    end: float | None = None


class TranscriptionOut(BaseModel):
    bpm: float
    note_count: int
    duration: float
    first_note: float = 0.0
    time_signature: str = "4/4"
    bar_offset: float = 0.0
    instrument: str = "alto"
    mode: str = "melody"


class JobOut(BaseModel):
    id: str
    status: JobStatus
    stage: str
    progress: int
    error: str | None = None
    filename: str
    duration: float = 0.0
    stems: list[StemOut] = Field(default_factory=list)
    selected_stem: str | None = None
    section: SectionOut | None = None
    preview_url: str | None = None
    original_url: str | None = None
    transcription: TranscriptionOut | None = None
    mock: bool = False
    created_at: float = 0.0


class TranscribeRequest(BaseModel):
    stem: str = "other"
    instrument: Instrument = "alto"
    mode: TranscriptionMode = "melody"
