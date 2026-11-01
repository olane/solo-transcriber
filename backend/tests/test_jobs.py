import json
from dataclasses import fields

from app.jobs import JobManager


def test_persisted_fields_cover_every_runtime_independent_field():
    from app.jobs import Job, RUNTIME_JOB_FIELDS

    persisted = {item.name for item in fields(Job)} - RUNTIME_JOB_FIELDS
    job = Job(id="abc", filename="take.wav", mock=False, dir=None)  # type: ignore[arg-type]
    assert set(job.metadata()) == persisted


def test_remove_forgets_the_job_and_deletes_its_files():
    manager = JobManager()
    job = manager.create("forget-me.wav")

    manager.remove(job)

    assert manager.get(job.id) is None
    assert not job.dir.exists()


def test_restore_falls_back_to_notes_when_metadata_is_stale():
    manager = JobManager()
    job = manager.create("stale.wav")
    job.section_start = 0.0
    job.section_end = 2.0
    job.stems = ["other"]
    job.source_wav.write_bytes(b"source")
    (job.dir / "stems" / "other.wav").write_bytes(b"stem")
    manager.persist(job)

    job.notes_path.write_text(
        json.dumps(
            {
                "stem": "other",
                "bpm": 144.0,
                "duration": 1.8,
                "first_note": 0.3,
                "instrument": "tenor",
                "mode": "basic_pitch",
                "notes": [{"start": 0.3, "end": 0.9, "pitch": 60}],
            }
        )
    )

    restored_manager = JobManager()
    restored_manager.load()
    restored = restored_manager.get(job.id)

    assert restored is not None
    assert restored.status == "done"
    assert restored.bpm == 144.0
    assert restored.transcription_duration == 1.8
    assert restored.transcription_first_note == 0.3
    assert restored.transcription_instrument == "tenor"
    assert restored.transcription_mode == "basic_pitch"


def test_persist_round_trip_restores_every_field(tmp_path):
    manager = JobManager()
    job = manager.create("round-trip.wav")
    job.status = "done"
    job.stage = "Ready to play"
    job.progress = 100
    job.duration = 12.5
    job.section_start = 1.25
    job.section_end = 9.5
    job.stems = ["vocals", "bass"]
    job.selected_stem = "vocals"
    job.bpm = 132.5
    job.note_count = 3
    job.transcription_duration = 8.25
    job.transcription_first_note = 0.4
    job.transcription_instrument = "tenor"
    job.transcription_mode = "basic_pitch"
    job.mock = True
    job.created_at = 1234.5

    job.source_wav.write_bytes(b"source")
    for stem in job.stems:
        (job.dir / "stems" / f"{stem}.wav").write_bytes(b"stem")
    job.notes_path.write_text(
        json.dumps(
            {
                "notes": [
                    {"start": 0.0, "end": 0.5, "pitch": 60},
                    {"start": 0.5, "end": 1.0, "pitch": 62},
                    {"start": 1.0, "end": 1.5, "pitch": 64},
                ]
            }
        )
    )
    manager.persist(job)

    restored_manager = JobManager()
    restored_manager.load()
    restored = restored_manager.get(job.id)
    assert restored is not None
    for item in fields(job):
        if item.name in {"dir", "lock"}:
            continue
        assert getattr(restored, item.name) == getattr(job, item.name), item.name
