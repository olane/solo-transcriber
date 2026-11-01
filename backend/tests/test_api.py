import time

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


def make_sine(path, seconds=3.0):
    sr = 44100
    t = np.arange(int(sr * seconds)) / sr
    sf.write(str(path), 0.1 * np.sin(2 * np.pi * 440 * t), sr)
    return path


def wait_for(client, job_id, statuses, timeout=20.0):
    deadline = time.time() + timeout
    job = None
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {statuses}; last job: {job}")


def upload(client, path):
    with path.open("rb") as handle:
        response = client.post(
            "/api/jobs", files={"file": (path.name, handle, "audio/wav")}
        )
    assert response.status_code == 200, response.text
    return wait_for(client, response.json()["id"], {"awaiting_section", "error"})


def test_mock_pipeline_end_to_end(tmp_path):
    client = TestClient(app)
    wav = make_sine(tmp_path / "take.wav")

    job = upload(client, wav)
    assert job["status"] == "awaiting_section", job
    assert job["preview_url"].endswith("/source.wav")
    assert job["duration"] == 3.0

    response = client.post(
        f"/api/jobs/{job['id']}/separate", json={"start": 0.25, "end": 2.5}
    )
    assert response.status_code == 200, response.text

    job = wait_for(client, job["id"], {"awaiting_stem", "error"})
    assert job["status"] == "awaiting_stem", job
    assert job["section"] == {"start": 0.25, "end": 2.5}
    assert job["original_url"].endswith("/section.wav")
    assert any(stem["suggested"] for stem in job["stems"])

    response = client.post(
        f"/api/jobs/{job['id']}/transcribe",
        json={"stem": "other", "instrument": "tenor", "mode": "basic_pitch"},
    )
    assert response.status_code == 200, response.text

    job = wait_for(client, job["id"], {"done", "error"})
    assert job["status"] == "done", job
    assert job["transcription"]["note_count"] > 0
    assert job["transcription"]["instrument"] == "tenor"
    assert job["transcription"]["mode"] == "basic_pitch"

    response = client.get(
        f"/api/jobs/{job['id']}/score",
        params={"instrument": "alto", "time_signature": "4/4"},
    )
    assert response.status_code == 200, response.text
    assert "<score-partwise" in response.text
    assert "<transpose>" in response.text

    response = client.get(
        f"/api/jobs/{job['id']}/score",
        params={"instrument": "concert", "time_signature": "3/4", "format": "midi"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/midi")


def test_unknown_stem_is_rejected(tmp_path):
    client = TestClient(app)
    job = upload(client, make_sine(tmp_path / "take2.wav"))
    client.post(f"/api/jobs/{job['id']}/separate", json={"start": 0, "end": 3})
    job = wait_for(client, job["id"], {"awaiting_stem", "error"})
    response = client.post(
        f"/api/jobs/{job['id']}/transcribe", json={"stem": "kazoo"}
    )
    assert response.status_code == 400


def test_section_must_be_at_least_two_seconds(tmp_path):
    client = TestClient(app)
    job = upload(client, make_sine(tmp_path / "take3.wav"))
    response = client.post(
        f"/api/jobs/{job['id']}/separate", json={"start": 0, "end": 1.0}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Choose at least 2 seconds of audio"


def test_section_end_must_follow_its_start(tmp_path):
    client = TestClient(app)
    job = upload(client, make_sine(tmp_path / "backwards.wav"))
    response = client.post(
        f"/api/jobs/{job['id']}/separate", json={"start": 2.0, "end": 1.0}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "The section end must be after its start"


def test_saved_jobs_are_listed_and_restored(tmp_path):
    client = TestClient(app)
    job = upload(client, make_sine(tmp_path / "saved.wav"))
    client.post(f"/api/jobs/{job['id']}/separate", json={"start": 0.25, "end": 2.5})
    job = wait_for(client, job["id"], {"awaiting_stem", "error"})
    assert job["status"] == "awaiting_stem", job
    client.post(
        f"/api/jobs/{job['id']}/transcribe",
        json={"stem": "other", "instrument": "tenor", "mode": "melody"},
    )
    job = wait_for(client, job["id"], {"done", "error"})
    assert job["status"] == "done", job

    listing = client.get("/api/jobs").json()
    assert any(item["id"] == job["id"] for item in listing)

    from app.jobs import JobManager

    restored_manager = JobManager()
    restored_manager.load()
    restored = restored_manager.get(job["id"])
    assert restored is not None
    snapshot = restored_manager.snapshot(restored)
    assert snapshot.status == "done"
    assert snapshot.filename == "saved.wav"
    assert snapshot.section is not None
    assert snapshot.section.start == 0.25
    assert snapshot.section.end == 2.5
    assert [stem.name for stem in snapshot.stems] == ["other", "vocals", "bass"]
    assert snapshot.transcription is not None
    assert snapshot.transcription.instrument == "tenor"
    assert snapshot.transcription.note_count > 0

    response = client.delete(f"/api/jobs/{job['id']}")
    assert response.status_code == 200, response.text
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
