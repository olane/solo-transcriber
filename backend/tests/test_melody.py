import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torchcrepe")

from app.pipeline.melody import transcribe_melody
from app.pipeline.transcribe import frequency_bounds


def test_crepe_follows_a_simple_melody(tmp_path):
    sr = 16000
    notes = [440.0, 493.88, 523.25, 587.33]
    samples = []
    for freq in notes:
        t = np.arange(sr) / sr
        samples.append(0.2 * np.sin(2 * np.pi * freq * t))
    audio = np.concatenate(samples)
    path = tmp_path / "melody.wav"
    sf.write(str(path), audio, sr)

    fmin, fmax = frequency_bounds("alto")
    detected = transcribe_melody(path, fmin, fmax, model="tiny")

    pitches = [note["pitch"] for note in detected]
    assert 69 in pitches
    assert 71 in pitches
    assert 72 in pitches
    assert 74 in pitches
    assert len(detected) <= 12
    for note in detected:
        assert note["end"] > note["start"]
