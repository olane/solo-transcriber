import numpy as np
import pytest
import soundfile as sf

from app.evaluation import load_reference_notes
from app.pipeline.transcribe import INSTRUMENT_RANGES
from app.synthetic import build_scenarios, write_benchmark


def test_scenarios_stay_within_each_instrument_range():
    for instrument, (low, high) in INSTRUMENT_RANGES.items():
        for scenario in build_scenarios(instrument):
            for note in scenario.notes:
                assert low <= note["pitch"] <= high


def test_write_benchmark_creates_audio_and_reference_pairs(tmp_path):
    manifest = write_benchmark(tmp_path, instrument="alto", seed=3)

    assert len(manifest["scenarios"]) >= 5
    for entry in manifest["scenarios"]:
        wav_path = tmp_path / entry["wav"]
        json_path = tmp_path / entry["json"]
        assert wav_path.exists()
        assert json_path.exists()

        info = sf.info(str(wav_path))
        audio, sr = sf.read(str(wav_path))
        assert sr == manifest["sample_rate"]
        assert np.max(np.abs(audio)) > 0.1

        notes = load_reference_notes(json_path)
        assert len(notes) == entry["notes"]
        assert info.frames / info.samplerate > notes[-1]["end"]


def test_only_selects_requested_scenarios(tmp_path):
    manifest = write_benchmark(tmp_path, only=["vibrato"])

    assert [entry["name"] for entry in manifest["scenarios"]] == ["vibrato"]
    assert (tmp_path / "vibrato.wav").exists()
    assert not (tmp_path / "bleed.wav").exists()


def test_unknown_scenario_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown scenario"):
        write_benchmark(tmp_path, only=["kazoo"])
