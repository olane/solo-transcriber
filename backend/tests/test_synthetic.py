import numpy as np
import pytest
import soundfile as sf

from app.evaluation import load_reference_notes
from app.pipeline.transcribe import INSTRUMENT_RANGES
from app.synthetic import (
    LEAD_SECONDS,
    TAIL_SECONDS,
    build_scenarios,
    render_scenario,
    write_benchmark,
)

EXPECTED_SCENARIOS = {
    "repeated-articulated",
    "repeated-legato",
    "vibrato",
    "chromatic",
    "fast-run",
    "octave-trill",
    "scoops",
    "bleed",
    "mixed-phrase",
    "wide-leaps",
    "low-register",
    "high-register",
    "staccato",
    "long-tones",
    "syncopated",
    "semitone-trill",
    "arpeggio",
    "descending-scale",
    "dotted-rhythm",
    "triplets",
    "falls",
    "wide-vibrato",
    "grace-notes",
}


def test_scenarios_stay_within_each_instrument_range():
    for instrument, (low, high) in INSTRUMENT_RANGES.items():
        for scenario in build_scenarios(instrument):
            for note in scenario.notes:
                assert low <= note["pitch"] <= high


def test_scenarios_cover_the_documented_suite():
    names = {scenario.name for scenario in build_scenarios("alto")}
    assert names == EXPECTED_SCENARIOS


def test_scenario_names_are_unique():
    names = [scenario.name for scenario in build_scenarios("alto")]
    assert len(names) == len(set(names))


def test_every_scenario_has_ordered_notes():
    for scenario in build_scenarios("alto"):
        assert scenario.notes
        for note in scenario.notes:
            assert 0 <= note["start"] < note["end"]
        starts = [note["start"] for note in scenario.notes]
        assert starts == sorted(starts)


@pytest.mark.parametrize("instrument", sorted(INSTRUMENT_RANGES))
def test_every_scenario_renders_non_silent_audio(instrument):
    sr = 8000
    for scenario in build_scenarios(instrument):
        audio = render_scenario(scenario, instrument, sr=sr, seed=2)

        assert audio.ndim == 1
        assert audio.size == int(
            (LEAD_SECONDS + max(n["end"] for n in scenario.notes) + TAIL_SECONDS)
            * sr
        )
        assert np.max(np.abs(audio)) > 0.1


def test_render_is_deterministic_for_a_seed():
    scenario = build_scenarios("alto")[0]

    first = render_scenario(scenario, seed=11)
    second = render_scenario(scenario, seed=11)
    third = render_scenario(scenario, seed=12)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_bleed_scenario_only_leaks_accompaniment():
    by_name = {scenario.name: scenario for scenario in build_scenarios("alto")}
    early = slice(0, int(0.2 * 8000))

    bleed = render_scenario(by_name["bleed"], sr=8000)
    clean = render_scenario(by_name["repeated-articulated"], sr=8000)

    assert np.max(np.abs(bleed[early])) > 0.0
    assert np.max(np.abs(clean[early])) == 0.0


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


def test_reference_offsets_match_scenario_timing(tmp_path):
    manifest = write_benchmark(tmp_path, instrument="alto", seed=1, sr=8000)
    scenarios = {scenario.name: scenario for scenario in build_scenarios("alto")}

    for entry in manifest["scenarios"]:
        scenario = scenarios[entry["name"]]
        notes = load_reference_notes(tmp_path / entry["json"])
        assert [note["start"] for note in notes] == pytest.approx(
            [LEAD_SECONDS + note["start"] for note in scenario.notes]
        )
        assert [note["pitch"] for note in notes] == [
            note["pitch"] for note in scenario.notes
        ]


def test_only_selects_requested_scenarios(tmp_path):
    manifest = write_benchmark(tmp_path, only=["vibrato"])

    assert [entry["name"] for entry in manifest["scenarios"]] == ["vibrato"]
    assert (tmp_path / "vibrato.wav").exists()
    assert not (tmp_path / "bleed.wav").exists()


def test_unknown_scenario_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown scenario"):
        write_benchmark(tmp_path, only=["kazoo"])
