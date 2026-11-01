import numpy as np

from app.pipeline.beats import BeatGrid, infer_meter


def repeated(pattern, repeats=6):
    return np.tile(np.asarray(pattern, dtype=float), repeats)


def test_infer_meter_finds_common_meters_and_phase():
    four = repeated([1.0, 0.15, 0.6, 0.15])
    three = repeated([1.0, 0.35, 0.35])
    six = repeated([1.0, 0.1, 0.4, 0.55, 0.1, 0.4])

    assert infer_meter(four)[:2] == ("4/4", 0)
    assert infer_meter(np.roll(four, 1))[:2] == ("4/4", 1)
    assert infer_meter(three)[:2] == ("3/4", 0)
    assert infer_meter(six)[:2] == ("6/8", 0)


def test_infer_meter_defaults_to_four_four_with_little_evidence():
    assert infer_meter(np.array([1.0, 0.8]))[:2] == ("4/4", 0)


def test_bar_start_uses_the_last_downbeat():
    grid = BeatGrid(
        bpm=120.0,
        time_signature="4/4",
        beats=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
    )

    assert grid.downbeats == (0.0, 2.0)
    assert grid.bar_start_at_or_before(2.4) == 2.0
    assert grid.bar_start_at_or_before(0.1) == 0.0


def test_bar_start_falls_back_to_first_pulse_for_a_pickup():
    grid = BeatGrid(
        bpm=120.0,
        time_signature="3/4",
        beats=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
        phase=0,
    )

    assert grid.bar_start_at_or_before(1.2) == 1.0
    assert grid.bar_start_at_or_before(0.5) is None


def test_detect_beat_grid_uses_accented_downbeats(tmp_path):
    import soundfile as sf

    sr = 22050
    beat = 0.5
    seconds = beat * 4 * 4
    time = np.arange(int(sr * seconds)) / sr
    audio = np.zeros_like(time)
    for index in range(16):
        start = int(index * beat * sr)
        length = int(0.05 * sr)
        gain = 1.0 if index % 4 == 0 else (0.5 if index % 4 == 2 else 0.2)
        envelope = np.exp(-np.linspace(0, 8, length))
        click = np.sin(2 * np.pi * 1000 * np.arange(length) / sr)
        audio[start : start + length] += gain * envelope * click

    path = tmp_path / "click.wav"
    sf.write(str(path), audio, sr)

    from app.pipeline.beats import detect_beat_grid

    grid = detect_beat_grid(path)

    assert grid is not None
    assert grid.time_signature == "4/4"
    assert round(grid.bpm) == 120
    assert grid.phase == 0
