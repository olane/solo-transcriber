import numpy as np

from app.pipeline.melody import _pick_onsets, _trim_releases


def test_pick_onsets_finds_separated_attacks():
    envelope = np.zeros(200)
    for center in (30, 90, 150):
        envelope[center - 2 : center + 3] = [0.2, 0.6, 1.0, 0.5, 0.1]

    onsets = _pick_onsets(envelope)

    assert len(onsets) == 3
    assert np.all(np.diff(onsets) > 10)


def test_pick_onsets_ignores_silence():
    assert _pick_onsets(np.zeros(100)).size == 0


def test_trim_releases_drops_the_trailing_tail():
    rms = np.array([1.0, 1.0, 1.0, 0.9, 0.3, 0.05, 0.02])

    trimmed = _trim_releases([(0, 6, 69)], rms)

    assert trimmed == [(0, 4, 69)]


def test_trim_releases_never_shortens_past_the_minimum():
    rms = np.array([1.0, 0.01, 0.01, 0.01])

    trimmed = _trim_releases([(0, 3, 69)], rms)

    assert trimmed == [(0, 2, 69)]


def test_onset_picker_keeps_legato_rearticulations(tmp_path):
    import soundfile as sf

    from app.pipeline.melody import HOP_LENGTH, _detect_onsets
    from app.pipeline.transcribe import frequency_bounds
    from app.synthetic import write_benchmark

    write_benchmark(
        tmp_path, instrument="alto", only=["repeated-legato"], sr=16000
    )
    audio, sr = sf.read(str(tmp_path / "repeated-legato.wav"))
    fmin, fmax = frequency_bounds("alto")

    onsets = _detect_onsets(audio.astype(float), sr, HOP_LENGTH, fmin, fmax)
    times = [float(frame) * HOP_LENGTH / sr for frame in onsets]

    assert sum(1 for time in times if 0.4 <= time <= 2.0) >= 6
