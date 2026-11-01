from app.pipeline.separate import order_stems


def test_order_stems_uses_the_canonical_order_then_extras():
    assert order_stems({"other", "bass", "vocals", "kazoo", "drums"}) == [
        "vocals",
        "drums",
        "bass",
        "other",
        "kazoo",
    ]


import numpy as np
import soundfile as sf

from app.pipeline.separate import _sax_likeness, suggest_stem


def _write_tone(path, frequency, seconds=1.5, sr=22050, harmonics=(1.0, 0.6, 0.3)):
    time = np.arange(int(sr * seconds)) / sr
    signal = sum(
        weight * np.sin(2 * np.pi * frequency * harmonic * time)
        for harmonic, weight in zip((1, 2, 3), harmonics)
    )
    sf.write(str(path), 0.2 * signal, sr)


def test_sax_likeness_prefers_a_sustained_tonal_stem(tmp_path):
    tonal = tmp_path / "tonal.wav"
    noise = tmp_path / "noise.wav"
    _write_tone(tonal, 440.0)
    rng = np.random.default_rng(0)
    sf.write(str(noise), 0.05 * rng.standard_normal(int(22050 * 1.5)), 22050)

    assert _sax_likeness(tonal) > _sax_likeness(noise)


def test_suggest_stem_picks_the_sax_like_stem(tmp_path):
    _write_tone(tmp_path / "other.wav", 440.0)
    _write_tone(tmp_path / "bass.wav", 55.0, harmonics=(1.0, 0.0, 0.0))
    rng = np.random.default_rng(1)
    sf.write(
        str(tmp_path / "drums.wav"),
        0.05 * rng.standard_normal(int(22050 * 1.5)),
        22050,
    )

    assert suggest_stem(tmp_path, ["drums", "bass", "other"]) == "other"


def test_suggest_stem_keeps_other_on_a_tie(tmp_path):
    _write_tone(tmp_path / "other.wav", 440.0)
    _write_tone(tmp_path / "vocals.wav", 440.0)

    assert suggest_stem(tmp_path, ["vocals", "other"]) == "other"
