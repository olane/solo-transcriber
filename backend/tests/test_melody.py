import numpy as np
import pytest
import soundfile as sf

from app.pipeline.melody import (
    _confirmed_onsets,
    _detect_onsets,
    _segment_notes,
    transcribe_melody,
)
from app.pipeline.transcribe import frequency_bounds


def times(frames):
    return np.arange(frames) * 0.01


def midi_to_f0(midi):
    return 440.0 * 2 ** ((np.asarray(midi) - 69.0) / 12.0)


def test_onset_splits_repeated_notes():
    frames = 100
    f0 = np.full(frames, 440.0)
    periodicity = np.ones(frames)
    onsets = np.zeros(frames, dtype=bool)
    onsets[[0, 50]] = True

    segments = _segment_notes(f0, periodicity, times(frames), onsets)

    assert [(start, end) for start, end, _ in segments] == [(0, 49), (50, 99)]
    assert [pitch for _, _, pitch in segments] == [69, 69]


def test_vibrato_stays_in_one_note():
    frames = 200
    t = times(frames)
    midi = 69.0 + 0.4 * np.sin(2 * np.pi * 5 * t)

    segments = _segment_notes(
        midi_to_f0(midi), np.ones(frames), t
    )

    assert len(segments) == 1
    assert segments[0][2] == 69


def test_semitone_change_splits_note():
    midi = np.concatenate([np.full(50, 69.0), np.full(50, 70.0)])

    segments = _segment_notes(
        midi_to_f0(midi), np.ones(100), times(100)
    )

    assert [pitch for _, _, pitch in segments] == [69, 70]


def test_short_gap_merges_without_an_onset():
    frames = 100
    f0 = np.full(frames, 440.0)
    periodicity = np.ones(frames)
    periodicity[45:48] = 0.0

    segments = _segment_notes(f0, periodicity, times(frames))

    assert len(segments) == 1


def test_onset_prevents_gap_merge():
    frames = 100
    f0 = np.full(frames, 440.0)
    periodicity = np.ones(frames)
    periodicity[45:48] = 0.0
    onsets = np.zeros(frames, dtype=bool)
    onsets[48] = True

    segments = _segment_notes(f0, periodicity, times(frames), onsets)

    assert len(segments) == 2


def test_articulation_needs_a_confidence_dip():
    periodicity = np.full(60, 0.95)
    periodicity[30:33] = 0.6
    onsets = np.array([10, 30])

    confirmed = _confirmed_onsets(onsets, periodicity)

    assert list(confirmed) == [30]


def test_early_onset_before_pitch_change_does_not_leave_a_fragment():
    midi = np.concatenate([np.full(50, 69.0), np.full(30, 70.0)])
    onsets = np.zeros(80, dtype=bool)
    onsets[46] = True

    segments = _segment_notes(
        midi_to_f0(midi), np.ones(80), times(80), onsets
    )

    assert [pitch for _, _, pitch in segments] == [69, 70]
    assert segments[0][1] == 49


def test_onset_detector_finds_repeated_attacks():
    sr = 16000
    t = np.arange(int(sr * 0.25)) / sr
    note = 0.2 * np.sin(2 * np.pi * 440 * t) * np.hanning(t.size)
    gap = np.zeros(int(sr * 0.03))
    audio = np.concatenate(
        [np.zeros(int(sr * 0.1)), note, gap, note, gap, note]
    )

    fmin, fmax = frequency_bounds("alto")
    onsets = _detect_onsets(audio, sr, 160, fmin, fmax)

    assert len(onsets) >= 3
    assert np.all(np.diff(onsets) > 10)


def test_crepe_follows_a_simple_melody(tmp_path):
    pytest.importorskip("torchcrepe")

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


from app.pipeline.melody import (  # noqa: E402
    _correct_octaves,
    _voiced_mask,
)


def _harmonic_tone(frequency, harmonics=(1.0, 0.6, 0.3), sr=16000, seconds=0.6):
    time = np.arange(int(sr * seconds)) / sr
    return sum(
        weight * np.sin(2 * np.pi * frequency * harmonic * time)
        for harmonic, weight in zip((1, 2, 3), harmonics)
    )


def test_voicing_hysteresis_keeps_a_dip_from_splitting_a_note():
    frames = 100
    f0 = np.full(frames, 440.0)
    periodicity = np.full(frames, 0.9)
    periodicity[40:45] = 0.4

    segments = _segment_notes(f0, periodicity, times(frames))

    assert len(segments) == 1
    assert segments[0][2] == 69


def test_voicing_hysteresis_still_requires_a_confident_start():
    periodicity = np.array([0.4, 0.45, 0.4, 0.4])

    assert not _voiced_mask(periodicity, np.full(4, 440.0), 0.5).any()
    assert _voiced_mask(np.array([0.6, 0.4, 0.4, 0.3]), np.full(4, 440.0), 0.5)[
        :3
    ].all()


def test_octave_correction_lifts_a_subharmonic():
    fmin, fmax = frequency_bounds("alto")
    audio = _harmonic_tone(440.0)
    frames = audio.size // 160

    corrected = _correct_octaves(audio, 16000, [(0, frames - 1, 57)], fmin, fmax)

    assert corrected[0][2] == 69


def test_octave_correction_drops_a_harmonic():
    fmin, fmax = frequency_bounds("alto")
    audio = _harmonic_tone(220.0)
    frames = audio.size // 160

    corrected = _correct_octaves(audio, 16000, [(0, frames - 1, 69)], fmin, fmax)

    assert corrected[0][2] == 57


def test_octave_correction_keeps_a_bright_but_correct_tone():
    fmin, fmax = frequency_bounds("alto")
    audio = _harmonic_tone(440.0, harmonics=(1.0, 3.0, 0.2))
    frames = audio.size // 160

    corrected = _correct_octaves(audio, 16000, [(0, frames - 1, 69)], fmin, fmax)

    assert corrected[0][2] == 69
