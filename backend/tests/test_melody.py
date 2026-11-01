import numpy as np
import pytest
import soundfile as sf

from app.pipeline.melody import _detect_onsets, _segment_notes, transcribe_melody
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
