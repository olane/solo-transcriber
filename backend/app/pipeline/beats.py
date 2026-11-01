from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
HOP_LENGTH = 512
MIN_BPM = 30.0
MAX_BPM = 300.0
MIN_METER_ACCENTS = 8
METER_SWITCH_MARGIN = 0.05

# How many tracked pulses make up one bar in each supported meter.
METER_BEATS = {"4/4": 4, "3/4": 3, "2/4": 2, "6/8": 6}

# Typical accent shape of a bar. The first beat is strongest; other beats are
# weaker, with a secondary accent on beat 3 in 4/4 and on beats 3/4 in 6/8.
METRIC_TEMPLATES = {
    "4/4": (1.0, 0.15, 0.6, 0.15),
    "3/4": (1.0, 0.35, 0.35),
    "2/4": (1.0, 0.35),
    "6/8": (1.0, 0.1, 0.4, 0.55, 0.1, 0.4),
}

# Prefer the common simple meters when the evidence is close.
METER_PRIOR = {"4/4": 0.08, "3/4": 0.02, "2/4": -0.03, "6/8": -0.10}


@dataclass(frozen=True)
class BeatGrid:
    """A global pulse grid detected on the whole track."""

    bpm: float
    time_signature: str
    beats: tuple[float, ...]
    phase: int = 0

    @property
    def beats_per_bar(self) -> int:
        return METER_BEATS[self.time_signature]

    @property
    def downbeats(self) -> tuple[float, ...]:
        return self.beats[self.phase :: self.beats_per_bar]

    def bar_start_at_or_before(self, time: float) -> float | None:
        """Return the last downbeat at or before ``time``.

        If the first note falls before the first detected downbeat, fall back
        to the first pulse so a note is never engraved before the score start.
        """
        downbeats = self.downbeats
        if not downbeats:
            return None
        index = int(np.searchsorted(downbeats, time + 1e-3, side="right")) - 1
        if index >= 0:
            return downbeats[index]
        if self.beats and self.beats[0] <= time + 1e-3:
            return self.beats[0]
        return None


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    deviation = float(np.std(values))
    if deviation <= 1e-9:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / deviation


def _sample_beats(
    values: np.ndarray, frames: np.ndarray, radius: int = 1
) -> np.ndarray:
    if values.size == 0 or frames.size == 0:
        return np.zeros(frames.size, dtype=float)
    offsets = np.arange(-radius, radius + 1)
    indices = np.clip(
        frames[:, None] + offsets[None, :], 0, values.size - 1
    )
    return np.max(values[indices], axis=1)


def infer_meter(accents: np.ndarray) -> tuple[str, int, float]:
    """Pick the meter and downbeat phase that best explain beat accents.

    Returns ``(time_signature, phase, score)``. The accents are correlation-
    matched against a simple metric template for each supported meter, so this
    stays deterministic and cheap enough to unit test without audio.
    """
    values = _zscore(np.asarray(accents, dtype=float))
    if values.size < MIN_METER_ACCENTS:
        return "4/4", 0, 0.0

    indices = np.arange(values.size)
    best: tuple[str, int, float] = ("4/4", 0, -np.inf)
    meter_scores: dict[str, tuple[int, float]] = {}
    for meter, template in METRIC_TEMPLATES.items():
        period = len(template)
        if values.size < period * 2:
            continue
        template_values = _zscore(np.asarray(template, dtype=float))
        for phase in range(period):
            expected = template_values[(indices - phase) % period]
            denominator = float(np.linalg.norm(values) * np.linalg.norm(expected))
            if denominator <= 1e-9:
                continue
            correlation = float(np.dot(values, expected) / denominator)
            score = correlation + METER_PRIOR.get(meter, 0.0)
            if meter not in meter_scores or score > meter_scores[meter][1]:
                meter_scores[meter] = (phase, score)
            if score > best[2]:
                best = (meter, phase, score)

    # Only leave 4/4 when another meter explains the accents clearly better.
    four = meter_scores.get("4/4")
    if best[0] != "4/4" and four is not None and best[2] < four[1] + METER_SWITCH_MARGIN:
        return "4/4", four[0], four[1]
    return best


def detect_beat_grid(path: Path) -> BeatGrid | None:
    """Track the pulse and detect downbeats on a whole mix or track.

    The tempo is fitted to the beats we actually group into bars instead of
    using a separate autocorrelation estimate that can disagree with them.
    """
    import librosa

    try:
        audio, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None
    if audio.size == 0:
        return None

    onset_envelope = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    )
    if onset_envelope.size < 2:
        return None

    try:
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=HOP_LENGTH,
            units="frames",
            trim=False,
        )
    except Exception:
        return None

    frames = np.asarray(beat_frames, dtype=int).ravel()
    if frames.size < 2:
        return None
    beat_times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP_LENGTH)

    intervals = np.diff(beat_times)
    valid = intervals[
        (intervals >= 60.0 / MAX_BPM) & (intervals <= 60.0 / MIN_BPM)
    ]
    if valid.size:
        # Fit a line through the beat indices instead of using a single median
        # interval. Librosa's frame hop quantises each beat time (22 frames,
        # then 21 frames at 120 BPM), and averaging that error over the whole
        # track keeps the tempo from drifting flat.
        nominal = float(np.median(valid))
        steps = np.maximum(1, np.rint(intervals / nominal).astype(int))
        indices = np.concatenate(([0], np.cumsum(steps)))
        slope = float(np.polyfit(indices, beat_times, 1)[0])
        pulse_bpm = 60.0 / slope if slope > 0 else 60.0 / nominal
    else:
        pulse_bpm = float(np.atleast_1d(tempo)[0])
    if not MIN_BPM <= pulse_bpm <= MAX_BPM:
        return None

    low_envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmin=30,
        fmax=160,
        n_mels=16,
    )
    accents = 0.65 * _zscore(_sample_beats(low_envelope, frames))
    accents += 0.35 * _zscore(_sample_beats(onset_envelope, frames))
    if accents.size >= MIN_METER_ACCENTS * 2:
        # Onset detection has no history for the first beat and no lookahead
        # for the last, so their accents are unreliable. Neutralise them
        # instead of letting one missing attack flip the whole bar phase.
        edge = float(np.median(accents[1:-1]))
        accents[0] = edge
        accents[-1] = edge

    time_signature, phase, _ = infer_meter(accents)
    if beat_times.size < METER_BEATS[time_signature] * 2:
        time_signature = "4/4"
        phase = 0

    # A 6/8 bar contains three quarter notes, i.e. six detected pulses.
    bpm = pulse_bpm / 2.0 if time_signature == "6/8" else pulse_bpm
    if not MIN_BPM <= bpm <= MAX_BPM:
        return None

    return BeatGrid(
        bpm=round(bpm, 1),
        time_signature=time_signature,
        beats=tuple(float(value) for value in beat_times),
        phase=phase,
    )
