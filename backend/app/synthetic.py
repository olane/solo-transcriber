from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .pipeline.transcribe import INSTRUMENT_RANGES

SAMPLE_RATE = 44100
LEAD_SECONDS = 0.25
TAIL_SECONDS = 0.35
BLEED_GAIN = 0.08
HARMONICS = 12


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    notes: list[dict[str, Any]]
    bleed: bool = False


def _event(
    start: float, end: float, pitch: int, **extra: Any
) -> dict[str, Any]:
    event = {
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "pitch": int(pitch),
        "velocity": 90,
    }
    event.update(extra)
    return event


def _fit(pitch: int, low: int, high: int) -> int:
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return int(pitch)


def build_scenarios(instrument: str = "alto") -> list[Scenario]:
    low, high = INSTRUMENT_RANGES[instrument]
    mid = _fit((low + high) // 2, low, high)

    def note(start: float, end: float, pitch: int, **extra: Any) -> dict[str, Any]:
        return _event(start, end, _fit(pitch, low, high), **extra)

    repeated = [note(0.20 + i * 0.28, 0.44 + i * 0.28, mid) for i in range(8)]
    legato = [note(0.20 + i * 0.30, 0.50 + i * 0.30, mid) for i in range(6)]
    vibrato = [
        note(
            0.20 + i * 0.92,
            1.00 + i * 0.92,
            pitch,
            vibrato_cents=35,
            vibrato_hz=5.5,
        )
        for i, pitch in enumerate((mid - 3, mid, mid + 4, mid + 7))
    ]
    chromatic = [
        note(0.20 + i * 0.28, 0.48 + i * 0.28, mid - 4 + i) for i in range(8)
    ]
    run_step = 60.0 / 160.0 / 4.0
    scale = (0, 2, 4, 5, 7, 9, 11, 12)
    fast_run = [
        note(
            0.20 + i * run_step,
            0.20 + (i + 1) * run_step,
            mid - 5 + scale[i % len(scale)],
        )
        for i in range(12)
    ]
    octave_trill = [
        note(0.20 + i * 0.13, 0.33 + i * 0.13, mid + (12 if i % 2 else 0))
        for i in range(10)
    ]
    scoops = [
        note(
            0.20 + i * 0.82,
            0.90 + i * 0.82,
            mid + 2,
            scoop_cents=200,
            scoop_seconds=0.07,
        )
        for i in range(4)
    ]
    bleed_notes = [
        note(
            0.20 + i * 0.45,
            0.58 + i * 0.45,
            mid + (0, 2, 4, 2, 0, -2)[i],
        )
        for i in range(6)
    ]
    mixed = [
        note(0.20, 0.48, mid),
        note(0.48, 0.76, mid),
        note(0.86, 1.66, mid + 4, vibrato_cents=35),
        note(1.78, 2.06, mid + 2),
        note(2.06, 2.34, mid + 3),
        note(2.34, 2.62, mid + 5),
        note(2.62, 3.42, mid + 7, vibrato_cents=25),
    ]

    def phrase(
        start: float, step: float, pitches: tuple[int, ...], length: float
    ) -> list[dict[str, Any]]:
        return [
            note(start + index * step, start + index * step + length, pitch)
            for index, pitch in enumerate(pitches)
        ]

    wide_leaps = phrase(
        0.20, 0.55, (mid, mid + 12, mid - 5, mid + 7, mid - 12, mid + 9), 0.42
    )
    low_register = phrase(0.20, 0.65, (low, low + 2, low + 4, low + 7), 0.55)
    high_register = phrase(
        0.20, 0.65, (high, high - 3, high - 5, high - 7), 0.55
    )
    staccato = phrase(
        0.20,
        0.30,
        (mid, mid + 2, mid + 4, mid + 5, mid + 4, mid + 2, mid, mid - 2),
        0.08,
    )
    long_tones = phrase(0.20, 1.75, (mid, mid + 5, mid - 3), 1.55)
    syncopated = [
        note(start, start + length, pitch)
        for start, length, pitch in (
            (0.20, 0.18, mid),
            (0.45, 0.30, mid + 2),
            (0.85, 0.18, mid + 4),
            (1.10, 0.30, mid + 5),
            (1.50, 0.18, mid + 4),
            (1.75, 0.30, mid + 2),
        )
    ]
    semitone_trill = [
        note(0.20 + i * 0.11, 0.29 + i * 0.11, mid + (i % 2)) for i in range(12)
    ]
    arpeggio = phrase(
        0.20, 0.26, (mid, mid + 4, mid + 7, mid + 12, mid + 7, mid + 4), 0.22
    )
    descending_scale = phrase(
        0.20,
        0.24,
        (mid + 12, mid + 10, mid + 9, mid + 7, mid + 5, mid + 4, mid + 2, mid),
        0.20,
    )
    dotted_rhythm = [
        note(start, start + length, pitch)
        for start, length, pitch in (
            (0.20, 0.42, mid),
            (0.62, 0.18, mid + 2),
            (0.95, 0.42, mid + 4),
            (1.37, 0.18, mid + 5),
            (1.70, 0.42, mid + 7),
            (2.12, 0.18, mid + 4),
        )
    ]
    triplet_step = 60.0 / 120.0 / 3.0
    triplets = phrase(
        0.20,
        triplet_step,
        (mid, mid + 2, mid + 4, mid + 5, mid + 7, mid + 5, mid + 4, mid + 2, mid),
        triplet_step * 0.9,
    )
    falls = [
        note(
            0.20 + i * 0.82,
            0.90 + i * 0.82,
            mid + 4,
            fall_cents=200,
            fall_seconds=0.18,
        )
        for i in range(4)
    ]
    wide_vibrato = [
        note(
            0.20 + i * 1.1,
            1.20 + i * 1.1,
            pitch,
            vibrato_cents=80,
            vibrato_hz=4.8,
        )
        for i, pitch in enumerate((mid - 2, mid + 3, mid + 7))
    ]
    grace_notes = [
        event
        for i in range(4)
        for event in (
            note(0.20 + i * 0.95, 0.26 + i * 0.95, mid + 2),
            note(0.26 + i * 0.95, 0.81 + i * 0.95, mid),
        )
    ]

    return [
        Scenario(
            "repeated-articulated",
            "Eight repeated notes on one pitch with 40 ms gaps",
            repeated,
        ),
        Scenario(
            "repeated-legato",
            "Six repeated notes re-articulated without gaps",
            legato,
        ),
        Scenario(
            "vibrato",
            "Four long notes with 35-cent, 5.5 Hz vibrato",
            vibrato,
        ),
        Scenario(
            "chromatic",
            "Legato chromatic run through eight semitones",
            chromatic,
        ),
        Scenario(
            "fast-run",
            "Twelfth-note run as sixteenths at 160 BPM",
            fast_run,
        ),
        Scenario(
            "octave-trill",
            "Fast octave alternation; octave repair must not flatten it",
            octave_trill,
        ),
        Scenario(
            "scoops",
            "Notes entered with a 200-cent upward scoop",
            scoops,
        ),
        Scenario(
            "bleed",
            "Sax line with a quiet, attacking accompaniment",
            bleed_notes,
            bleed=True,
        ),
        Scenario(
            "mixed-phrase",
            "Repeated notes, vibrato and a legato run",
            mixed,
        ),
        Scenario(
            "wide-leaps",
            "Six notes alternating upward and downward leaps up to an octave",
            wide_leaps,
        ),
        Scenario(
            "low-register",
            "Four sustained notes around the bottom of the range",
            low_register,
        ),
        Scenario(
            "high-register",
            "Four sustained notes around the top of the range",
            high_register,
        ),
        Scenario(
            "staccato",
            "Eight 80 ms staccato notes with clear gaps",
            staccato,
        ),
        Scenario(
            "long-tones",
            "Three long sustained notes with 200 ms gaps",
            long_tones,
        ),
        Scenario(
            "syncopated",
            "Off-beat short and long notes in a syncopated rhythm",
            syncopated,
        ),
        Scenario(
            "semitone-trill",
            "Fast semitone alternation that must not be merged",
            semitone_trill,
        ),
        Scenario(
            "arpeggio",
            "Ascending and descending arpeggio spanning an octave",
            arpeggio,
        ),
        Scenario(
            "descending-scale",
            "Diatonic descending scale over an octave",
            descending_scale,
        ),
        Scenario(
            "dotted-rhythm",
            "Alternating dotted and short notes",
            dotted_rhythm,
        ),
        Scenario(
            "triplets",
            "Eighth-note triplets at 120 BPM",
            triplets,
        ),
        Scenario(
            "falls",
            "Notes ending with a 200-cent downward fall",
            falls,
        ),
        Scenario(
            "wide-vibrato",
            "Three long notes with 80-cent, 4.8 Hz vibrato",
            wide_vibrato,
        ),
        Scenario(
            "grace-notes",
            "Short 60 ms grace notes before each sustained note",
            grace_notes,
        ),
    ]


def _envelope(length: int, sr: int, attack: float = 0.012, release: float = 0.035) -> np.ndarray:
    envelope = np.ones(length)
    attack_frames = min(length // 3, max(1, int(attack * sr)))
    release_frames = min(length // 3, max(1, int(release * sr)))
    envelope[:attack_frames] = np.linspace(0.0, 1.0, attack_frames)
    envelope[length - release_frames :] = np.linspace(1.0, 0.0, release_frames)
    return envelope


def _pitch_curve(note: dict[str, Any], length: int, sr: int) -> np.ndarray:
    time = np.arange(length) / sr
    midi = np.full(length, float(note["pitch"]))
    scoop_seconds = float(note.get("scoop_seconds", 0.0))
    scoop_cents = float(note.get("scoop_cents", 0.0))
    if scoop_cents > 0 and scoop_seconds > 0:
        scoop_frames = min(length, max(1, int(scoop_seconds * sr)))
        start_pitch = note["pitch"] - scoop_cents / 100.0
        midi[:scoop_frames] = start_pitch + (
            note["pitch"] - start_pitch
        ) * np.linspace(0.0, 1.0, scoop_frames)
    vibrato_cents = float(note.get("vibrato_cents", 0.0))
    if vibrato_cents > 0:
        vibrato_hz = float(note.get("vibrato_hz", 5.5))
        ramp = np.clip((time - scoop_seconds) / 0.10, 0.0, 1.0)
        midi += (
            vibrato_cents / 100.0
            * np.sin(2.0 * np.pi * vibrato_hz * time)
            * ramp
        )
    fall_cents = float(note.get("fall_cents", 0.0))
    fall_seconds = float(note.get("fall_seconds", 0.0))
    if fall_cents > 0 and fall_seconds > 0:
        fall_frames = min(length, max(1, int(fall_seconds * sr)))
        drop = np.linspace(0.0, fall_cents / 100.0, fall_frames)
        midi[length - fall_frames :] -= drop
    return 440.0 * 2 ** ((midi - 69.0) / 12.0)


def _render_note(
    note: dict[str, Any],
    length: int,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    frequency = _pitch_curve(note, length, sr)
    phase = np.cumsum(2.0 * np.pi * frequency / sr)
    signal = np.zeros(length)
    for harmonic in range(1, HARMONICS + 1):
        signal += (1.0 / harmonic ** 1.1) * np.sin(harmonic * phase)
    envelope = _envelope(length, sr)
    signal *= envelope
    breath = rng.normal(0.0, 1.0, length)
    breath = np.convolve(breath, np.ones(6) / 6.0, mode="same")
    signal += 0.02 * breath * envelope
    return signal


def _bleed(length: int, sr: int, low: int) -> np.ndarray:
    audio = np.zeros(length)
    chord = (low + 7, low + 12, low + 16)
    for start in np.arange(0.0, length / sr, 0.5):
        for index, pitch in enumerate(chord):
            begin = int(start * sr)
            count = min(length - begin, int((0.45 - index * 0.04) * sr))
            if count <= 0:
                continue
            frequency = 440.0 * 2 ** ((pitch - 69.0) / 12.0)
            time = np.arange(count) / sr
            tone = 0.5 * np.sin(2.0 * np.pi * frequency * time) * np.exp(-3.0 * time)
            audio[begin : begin + count] += tone
    return audio


def render_scenario(
    scenario: Scenario,
    instrument: str = "alto",
    sr: int = SAMPLE_RATE,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed + zlib.crc32(scenario.name.encode()))
    end = max(LEAD_SECONDS + note["end"] for note in scenario.notes)
    length = int((end + TAIL_SECONDS) * sr)
    audio = np.zeros(length)
    for note in scenario.notes:
        begin = int(round((LEAD_SECONDS + note["start"]) * sr))
        stop = int(round((LEAD_SECONDS + note["end"]) * sr))
        count = max(1, min(length - begin, stop - begin))
        audio[begin : begin + count] += _render_note(note, count, sr, rng)
    if scenario.bleed:
        low, _ = INSTRUMENT_RANGES[instrument]
        audio += BLEED_GAIN * _bleed(length, sr, low)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio *= 0.6 / peak
    return audio


def write_benchmark(
    out_dir: Path | str,
    instrument: str = "alto",
    seed: int = 7,
    only: list[str] | None = None,
    sr: int = SAMPLE_RATE,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios(instrument)
    if only:
        wanted = set(only)
        known = {scenario.name for scenario in scenarios}
        unknown = wanted - known
        if unknown:
            raise ValueError(f"Unknown scenario(s): {', '.join(sorted(unknown))}")
        scenarios = [scenario for scenario in scenarios if scenario.name in wanted]

    entries: list[dict[str, Any]] = []
    for scenario in scenarios:
        audio = render_scenario(scenario, instrument, sr, seed)
        wav_path = out / f"{scenario.name}.wav"
        sf.write(str(wav_path), audio, sr)
        notes = [
            {
                "start": round(LEAD_SECONDS + note["start"], 3),
                "end": round(LEAD_SECONDS + note["end"], 3),
                "pitch": note["pitch"],
                "velocity": note["velocity"],
            }
            for note in scenario.notes
        ]
        payload = {
            "scenario": scenario.name,
            "description": scenario.description,
            "instrument": instrument,
            "seed": seed,
            "notes": notes,
        }
        json_path = out / f"{scenario.name}.json"
        json_path.write_text(json.dumps(payload, indent=2))
        entries.append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "wav": wav_path.name,
                "json": json_path.name,
                "notes": len(notes),
            }
        )

    manifest = {
        "instrument": instrument,
        "seed": seed,
        "sample_rate": sr,
        "scenarios": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
