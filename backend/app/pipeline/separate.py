from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from .. import config

STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]
SUGGESTED_STEM = "other"

# Broad concert-pitch sax register used when guessing which stem carries the
# solo. pyin only tracks a single f0, so this is a hint, not a decision.
STEM_FMIN_MIDI = 40
STEM_FMAX_MIDI = 92
STEM_SAMPLE_SECONDS = 30.0
STEM_MAX_JUMP_SEMITONES = 2.0
STEM_TIE_MARGIN = 1.05


def order_stems(names: Iterable[str]) -> list[str]:
    """Order stem names by STEM_ORDER, then alphabetically for extras."""
    found = set(names)
    ordered = [name for name in STEM_ORDER if name in found]
    ordered += sorted(found - set(ordered))
    return ordered


def _sax_likeness(path: Path) -> float:
    """Score how strongly a stem behaves like a solo sax line in its register.

    The score combines how much of the stem is tracked as a single f0, how
    confident that tracking is, and how continuous the pitch is. Polyphonic
    bleed and drums fall out because they do not sustain a stable f0.
    """
    import librosa

    try:
        audio, sr = librosa.load(
            str(path), sr=22050, mono=True, duration=STEM_SAMPLE_SECONDS
        )
    except Exception:
        return 0.0
    if audio.size < sr // 4:
        return 0.0

    try:
        f0, _, voiced_prob = librosa.pyin(
            audio,
            fmin=float(librosa.midi_to_hz(STEM_FMIN_MIDI)),
            fmax=float(librosa.midi_to_hz(STEM_FMAX_MIDI)),
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
    except Exception:
        return 0.0

    import numpy as np

    voiced = np.isfinite(f0)
    if voiced.sum() < 2:
        return 0.0
    coverage = float(voiced.mean())
    confidence = float(np.mean(np.asarray(voiced_prob)[voiced]))
    midi = 69.0 + 12.0 * np.log2(np.maximum(f0[voiced], 1e-9) / 440.0)
    jumps = np.abs(np.diff(midi))
    continuity = (
        float(np.mean(jumps <= STEM_MAX_JUMP_SEMITONES)) if jumps.size else 0.0
    )
    return coverage * confidence * (0.5 + 0.5 * continuity)


def suggest_stem(stems_dir: Path, names: Iterable[str]) -> str:
    """Pick the stem most likely to carry the sax solo.

    Falls back to ``other`` when nothing stands out, so a near-tie does not
    silently switch the suggestion to vocals or a backing pad.
    """
    available = list(names)
    if not available:
        return SUGGESTED_STEM
    scores = {name: _sax_likeness(stems_dir / f"{name}.wav") for name in available}
    best = max(available, key=lambda name: scores[name])
    if scores[best] <= 0.0:
        return SUGGESTED_STEM if SUGGESTED_STEM in available else best
    if best == SUGGESTED_STEM or SUGGESTED_STEM not in available:
        return best
    if scores[best] <= scores[SUGGESTED_STEM] * STEM_TIE_MARGIN:
        return SUGGESTED_STEM
    return best


def pick_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _copy_mock_stems(source: Path, stems_dir: Path) -> list[str]:
    stems_dir.mkdir(parents=True, exist_ok=True)
    names = order_stems(["other", "vocals", "bass"])
    for name in names:
        shutil.copyfile(source, stems_dir / f"{name}.wav")
    return names


def _run_demucs(source: Path, workdir: Path, device: str, log_path: Path) -> None:
    source = source.resolve()
    log_path = log_path.resolve()
    out_dir = (workdir / "demucs").resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        config.DEMUCS_MODEL,
        "-d",
        device,
        "-o",
        str(out_dir),
        "--filename",
        "{stem}.{ext}",
        "--shifts",
        str(max(1, config.DEMUCS_SHIFTS)),
        "--overlap",
        str(config.DEMUCS_OVERLAP),
        str(source),
    ]
    with log_path.open("a") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        result = subprocess.run(cmd, cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = log_path.read_text(errors="replace")[-2000:]
        raise RuntimeError(f"Demucs failed (device={device}).\n{tail}")


def separate_stems(source_wav: Path, workdir: Path, mock: bool = False) -> list[str]:
    workdir = workdir.resolve()
    stems_dir = workdir / "stems"
    if mock:
        return _copy_mock_stems(source_wav.resolve(), stems_dir)

    if importlib.util.find_spec("demucs") is None:
        raise RuntimeError(
            "Demucs is not installed. Install the ML extras with: uv sync --extra ml"
        )

    log_path = workdir / "demucs.log"
    device = pick_device(config.DEVICE)
    try:
        _run_demucs(source_wav, workdir, device, log_path)
    except RuntimeError:
        if device == "cpu":
            raise
        _run_demucs(source_wav, workdir, "cpu", log_path)

    out_dir = workdir / "demucs"
    found = {path.stem: path for path in out_dir.rglob("*.wav")}
    if not found:
        raise RuntimeError("Demucs did not produce any stems")

    stems_dir.mkdir(parents=True, exist_ok=True)
    ordered = order_stems(found)
    for name in ordered:
        shutil.copyfile(found[name], stems_dir / f"{name}.wav")
    return ordered
