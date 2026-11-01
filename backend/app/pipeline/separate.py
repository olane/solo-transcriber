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


def order_stems(names: Iterable[str]) -> list[str]:
    """Order stem names by STEM_ORDER, then alphabetically for extras."""
    found = set(names)
    ordered = [name for name in STEM_ORDER if name in found]
    ordered += sorted(found - set(ordered))
    return ordered


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
