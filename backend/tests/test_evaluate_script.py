import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]


def write_sine(path: Path) -> None:
    sr = 8000
    time = np.arange(sr) / sr
    sf.write(str(path), 0.1 * np.sin(2 * np.pi * 440 * time), sr)


def test_partial_benchmark_failure_exits_nonzero(tmp_path):
    write_sine(tmp_path / "good.wav")
    (tmp_path / "good.json").write_text(
        json.dumps({"notes": [{"start": 0.0, "end": 0.5, "pitch": 60}]})
    )
    write_sine(tmp_path / "bad.wav")
    (tmp_path / "bad.json").write_text("{not json")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            str(tmp_path),
            "--mock",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1, result.stderr
    assert "good.wav" in result.stdout
    assert "fail bad.wav" in result.stdout
