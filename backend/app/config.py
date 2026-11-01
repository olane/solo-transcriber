from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("ST_DATA_DIR", ROOT / "data")).resolve()
MOCK = os.environ.get("ST_MOCK", "").lower() in {"1", "true", "yes"}
DEMUCS_MODEL = os.environ.get("ST_DEMUCS_MODEL", "htdemucs_6s")
# Equivariant stabilisation: more shifts trade separation time for quality.
DEMUCS_SHIFTS = int(os.environ.get("ST_DEMUCS_SHIFTS", "4"))
DEMUCS_OVERLAP = float(os.environ.get("ST_DEMUCS_OVERLAP", "0.25"))
DEVICE = os.environ.get("ST_DEVICE", "auto")
CREPE_MODEL = os.environ.get("ST_CREPE_MODEL", "tiny")
MAX_UPLOAD_MB = int(os.environ.get("ST_MAX_UPLOAD_MB", "300"))
