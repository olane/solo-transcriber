#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

trap 'kill 0' EXIT INT TERM

uv run --extra ml uvicorn app.main:app --app-dir backend --reload --port 8000 &
(cd frontend && npm run dev)
