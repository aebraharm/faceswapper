#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
exec uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port "${PORT:-8000}" --reload
