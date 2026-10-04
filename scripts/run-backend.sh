#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
exec uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port "${PORT:-8000}" --reload
