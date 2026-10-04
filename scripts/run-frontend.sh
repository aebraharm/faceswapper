#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT/frontend"
exec npm run dev -- --host 0.0.0.0 --port "${PORT:-5173}"
