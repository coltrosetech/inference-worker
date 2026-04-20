#!/usr/bin/env bash
# Start the webapp on :8001. Assumes the worker is already running on :8000.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d webapp/frontend/dist ]; then
  echo "⚠️  frontend not built — running scripts/build_frontend.sh"
  bash scripts/build_frontend.sh
fi

if [ -z "${WORKER_API_KEY:-}" ]; then
  set -a
  source .env
  set +a
fi

exec .venv/bin/uvicorn webapp.main:app \
  --host 127.0.0.1 --port 8001 \
  --log-level info
