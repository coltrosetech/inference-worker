#!/usr/bin/env bash
# Container entrypoint — runs bootstrap, then supervisord.
set -euo pipefail

cd /app

# Load .env if present (values from the environment take precedence).
if [ -f /app/.env ]; then
    set -a; source /app/.env; set +a
fi

# Defaults for required env expected by supervisord.conf.
: "${WORKER_PORT:=8000}"
: "${COMFYUI_HOST:=127.0.0.1}"
: "${COMFYUI_INTERNAL_PORT:=8188}"
export WORKER_PORT COMFYUI_HOST COMFYUI_INTERNAL_PORT

# Bootstrap: idempotent setup (populated in later tasks).
if [ -x /app/scripts/bootstrap.sh ]; then
    bash /app/scripts/bootstrap.sh
fi

exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
