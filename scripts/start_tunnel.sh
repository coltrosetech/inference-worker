#!/usr/bin/env bash
# Expose the webapp via a Cloudflare quick tunnel (ephemeral *.trycloudflare.com URL).
# Prints the public URL once the tunnel is connected.
#
# Re-run to get a fresh URL — quick tunnels are regenerated each start.
# For a stable custom domain, switch to a named tunnel (requires a Cloudflare account).
set -euo pipefail
PORT="${WEBAPP_PORT:-8001}"

echo "[tunnel] starting cloudflared → http://localhost:${PORT}"
exec cloudflared tunnel --no-autoupdate --url "http://localhost:${PORT}"
