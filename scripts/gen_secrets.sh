#!/usr/bin/env bash
# Populate missing secrets in .env. Idempotent: never overwrites existing non-empty values.
set -euo pipefail

ENV_FILE="${1:-.env}"
if [ ! -f "$ENV_FILE" ]; then
    cp .env.example "$ENV_FILE"
fi

gen() { python3 -c "import secrets; print(secrets.token_hex(32))"; }

for KEY in WORKER_API_KEY CALLBACK_HMAC_SECRET; do
    CUR=$(grep -E "^$KEY=" "$ENV_FILE" | cut -d= -f2-)
    if [ -z "${CUR:-}" ]; then
        VAL=$(gen)
        python3 - <<PY
from pathlib import Path
p = Path("$ENV_FILE")
lines = p.read_text().splitlines()
for i, ln in enumerate(lines):
    if ln.startswith("$KEY="):
        lines[i] = "$KEY=$VAL"
        break
else:
    lines.append("$KEY=$VAL")
p.write_text("\n".join(lines) + "\n")
PY
        echo "generated $KEY"
    fi
done
echo "secrets OK in $ENV_FILE"
