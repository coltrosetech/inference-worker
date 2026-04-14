#!/usr/bin/env bash
# Idempotent bootstrap. Safe to run on every boot.
set -euo pipefail

cd /app

echo "[bootstrap] 1. env / secrets"
if [ ! -f /app/.env ]; then
    cp /app/.env.example /app/.env
fi
bash /app/scripts/gen_secrets.sh /app/.env
# export only KEY=VAL lines
set -a; source /app/.env; set +a

echo "[bootstrap] 2. comfyui present at ${COMFYUI_PATH}"
if [ ! -f "${COMFYUI_PATH}/main.py" ]; then
    mkdir -p "$(dirname "${COMFYUI_PATH}")"
    git clone https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_PATH}"
    (cd "${COMFYUI_PATH}" && python -m pip install -r requirements.txt)
fi

echo "[bootstrap] 3. workflow-to-api converter node"
CONV_DIR="${COMFYUI_PATH}/custom_nodes/comfyui-workflow-to-api-converter-endpoint"
if [ ! -d "${CONV_DIR}" ]; then
    git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git "${CONV_DIR}"
fi

echo "[bootstrap] 4. custom nodes"
COMFYUI_PATH="${COMFYUI_PATH}" bash /app/scripts/install_custom_nodes.sh /app/configs/custom_nodes.yaml

echo "[bootstrap] 5. models"
python /app/scripts/download_models.py \
    --manifest /app/configs/models.yaml \
    --models-dir "${MODELS_PATH}" \
    --hf-token "${HF_TOKEN:-}"

echo "[bootstrap] 6. symlink models into ComfyUI models dir"
# ComfyUI reads from ${COMFYUI_PATH}/models by default. Symlink our MODELS_PATH over it.
if [ "${COMFYUI_PATH}/models" != "${MODELS_PATH}" ]; then
    # Keep the original (or its contents) reachable if it was populated.
    if [ -d "${COMFYUI_PATH}/models" ] && [ ! -L "${COMFYUI_PATH}/models" ]; then
        # Copy anything already there into MODELS_PATH without overwriting.
        rsync -a --ignore-existing "${COMFYUI_PATH}/models/" "${MODELS_PATH}/" 2>/dev/null || true
        mv "${COMFYUI_PATH}/models" "${COMFYUI_PATH}/models.orig.$(date +%s)" 2>/dev/null || rm -rf "${COMFYUI_PATH}/models"
    fi
    ln -sfn "${MODELS_PATH}" "${COMFYUI_PATH}/models"
fi

echo "[bootstrap] 7. cloudflare tunnel (if token provided)"
if [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
    if ! command -v cloudflared >/dev/null 2>&1; then
        echo "cloudflared not installed; skipping"
    else
        nohup cloudflared tunnel --no-autoupdate run --token "${CLOUDFLARE_TUNNEL_TOKEN}" \
            > /data/logs/cloudflared.log 2>&1 &
    fi
fi

echo "[bootstrap] done"
