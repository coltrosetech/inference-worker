# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl wget ca-certificates tini supervisor ffmpeg libgl1 libglib2.0-0 \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python

# PyTorch 2.10 + CUDA 12.8 (use the index that matches; fall back to default cu124 if needed).
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.10.0 torchvision torchaudio

# ComfyUI installed into /opt/comfyui at image build time (fat image).
ARG COMFYUI_SHA=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/comfyui \
    && cd /opt/comfyui && git checkout "${COMFYUI_SHA}" \
    && python -m pip install -r requirements.txt

# Workflow converter custom node (reused from spec §8.2).
RUN git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git \
        /opt/comfyui/custom_nodes/comfyui-workflow-to-api-converter-endpoint

# App code.
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY worker /app/worker
COPY workflows /app/workflows
COPY configs /app/configs
COPY scripts /app/scripts
COPY supervisord.conf /etc/supervisor/supervisord.conf

RUN python -m pip install -e "/app[dev]"

RUN chmod +x /app/scripts/*.sh

EXPOSE 8000 9090
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/scripts/start.sh"]
