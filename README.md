# Inference Worker (Phase 1)

Stateless GPU inference worker for image generation. Runs ComfyUI headlessly
on `127.0.0.1:8188` and exposes a FastAPI service on port `$WORKER_PORT` that
accepts prompt + image from the user's backend, produces a revised image, and
PUTs the result to a signed URL plus a HMAC-signed callback.

## Quick start (any Linux + NVIDIA GPU host)

```bash
git clone <REPO_URL> /workspace/works
cd /workspace/works
cp .env.example .env
bash scripts/gen_secrets.sh .env     # populates missing API key + HMAC secret
docker compose up -d --build
```

Wait ~5–8 minutes on first boot (model download). Check readiness:

```bash
curl -s http://127.0.0.1:8000/v1/health | python -m json.tool
# "ready": true  when warm-up completes
```

### Without Docker

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/bootstrap.sh
bash scripts/start.sh
```

## Environment

All config via `.env`. See `.env.example` for the complete list.
Required secrets (`WORKER_API_KEY`, `CALLBACK_HMAC_SECRET`) are auto-generated
by `scripts/gen_secrets.sh` if missing.

## Request flow

1. User's backend POSTs to `http://<worker>/v1/generate` with bearer auth,
   Idempotency-Key header, and the JSON body documented in
   `docs/superpowers/specs/2026-04-14-img-video-worker-design.md` §3.1.
2. Worker responds `202 Accepted`, enqueues the job.
3. Worker downloads the input, runs inference, PUTs the output to the
   signed `upload_url`, and POSTs a HMAC-signed callback to `callback_url`.

## Presets (Phase 1)

- `edit` — IP-Adapter + Lightning SDXL, prompt-driven edit
- `style` — IP-Adapter style transfer using a reference image

## Development

```bash
pytest                       # unit + component tests (no GPU)
pytest -m "not gpu"          # same, explicit
pytest -m gpu                # integration (requires a running GPU + worker)
ruff check . && pyright      # lint + typecheck
```

## Local e2e

```bash
# Terminal 1 — worker
docker compose up
# Terminal 2 — mock backend
MOCK_HMAC_SECRET="$(grep '^CALLBACK_HMAC_SECRET=' .env | cut -d= -f2)" \
    python scripts/mock_backend.py --port 9100 --dir ./mock_data
# Terminal 3 — integration test
WORKER_URL=http://127.0.0.1:8000 \
  WORKER_API_KEY="$(grep '^WORKER_API_KEY=' .env | cut -d= -f2)" \
  MOCK_BACKEND_URL=http://127.0.0.1:9100 \
  pytest -m gpu tests/integration -v
```

## Observability

- Logs: JSON to stdout (`docker logs` / journald)
- Health: `GET /v1/health`
- Metrics (Phase 2): `GET /v1/metrics` on `$METRICS_PORT`

## Portability to a new machine

```bash
git clone <REPO_URL> /workspace/works && cd /workspace/works
cp .env.example .env && bash scripts/gen_secrets.sh .env
docker compose up -d
```

The three commands are sufficient on any Linux host with the NVIDIA Container Toolkit installed.

## Next (Phase 2, separate plan)

- Video preset (LTX-Video)
- `controlnet` + `inpaint` presets
- Full Prometheus metrics + Grafana dashboard
- CI → GHCR pipeline
- Load testing
