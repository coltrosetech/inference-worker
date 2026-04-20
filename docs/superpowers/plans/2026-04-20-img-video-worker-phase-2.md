# Image/Video Worker — Phase 2 Implementation Plan

**Status:** Draft — 2026-04-20
**Predecessor:** `2026-04-14-img-video-worker-phase-1.md` (Phase 1 shipped; see that
plan for foundation tasks).
**Reference spec:** `docs/superpowers/specs/2026-04-14-img-video-worker-design.md`.

## Scope

Phase 1 shipped: `edit` + `style` presets with IP-Adapter, FastAPI service,
HMAC callbacks, auto-warmup, readiness gating. This plan (Phase 2) extends the
worker with three new presets, full observability, stable external exposure,
and a CI/CD path to GHCR.

### Explicitly in scope

1. **`ltx_video` preset** — image-to-video with LTX-Video (§5.5 of spec).
2. **`controlnet` preset** — structure-preserving redraw via ControlNet Union
   SDXL ProMax (§5.3 of spec).
3. **`inpaint` preset** — mask-scoped region edit (§5.4 of spec).
4. **Prometheus metrics** — `:METRICS_PORT/metrics` endpoint with the metric
   set from §7.2 of the spec.
5. **Cloudflare Tunnel** — stable external URL per vast.ai instance.
6. **GHCR CI pipeline** — GitHub Actions builds the fat image, pushes to
   `ghcr.io/coltrosetech/inference-worker`, tags by git sha + `latest`.
7. **Load testing** — establish p50/p95 latency baselines per preset under
   sustained load (target: SLOs from §6.7 of the spec).

### Explicitly out of scope (deferred to Phase 3+)

- Raw-Diffusers port of the hottest pipeline (`torch.compile`, SFast).
- True batching across the queue.
- Additional presets (`upscale`, `background_remove`, `face_restore`).
- FLUX.1-Kontext premium mode.
- Self-hosted GPU runners for CI.

## Sprint 2 — New presets

Each preset follows the pattern established in Phase 1 Sprint 1 (Task 20–23):
workflow JSON in API format, `Preset` subclass with typed `Parameters`,
injection logic in `inject()`, registry entry, tests.

### Task P2-1: Extend model manifest for Phase 2 assets

Add to `configs/models.yaml`:

| Name | Size | Dest | Purpose |
|---|---|---|---|
| ControlNet Union SDXL ProMax | 2.5 GB | `controlnet/cn_union_sdxl_promax.safetensors` | Single-file multi-task CN |
| Depth-Anything-V2 (small) | 400 MB | `depth/depth_anything_v2_vits.pth` | Depth preprocessor |
| DWPose (body+face+hand) | 200 MB | `dwpose/` (multi-file) | Pose preprocessor |
| IP-Adapter Plus FaceID SDXL | 700 MB | `ipadapter/ip-adapter-plus-faceid_sdxl.safetensors` | Face-preserving IP-Adapter |
| LTX-Video 0.9.5 | 9 GB | `diffusion_models/ltx_video_0_9_5.safetensors` | Video model |
| LTX-Video VAE | 800 MB | `vae/ltx_video_vae.safetensors` | LTX VAE |
| T5-XXL fp8 encoder | 4.5 GB | `text_encoders/t5xxl_fp8_e4m3fn.safetensors` | LTX text encoder |

Total +18 GB. Update `scripts/download_models.py` tests with the new entries
(no code change needed — the script already iterates the manifest).

**Acceptance:** `python scripts/download_models.py --models-dir <dir>` idempotently
fetches all new files; second run is a no-op.

### Task P2-2: Extend custom nodes manifest

Add to `configs/custom_nodes.yaml`:

```yaml
- name: ComfyUI-Advanced-ControlNet
  repo: https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git
  ref: main
  post_install: "python -m pip install -r requirements.txt || true"

- name: comfyui_controlnet_aux
  repo: https://github.com/Fannovel16/comfyui_controlnet_aux.git
  ref: main
  post_install: "python -m pip install -r requirements.txt || true"

- name: ComfyUI-LTXVideo
  repo: https://github.com/Lightricks/ComfyUI-LTXVideo.git
  ref: main
  post_install: "python -m pip install -r requirements.txt || true"

- name: ComfyUI-VideoHelperSuite
  repo: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
  ref: main
  post_install: "python -m pip install -r requirements.txt || true"
```

**Acceptance:** `bash scripts/install_custom_nodes.sh configs/custom_nodes.yaml`
clones the four repos into `$COMFYUI_PATH/custom_nodes/`. ComfyUI restart
exposes `ControlNetLoaderAdvanced`, `AIO_Preprocessor`, `LTXVModelLoader`,
`VHS_VideoCombine` (and siblings) in `/object_info`.

### Task P2-3: `controlnet` preset

**Files:**
- `workflows/controlnet.json` (API format)
- `worker/presets/controlnet.py`
- `tests/test_presets/test_controlnet.py`

**Workflow graph:**

```
LoadImage(input) ─┐
                  ├→ AIO_Preprocessor(preprocessor={canny|depth|dwpose|lineart|scribble})
                  │      ↓
                  │   ControlNetLoaderAdvanced(cn_union_sdxl_promax.safetensors)
                  │      ↓
                  ├→ ACN_ControlNetApplyAdvanced(positive, negative, control_image,
                  │      strength, start, end)
CheckpointLoaderSimple → model + clip + vae
CLIPTextEncode (positive, negative)
VAEEncode(input) → latent
KSampler(model, positive_cn, negative_cn, latent, seed, steps, cfg, denoise)
VAEDecode → SaveImage
```

**`ControlnetPreset.Parameters`:**

```python
class Parameters(Preset.BaseParameters):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    controlnet_type: Literal["canny", "depth", "dwpose", "lineart", "scribble"] = "canny"
    controlnet_strength: float = Field(default=0.8, ge=0.0, le=2.0)
    strength: float = Field(default=0.7, ge=0.0, le=1.0)  # KSampler denoise
    steps: int = Field(default=6, ge=1, le=50)
    cfg: float = Field(default=1.8, ge=0.0, le=15.0)
```

**`ControlnetPreset.inject()`:** set `input_image.image`, `preprocessor.preprocessor`
(via `controlnet_type`), `controlnet_apply.strength`, standard prompt/sampler
knobs. Branch on `is_api_format()` mirroring `EditPreset`.

**Tests** (in test_presets/test_controlnet.py):
- Params require prompt; reject out-of-range `controlnet_strength`.
- inject sets preprocessor name for each `controlnet_type` value.
- Workflow wires preprocessor → CN apply → sampler.
- Unknown `controlnet_type` is rejected by pydantic.

### Task P2-4: `inpaint` preset

**Files:** `workflows/inpaint.json`, `worker/presets/inpaint.py`, `tests/test_presets/test_inpaint.py`.

**Workflow graph:**

```
LoadImage(input)  ─┐
LoadImage(mask)  ─ ┼→ GrowMask(grow_mask_px) ─┐
                   │                          │
                   └→ VAEEncodeForInpaint(image, vae, mask, grow_mask_for_inpaint=True)
                      ↓
CheckpointLoaderSimple → model + clip + vae
CLIPTextEncode(positive, negative)
KSampler(model, positive, negative, latent)
VAEDecode → ImageCompositeMasked(input, decoded, mask) → SaveImage
```

**`InpaintPreset.Parameters`:**

```python
class Parameters(Preset.BaseParameters):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    grow_mask_px: int = Field(default=8, ge=0, le=128)
    strength: float = Field(default=0.9, ge=0.0, le=1.0)
    steps: int = Field(default=6, ge=1, le=50)
    cfg: float = Field(default=1.8, ge=0.0, le=15.0)
```

**`InpaintPreset.inject()`:** Requires `mask_image` in `InputPaths`; raises
`ValueError` otherwise (mirror Style's `reference_image` check). Sets
`grow_mask.expand` from `grow_mask_px`.

**API contract note:** `mask_image_url` becomes REQUIRED for this preset
(already optional at the request layer per spec §3.1 — validation happens
in the preset). Surface clearly in error response:
`{"error":{"code":"INVALID_PARAMETERS","message":"inpaint preset requires mask_image_url"}}`.

**Tests:**
- Params defaults + bounds for grow_mask_px.
- inject rejects missing `mask_image` in `InputPaths`.
- Workflow includes `ImageCompositeMasked` so only the masked region changes.

### Task P2-5: `ltx_video` preset

**Files:** `workflows/ltx_video.json`, `worker/presets/ltx_video.py`, `tests/test_presets/test_ltx_video.py`.

**Workflow graph:**

```
LoadImage(input) → LTXVImageToVideo (bundled by ComfyUI-LTXVideo)
LTXVModelLoader(ltx_video_0_9_5.safetensors)
LTXVVAELoader(ltx_video_vae.safetensors)
T5XXLTextEncoder(t5xxl_fp8_e4m3fn.safetensors)
LTXVPromptEncode(positive, negative) via T5
LTXVConditioningApply(image, latent_dim)
LTXVSampler(num_frames, steps, cfg, stg_rescale, seed, guidance_scale)
LTXVDecode → VHS_VideoCombine(format="video/h264-mp4", fps) → SaveVideo
```

**`LtxVideoPreset.Parameters`:**

```python
class Parameters(Preset.BaseParameters):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    num_frames: Literal[25, 49, 97, 121] = 97
    fps: int = Field(default=24, ge=8, le=60)
    guidance_scale: float = Field(default=3.0, ge=0.0, le=20.0)
    steps: int = Field(default=40, ge=1, le=150)
    width: int = Field(default=768, ge=256, le=1216)
    height: int = Field(default=512, ge=256, le=704)
```

Update `Preset.output_extension` / `output_content_type` for video:
`"mp4"` / `"video/mp4"`. `Executor._resolve_output_path` already scans `videos`
and `gifs` buckets — no change.

**Tests:**
- Params accept only the four allowed `num_frames`.
- Resolution bound (width ≤ 1216, height ≤ 704 per spec).
- inject sets sampler steps, guidance, num_frames, fps; sets LoadImage path.

**VRAM note:** LTX-Video + T5-XXL fp8 is ~13 GB peak. On RTX 5080 (16 GB) this
works but leaves ~2 GB headroom. Add a `mode != IMAGE` guard in the
`ModelManager.ensure()` flow so image models are released before video starts
and vice versa. Implementation: `ModelManager.ensure(mode)` POSTs
`/free` to ComfyUI when switching mode. Record the swap in
`model_swap_total{from,to}` counter (Task P2-8).

### Task P2-6: Extend preset registry + warm-up

- Register `ControlnetPreset`, `InpaintPreset`, `LtxVideoPreset` in
  `worker/presets/__init__.py`.
- `worker/pipeline/warmup.py` already iterates `PRESETS` by default — no
  changes needed. But:
  - For `inpaint`, warmup needs a stub mask (solid white). Extend
    `_ensure_stub_image` to generate an all-white 512×512 PNG as the mask.
  - For `ltx_video`, reduce warmup timeout bound appropriately (video sampling
    is 30–90 s); add a per-preset `warmup_timeout_sec` override hook.
  - Add config knob `WARMUP_PRESETS` (comma list, empty = all) so operators
    can trim warm-up on constrained instances.

**Tests:**
- `test_warmup_stub_mask_is_white` — mask is a valid grayscale PNG.
- `test_warmup_per_preset_timeout_override` — video preset uses longer timeout.

## Sprint 3 — Observability

### Task P2-7: Prometheus metrics registry

**New files:** `worker/core/metrics.py`, `tests/test_metrics.py`.

**Metrics (from spec §7.2):**

```python
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

registry = CollectorRegistry()

inference_duration_seconds = Histogram(
    "inference_duration_seconds",
    "End-to-end wall time for a job (including IO).",
    labelnames=("preset",),
    buckets=(0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256),
    registry=registry,
)
stage_duration_seconds = Histogram(
    "stage_duration_seconds", "Per-stage wall time.",
    labelnames=("stage",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
    registry=registry,
)
gpu_util_ratio = Gauge("gpu_util_ratio", "GPU utilization ratio [0,1].", registry=registry)
gpu_vram_bytes = Gauge("gpu_vram_bytes", "GPU VRAM bytes.", labelnames=("type",), registry=registry)
queue_depth = Gauge("queue_depth", "Pending jobs.", registry=registry)
model_swap_total = Counter("model_swap_total", "Image/video mode swaps.",
                           labelnames=("from", "to"), registry=registry)
jobs_total = Counter("jobs_total", "Finished jobs.",
                     labelnames=("preset", "status"), registry=registry)
callback_attempts_total = Counter("callback_attempts_total", "Callback POSTs.",
                                  labelnames=("status",), registry=registry)
worker_ready = Gauge("worker_ready", "1 after warm-up, 0 before.", registry=registry)
```

### Task P2-8: Wire metrics at emission sites

- `worker/pipeline/executor.py`: on job completion, `inference_duration_seconds.labels(preset=p).observe(duration_s)`; per stage,
  `stage_duration_seconds.labels(stage=s).observe(...)`. Increment
  `jobs_total.labels(preset, status).inc()`.
- `worker/core/callback.py`: `callback_attempts_total.labels(status=<"success"|"failed">).inc()`.
- `worker/pipeline/queue.py`: `queue_depth.set(q.depth())` on enqueue/dequeue.
- `worker/pipeline/model_manager.py`: on `ensure(new_mode)` where
  `old_mode != new_mode`, `model_swap_total.labels(from=old, to=new).inc()`.
- `worker/state.py`: `worker_state.mark_ready()` → `worker_ready.set(1)`.
- `worker/main.py`: background task samples GPU every 5 s via existing
  `worker.gpu_info.safe_gpu_info()` and updates `gpu_util_ratio` +
  `gpu_vram_bytes{type}`.

### Task P2-9: `/v1/metrics` endpoint + separate port

**Option A (single FastAPI):** mount at `/v1/metrics` on the main port.
Simpler; matches Phase 1 health endpoint exposure.

**Option B (separate port `$METRICS_PORT`):** start a second `uvicorn` or
a lightweight `prometheus_client.start_http_server(METRICS_PORT)` thread.
Allows scraping without exposing the generate API.

Go with **A** for Phase 2 (simpler, one process). If Cloudflare Tunnel
forwarding becomes a concern, gate by IP allow-list in middleware later.

```python
# worker/api/metrics.py
from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from worker.core.metrics import registry

router = APIRouter(tags=["metrics"])

@router.get("/v1/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
```

Register in `worker/main.py` `create_app()`.

**Tests:**
- `GET /v1/metrics` returns 200 + `text/plain; version=0.0.4`.
- After one job (mocked), `inference_duration_seconds_count{preset="edit"}` increments.
- After a failed callback, `callback_attempts_total{status="failed"}` increments.

### Task P2-10: Optional Sentry wiring

Gated by `SENTRY_DSN`. In `worker/main.py` `create_app()`:

```python
if s.sentry_dsn:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    sentry_sdk.init(
        dsn=s.sentry_dsn,
        traces_sample_rate=0.05,
        environment=s.worker_id,
        integrations=[FastApiIntegration()],
    )
```

Add `sentry-sdk>=2.17` to optional deps (new group `[sentry]`). Keep out of
base install to minimize image size when disabled.

Sprinkle `sentry_sdk.set_tag("job_id", job.job_id)` in `Executor.run_job`.

## Sprint 4 — Deployment, CI, Load testing

### Task P2-11: Cloudflare Tunnel integration

**Goal:** each worker instance reachable at a stable URL without exposing the
vast.ai instance port to the public internet.

Approach: `cloudflared` sidecar in the container (already reachable via
`$CLOUDFLARE_TUNNEL_TOKEN` env var). Add to `supervisord.conf`:

```ini
[program:cloudflared]
command=/usr/local/bin/cloudflared tunnel --no-autoupdate run --token %(ENV_CLOUDFLARE_TUNNEL_TOKEN)s
autostart=%(ENV_CLOUDFLARE_TUNNEL_ENABLED)s
autorestart=true
stdout_logfile=/var/log/cloudflared.out.log
stderr_logfile=/var/log/cloudflared.err.log
```

Install `cloudflared` in the Dockerfile:

```dockerfile
RUN curl -fsSL https://pkg.cloudflare.com/install.sh | sh \
 && apt-get install -y cloudflared
```

Control plane (operator) provisions a tunnel per worker (`cloudflared tunnel
create wk-<hostname>`) and sets `CLOUDFLARE_TUNNEL_TOKEN` in the instance's
`.env`. Tunnel routes `worker-<n>.example.com → localhost:$WORKER_PORT`.

**Acceptance:** With a test tunnel token, `curl https://worker-dev.example.com/v1/health`
returns the same JSON as `curl http://127.0.0.1:8000/v1/health`.

### Task P2-12: GHCR CI pipeline

**File:** `.github/workflows/build-and-push.yml`.

```yaml
name: build-and-push
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pyright worker
      - run: pytest -m "not gpu" -q

  image:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: |
            ghcr.io/coltrosetech/inference-worker:${{ github.sha }}
            ghcr.io/coltrosetech/inference-worker:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

**Not running GPU tests in CI:** GPU tests (marked `gpu`) stay local / on
self-hosted runners. Phase 2 does NOT provision those; a follow-up (Phase 3)
can.

**Acceptance:** Push to `main` → image appears at
`ghcr.io/coltrosetech/inference-worker:<sha>`. `docker pull` from a clean
host succeeds with a valid GITHUB_TOKEN or public read.

### Task P2-13: Load testing harness

**File:** `scripts/loadtest.py`.

Parameters:
- `--concurrency N` (default 4)
- `--total M` (default 100 jobs)
- `--preset edit|style|controlnet|inpaint|ltx_video`
- `--worker URL` (default `http://127.0.0.1:8000`)

Emits one request every `M / N` jobs per slot, reads callbacks via a minimal
HTTP server started inline. Outputs a histogram (p50/p90/p95/p99 + error rate)
to stdout + a JSON file.

**SLO targets (from spec §6.7) to verify:**
- `edit` preset: p50 ≤ 3s, p95 ≤ 6s (6 steps, 1024²).
- `style` preset: p50 ≤ 3.5s, p95 ≤ 7s.
- `controlnet`: p50 ≤ 4s, p95 ≤ 8s (includes preprocessor).
- `inpaint`: p50 ≤ 3s, p95 ≤ 6s.
- `ltx_video`: p50 ≤ 60s, p95 ≤ 120s (97 frames, 40 steps).

**Output:** `reports/loadtest_<preset>_<date>.json` checked into the repo
under a one-off tag `loadtest-phase2-<date>` so future regressions are
comparable.

**Out of scope here:** automated regression gating (compare p95 against last
run and fail CI if it regressed >20%). That lands in Phase 3.

### Task P2-14: Update CLAUDE.md + README

- CLAUDE.md: update "Current state" to 2026-0X-XX Phase 2 shipped; add new
  presets to "Architecture" section; clear "Open items" for items 1-4.
- README.md: add `controlnet` / `inpaint` / `ltx_video` to Presets list;
  show `/v1/metrics` usage; document Cloudflare Tunnel env var.

### Task P2-15: Phase 2 smoke test

Mirror Phase 1 Task 38. On a fresh vast.ai RTX 5080:

1. `docker compose up -d --build` (first-boot fetches 51 GB of models).
2. `GET /v1/health` → `ready: true` within 3 min.
3. One job through each preset via `POST /v1/generate`:
   - `edit`, `style` (regression — must still work)
   - `controlnet` (`controlnet_type=canny`)
   - `inpaint` (with a small mask)
   - `ltx_video` (25-frame quick mode)
4. Each job posts a valid HMAC callback and a correctly-sized output to the
   upload URL.
5. `GET /v1/metrics` shows non-zero `inference_duration_seconds_sum{preset=<p>}`
   for each preset exercised.
6. Kill the instance, re-create it, confirm `docker compose up -d` reaches
   `ready: true` from cached models in <90 s.

Document results in a CHANGELOG entry.

## Cross-cutting concerns

### VRAM budgeting on 16 GB

| Mode | Peak VRAM | Coexistence |
|---|---|---|
| edit / style (SDXL + IP-Adapter + CLIP-ViT-H) | ~11 GB | — |
| controlnet (+ CN Union) | ~13 GB | OK |
| inpaint (SDXL only, no IP-A) | ~9 GB | OK |
| ltx_video (LTX + T5-XXL fp8) | ~13 GB | Requires full image-mode unload |

`ModelManager.ensure(mode)` must issue `POST /free` to ComfyUI when
`current_mode != new_mode`, waiting for completion before submitting the
new prompt. Measure swap time in load test; budget ≤5 s for the swap
(influences `ltx_video` p50 if the queue alternates modes).

### API contract stability

No breaking changes to `POST /v1/generate`. The new presets fit the existing
schema (§3.1 of the spec already lists `preset` as a string enum with four
values plus `ltx_video`). `mask_image_url` becomes required when
`preset == "inpaint"` — document in the error payload, do NOT reject at the
generic parameter layer.

### Test posture

Phase 2 adds ~30–40 tests spread across:
- `tests/test_presets/test_{controlnet,inpaint,ltx_video}.py` (unit)
- `tests/test_pipeline/test_warmup.py` (stub mask, timeout override)
- `tests/test_metrics.py` (endpoint + registry emission)
- `tests/test_comfyui/test_client.py` (no changes; Phase 1 race fix covers)
- `tests/integration/` (two new smoke tests: one per new preset family
  behind the `gpu` marker)

Target: pytest green at **≥150 passed, 1 skipped** after Phase 2 ships.

## Done criteria

Phase 2 is shippable when:

1. All five presets serve real inference and post signed callbacks.
2. `/v1/health.ready == true` after startup on a fresh instance in ≤3 min.
3. `/v1/metrics` reports the nine-metric set with non-zero values after one
   job of each preset.
4. GitHub Actions publishes the image on merge to `main`.
5. Cloudflare Tunnel routes reach `/v1/health` from outside the vast.ai
   network.
6. Load-test report exists showing p50/p95 within SLO for each preset.
7. `pytest -m "not gpu"` green; smoke script `scripts/phase2_smoke.sh`
   returns 0 on a real GPU.

## Risk register

- **LTX-Video VRAM pressure on 16 GB:** if peak exceeds 14 GB in practice,
  fall back to fp8 quantized T5 + LTX low-VRAM mode or reduce default
  `width×height`. Decision point: first load test on a vast.ai RTX 5080.
- **DWPose dependency weight:** adds ~300 MB of extra pip deps (mmpose,
  mmcv). If the image grows beyond a comfortable size (>25 GB), make DWPose
  optional (install gated by `ENABLE_DWPOSE=1`) and default to OpenPose
  lightweight.
- **CN Union ProMax availability:** the model is community-hosted; pin the
  specific HF commit in `models.yaml` to avoid surprise changes.
- **GHCR image size:** fat image will be ~22–25 GB. First pull on fresh
  vast.ai instance takes 3–4 min at 1 Gbps. If this is unacceptable,
  Phase 3 considers the thin-image + runtime-fetch approach (rejected in
  Phase 1 §8 — revisit only with real data).
