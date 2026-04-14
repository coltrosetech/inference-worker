# Image/Video Generation Worker — Design Spec

**Date:** 2026-04-14
**Status:** Approved (brainstorming phase)
**Next step:** Create implementation plan via `writing-plans` skill

## 1. Goal

Build a stateless GPU inference worker that performs image-to-image and image-to-video generation on remote GPU nodes (vast.ai today, more later). A separate central backend (owned by the user) handles frontend, storage, auth, queue, and orchestration; it dispatches jobs to one or more workers via HTTP and receives results via signed upload + HMAC-signed callback.

The worker must be:
- **Portable:** clone repo, set `.env`, run one command on any Linux + NVIDIA GPU host → working service.
- **Maximally optimized:** target >92% GPU utilization under load, p50 ≤ 4 s for image edit, ≤ 25 s for 4 s video clip.
- **Permissive by default:** uses community-uncensored SDXL fine-tunes and LTX-Video; no safety classifier is installed or invoked.
- **Terminal-only operable:** no ComfyUI web UI interaction; all workflows authored as JSON, all setup scripted.

## 2. System context

```
┌──────────────────────────────────┐       ┌──────────────────────────────────────┐
│  USER'S CENTRAL BACKEND (stable) │       │  GPU WORKER NODE (ephemeral, vast.ai)│
│                                  │       │                                      │
│  • Frontend (elsewhere)          │       │  ┌────────────────────────────────┐  │
│  • Orchestration API             │       │  │ inference-worker (FastAPI)     │  │
│  • Job queue (Redis)             │       │  │  POST /v1/generate             │  │
│  • Auth, rate limit              │──────▶│  │  GET  /v1/health               │  │
│  • Object storage (signed URLs)  │ HTTPS │  │  GET  /v1/metrics              │  │
│  • Postgres (jobs, users)        │       │  │  POST /v1/cancel               │  │
│                                  │       │  │                                │  │
│                                  │◀──────│  │  ↓ asyncio scheduler           │  │
│  callback + PUT to signed URL    │       │  │  ↓ model manager (swap+cache)  │  │
│                                  │       │  │  ↓ ComfyUI bridge              │  │
│                                  │       │  │                                │  │
│                                  │       │  │  └── ComfyUI (127.0.0.1:8188)  │  │
│                                  │       │  │      (headless, no UI)         │  │
│                                  │       │  └────────────────────────────────┘  │
│                                  │       │  Cloudflare Tunnel (stable URL)      │
└──────────────────────────────────┘       └──────────────────────────────────────┘
```

**Separation invariants:**
- Worker is stateless: no DB, no shared storage, no persistent job state. It holds at most `MAX_QUEUE_DEPTH` in-flight jobs in memory.
- Worker has zero access to the user's storage other than the per-job signed PUT URL it receives in the request.
- ComfyUI listens on `127.0.0.1:8188` only; nothing but the worker process on the same host reaches it.
- Worker authenticates inbound with `Authorization: Bearer <WORKER_API_KEY>`; outbound callbacks are HMAC-SHA256 signed with a separate `CALLBACK_HMAC_SECRET`.

## 3. API contract

### 3.1 `POST /v1/generate`

Headers:
- `Authorization: Bearer <WORKER_API_KEY>` (required)
- `Content-Type: application/json`
- `Idempotency-Key: <uuid>` (required; same key → 409 if job already seen)

Body:
```jsonc
{
  "job_id": "01JQZ...",                     // ULID/UUID from backend
  "preset": "edit" | "style" | "controlnet" | "inpaint" | "ltx_video",
  "prompt": "...",
  "negative_prompt": "",
  "input_image_url": "https://storage.example.com/in/xyz.png",
  "mask_image_url":   null,                 // inpaint only
  "reference_image_url": null,              // style only (IP-Adapter style ref)
  "parameters": {
    "steps": 8, "cfg": 1.8, "seed": null,
    "width": 1024, "height": 1024,
    "strength": 0.7,
    "controlnet_type": "canny",             // controlnet preset only
    "controlnet_strength": 0.75,
    "grow_mask_px": 8,                      // inpaint only
    "num_frames": 97,                       // ltx_video only
    "fps": 24
  },
  "callback_url": "https://backend.example.com/internal/jobs/01JQZ.../complete",
  "upload_url":   "https://storage.example.com/out/01JQZ.../result?signature=...",
  "upload_method": "PUT",
  "timeout_sec": 300
}
```

**Mode derivation:** `mode` is not sent in the request; it is inferred from `preset` (`ltx_video` → `video`, everything else → `image`). The worker uses `mode` internally to decide model-manager swap policy.

Responses:
- `202 Accepted` → `{ "accepted": true, "job_id": "...", "queue_position": 3, "worker_id": "wk-..." }`
- `400` invalid payload (pydantic validation error in detail)
- `401` bad token
- `409` duplicate `Idempotency-Key` or `job_id`
- `429` queue full (`Retry-After` header)
- `503` warming / model loading (`Retry-After` header)

### 3.2 Callback (worker → backend)

Worker first `PUT`s binary to `upload_url` (only on `status=success`), then `POST`s to `callback_url`:

```jsonc
{
  "job_id": "01JQZ...",
  "worker_id": "wk-5080-tr1",
  "status": "success" | "failed" | "cancelled",
  // Present only when status == "success":
  "output_url": "https://storage.example.com/out/01JQZ.../result",
  "output_kind": "image/png" | "video/mp4",
  "output_bytes": 842117,
  "duration_ms": 12345,
  "stages_ms": { "download": 120, "preprocess": 40, "inference": 11800, "upload": 380 },
  // Present only when status == "failed":
  "error": { "code": "OOM_CUDA", "message": "...", "retryable": true },
  "metadata": { "seed": 123456, "model": "juggernautXL_v9_lightning", "steps": 8, "preset": "edit" }
}
```

On `status=failed`, `output_url`/`output_kind`/`output_bytes` are omitted; `stages_ms` contains whichever stages completed before failure. On `status=cancelled`, `error` is omitted; `stages_ms` records elapsed time up to cancellation.

Security:
- Header `X-Worker-Signature: t=<unix-ts>, v1=<hex>` where `v1 = HMAC_SHA256(CALLBACK_HMAC_SECRET, t + "." + body)`
- Backend rejects callbacks with `|now - t| > 300 s` or bad HMAC
- Backend is expected to respond `2xx` to acknowledge; otherwise worker retries (see §10)

### 3.3 `GET /v1/health`

```json
{
  "ok": true,
  "ready": true,
  "gpu": "NVIDIA GeForce RTX 5080",
  "vram_used_gb": 11.4,
  "vram_total_gb": 16.0,
  "queue_depth": 2,
  "current_job": "01JQZ...",
  "model_loaded": "sdxl",
  "uptime_sec": 3412,
  "comfyui_alive": true,
  "version": "1.2.3"
}
```

`ready` stays `false` until warm-up (§6.5) completes. Load balancers and the backend should only route traffic when `ready=true`.

### 3.4 `GET /v1/metrics`

Prometheus text exposition format on `METRICS_PORT` (default `9090`, separate from main port). Metrics enumerated in §7.2.

### 3.5 `POST /v1/cancel`

```json
{ "job_id": "01JQZ..." }
```

If queued → remove from queue, emit `cancelled` callback. If running → call ComfyUI `/interrupt`, emit `cancelled` callback. Returns `200 OK` in both cases; `404` if job unknown.

## 4. Inference engine

**Phase 1 (this spec):** ComfyUI as inference backend, driven over HTTP + WebSocket. Workflows authored as full workflow JSON; converted to API format at runtime via the existing `comfyui-workflow-to-api-converter-endpoint` node.

**Phase 2 (deferred):** For a single proven-hot pipeline, port to raw Diffusers + `torch.compile` + SFast to claw back an additional 30–50 %. Phase 2 is a follow-up spec, not part of this one.

**ComfyUI launch:**
```
python main.py \
  --listen 127.0.0.1 --port 8188 \
  --disable-auto-launch \
  --disable-metadata \
  --normalvram
```

**Communication:**
- `POST /prompt` — submit workflow (API format)
- `POST /workflow/convert` — convert full workflow → API format (from the installed custom node)
- `WebSocket /ws?clientId=...` — progress events, completion (`executing: null, prompt_id: X`)
- `GET /history/{prompt_id}` — fallback if websocket drops
- `POST /interrupt` — cancel running
- `GET /system_stats` — VRAM, queue info

**File path conventions inside ComfyUI dir:**
- `ComfyUI/input/` — worker writes fetched source image as `<job_id>_input.png` (and `_mask.png`, `_ref.png` as needed), deletes after job
- `ComfyUI/output/` — ComfyUI writes result as `<job_id>.<ext>`, worker reads and deletes after upload

## 5. Workflow presets

Each preset is a full-format ComfyUI workflow JSON in `workflows/<preset>.json`. The worker loads the template, injects runtime values (prompt, seed, image paths, parameter widgets), calls `/workflow/convert`, then `/prompt`.

### 5.1 `edit` — general edit (IP-Adapter + prompt)
Graph: `LoadImage(input) → IP-Adapter Apply (ref=input, w=0.7) → PromptEncode → KSampler(LCM, 6 steps, cfg 1.8) → VAEDecode → SaveImage`
Params: `prompt`, `negative_prompt`, `strength` (0.5–0.9), `steps` (4–12), `cfg` (1.5–3.0), `seed`, `width`, `height`.

### 5.2 `style` — style transfer
Graph: `LoadImage(input) → Canny(weak) → ControlNet Apply (w=0.4)` and `LoadImage(ref) → IP-Adapter Plus (w=0.7)` merged → `PromptEncode → KSampler → VAEDecode → SaveImage`
Params: `prompt`, `style_strength` (IP-Adapter weight), `structure_strength` (ControlNet weight), steps/cfg/seed.

### 5.3 `controlnet` — structure-preserving redraw
Graph: `LoadImage(input) → Preprocessor(canny|depth|pose|lineart|scribble) → ControlNet Union Apply → PromptEncode → KSampler → VAEDecode → SaveImage`
Params: `controlnet_type` enum, `controlnet_strength` (0.5–1.0), `prompt`, steps/cfg/seed.

### 5.4 `inpaint` — region edit with mask
Graph: `LoadImage(input) + LoadImage(mask) → GrowMask(grow_mask_px) → VAEEncodeForInpaint → PromptEncode → KSampler → VAEDecode → ImageCompositeMasked → SaveImage`
Params: `prompt`, `mask_image_url` (required), `grow_mask_px` (default 8), steps/cfg/seed.

### 5.5 `ltx_video` — image-to-video
Graph: `LoadImage(input) → LTX VAE Encode → LTX T5 PromptEncode → LTX Sampler (num_frames, guidance, stg_rescale) → LTX VAE Decode → SaveVideoMP4`
Params: `prompt`, `num_frames` ∈ {25, 49, 97, 121} (default 97 ≈ 4 s @ 24 fps), `fps` (default 24), `guidance_scale` (default 3.0), `seed`, `width` (default 768), `height` (default 512); max resolution 1216×704.

**Common constraints:**
- Input image resized so max side ≤ 2048 (Pillow `thumbnail`) before write into ComfyUI `input/`
- Seed: if `null`, generate random 64-bit int; always returned in callback metadata
- If `steps < 15` and preset supports LoRA chaining, LCM/Lightning LoRA is auto-applied (controlled by preset template)
- Preset validation: pydantic schema per preset; extraneous or malformed params → 400 before the job touches the GPU

## 6. Optimization layers

### 6.1 Model
- JuggernautXL **Lightning** variant (6-step native) as primary SDXL checkpoint; RealVisXL 5.0 as alternate.
- LCM LoRA available to chain onto non-Lightning checkpoints if introduced later.
- LTX-Video + **fp8 T5 text encoder** (cuts ~4.5 GB VRAM versus fp16 T5).
- SDXL VAE fp16-fix variant (avoids the known fp16 overflow in the stock VAE).

### 6.2 CUDA / attention
- PyTorch 2.10 SDPA (FlashAttention-2 path enabled by default on Ampere+; RTX 5080 is Blackwell, supported).
- Env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `torch.backends.cuda.matmul.allow_tf32 = True`.
- Channels-last memory format for SDXL UNet.

### 6.3 Pipeline scheduling (async overlap)
Three pools inside the worker:
- IO pool (`asyncio`, up to 8 concurrent) — image download + upload
- CPU pool (thread pool, 2 workers) — preprocessing (resize, canny, depth map)
- GPU pool (asyncio semaphore = 1) — ComfyUI inference

Target: while one job runs on the GPU, the next job is downloaded and preprocessed, and the previous result is uploading. GPU is kept >92 % utilized under sustained load.

### 6.4 Model lifecycle (swap & warm cache)
On 16 GB VRAM we cannot keep SDXL (~11 GB) and LTX-Video (~10 GB) resident simultaneously. Strategy:
- After first load of each model family, keep the inactive family resident in system RAM (`model.to("cpu")`); do not release weights to disk.
- Transfer buffers are allocated in pinned host memory so GPU↔CPU copies stream over full PCIe bandwidth.
- On mode switch: `current.to("cpu", non_blocking=True)` → `other.to("cuda", non_blocking=True)`. Target swap time ≤ 4 s.
- **Mode affinity batching:** the scheduler drains all same-mode jobs in the queue before switching (prevents thrash).
- Backend-side hint: router prefers workers already in the target mode.

### 6.5 Start-up warm-up
`scripts/warm_comfyui.py` runs after ComfyUI is reachable:
- One dummy `edit` inference (64×64 stub image, 4 steps) to compile SDXL kernels.
- One dummy `ltx_video` inference (25 frames, lowest resolution) to compile LTX kernels and cache T5 encoder.
- `/v1/health` flips `ready=true` only after warm-up completes.

### 6.6 Batching (Phase 2, stub in this design)
Architecture leaves a seam for batched text-encode + batched KSampler when the queue holds 2+ same-preset, same-resolution jobs. Not enabled in Phase 1; revisit after production load data.

### 6.7 Target SLOs

| Scenario | p50 | p95 |
|---|---|---|
| `edit` / `style` (SDXL Lightning, 6 steps, 1024×1024) | 4 s | 7 s |
| `controlnet` (with preprocessing) | 6 s | 9 s |
| `inpaint` | 5 s | 8 s |
| `ltx_video` (97 frames, 768×512) | 25 s | 35 s |
| GPU utilization under sustained queue | > 92 % | — |
| Model swap (image ↔ video) | < 4 s | — |
| Cold first-job after warm-up | < 5 s overhead | — |

## 7. Observability

### 7.1 Logs
- `structlog` JSON to stdout, one line per important event (`job.accepted`, `job.completed`, `job.failed`, `model.swap`, `warmup.done`, `comfyui.restart`).
- `job.completed` contains all stage timings and final metadata; one line is sufficient to derive the full histogram.
- Docker: `docker logs`; host: journald. Optional Loki shipping is out of scope here.

### 7.2 Prometheus metrics (`:METRICS_PORT/metrics`)
- `inference_duration_seconds{preset}` histogram (end-to-end, including IO)
- `stage_duration_seconds{stage}` histogram (`download|preprocess|inference|upload|callback`)
- `gpu_util_ratio` gauge (sampled from NVML every 5 s)
- `gpu_vram_bytes{type="used|total"}` gauge
- `queue_depth` gauge
- `model_swap_total{from,to}` counter
- `jobs_total{preset,status}` counter
- `callback_attempts_total{status}` counter
- `worker_ready` gauge (0 during warm-up, 1 after)

### 7.3 Sentry
Optional; `SENTRY_DSN` env var. Captures unhandled exceptions and ComfyUI protocol errors with `job_id`, `preset`, `worker_id` tags. Disabled if DSN is empty.

## 8. Models & custom nodes

### 8.1 Model manifest (`configs/models.yaml`)
Downloaded via `scripts/download_models.py` (idempotent, SHA-256 verified, resumable via HF `huggingface_hub`).

| Name | Size | Destination | Purpose |
|---|---|---|---|
| JuggernautXL v9 Lightning | 6.5 GB | `models/checkpoints/` | SDXL base (primary) |
| RealVisXL 5.0 | 6.5 GB | `models/checkpoints/` | SDXL base (alternate) |
| SDXL VAE fp16-fix | 335 MB | `models/vae/` | VAE (explicit) |
| SDXL LCM LoRA | 400 MB | `models/loras/` | Few-step distillation |
| IP-Adapter Plus SDXL | 700 MB | `models/ipadapter/` | General IP-Adapter |
| IP-Adapter Plus FaceID SDXL | 700 MB | `models/ipadapter/` | Face-preserving IP-Adapter |
| CLIP-ViT-H-14 (IP-Adapter encoder) | 1.2 GB | `models/clip_vision/` | IP-Adapter image encoder |
| ControlNet Union SDXL ProMax | 2.5 GB | `models/controlnet/` | Canny/depth/pose/lineart (single model) |
| Depth-Anything-V2 | 400 MB | `models/depth/` | Depth preprocessor |
| DWPose / OpenPose | 200 MB | `models/openpose/` | Pose preprocessor |
| LTX-Video 0.9.5 | 9 GB | `models/diffusion_models/` | Video model |
| LTX-Video VAE | 800 MB | `models/vae/` | LTX VAE |
| T5-XXL fp8 encoder | 4.5 GB | `models/text_encoders/` | LTX text encoder |

Total ≈ 33 GB. At the measured 143 MB/s on this node, first-boot download is ~4 minutes; subsequent boots find everything cached.

### 8.2 Custom nodes (`configs/custom_nodes.yaml`)
Installed via `scripts/install_custom_nodes.sh` (git clone at pinned SHA, `pip install -r requirements.txt`):

- `ComfyUI_IPAdapter_plus`
- `comfyui-tooling-nodes`
- `ComfyUI-Advanced-ControlNet`
- `comfyui_controlnet_aux`
- `ComfyUI-LTXVideo`
- `ComfyUI-VideoHelperSuite`
- `ComfyUI-KJNodes`
- `ComfyUI-Differential-Diffusion`

Already present and reused: `ComfyUI-Manager`, `comfyui-workflow-to-api-converter-endpoint`.

## 9. Deployment & portability

### 9.1 Portability contract
On any Linux host with NVIDIA GPU + CUDA 12.8 driver (nvidia-container-toolkit for the Docker path):

```bash
git clone <REPO_URL> /workspace/works
cd /workspace/works
cp .env.example .env   # fill in secrets
docker compose up -d
```

or, without Docker:

```bash
bash scripts/bootstrap.sh && bash scripts/start.sh
```

Either path ends with a worker listening on `$WORKER_PORT`, reachable via Cloudflare Tunnel if `CLOUDFLARE_TUNNEL_TOKEN` is set.

### 9.2 Docker image
- Base: `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` (or equivalent).
- "Fat image" strategy: ComfyUI, all custom nodes, Python deps, and application code are baked in. Only model weights live on a mounted volume.
- Pushed to GHCR on `main` by CI; pulled on each vast.ai instance.

### 9.3 Volumes
```yaml
volumes:
  - ./data/models:/data/models        # model weights cache
  - ./data/cache:/data/cache          # HF hub + torch compile cache
  - ./data/comfyui:/data/comfyui      # ComfyUI runtime dir if not using baked-in copy
  - ./data/logs:/data/logs
```

### 9.4 Environment (`.env.example`)
```
WORKER_ID=wk-auto
WORKER_API_KEY=
CALLBACK_HMAC_SECRET=

COMFYUI_PATH=/data/comfyui
MODELS_PATH=/data/models
CACHE_PATH=/data/cache
WORKFLOWS_PATH=/app/workflows

WORKER_PORT=8000
COMFYUI_INTERNAL_PORT=8188
MAX_QUEUE_DEPTH=8
JOB_TIMEOUT_SEC_DEFAULT=300
GPU_DEVICE=0

HF_TOKEN=
CLOUDFLARE_TUNNEL_TOKEN=
ALLOWED_BACKEND_IPS=

LOG_LEVEL=info
LOG_FORMAT=json
METRICS_PORT=9090
SENTRY_DSN=
```

If `WORKER_API_KEY` or `CALLBACK_HMAC_SECRET` are empty at bootstrap, the script generates random 32-byte values and writes them back to `.env` so later boots stay consistent.

### 9.5 On this machine specifically
`/workspace/ComfyUI` already exists and is functional. For this host, `COMFYUI_PATH=/workspace/ComfyUI`; bootstrap installs only missing custom nodes and missing models, never touches the UI. On any other host, bootstrap installs ComfyUI fresh at the configured path.

### 9.6 Directory layout

```
works/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
├── scripts/
│   ├── bootstrap.sh
│   ├── start.sh
│   ├── download_models.py
│   ├── install_custom_nodes.sh
│   ├── warm_comfyui.py
│   └── cloudflared_setup.sh
├── worker/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── generate.py
│   │   ├── health.py
│   │   ├── metrics.py
│   │   └── cancel.py
│   ├── core/
│   │   ├── config.py
│   │   ├── auth.py
│   │   ├── logging.py
│   │   └── errors.py
│   ├── pipeline/
│   │   ├── queue.py
│   │   ├── scheduler.py
│   │   ├── model_manager.py
│   │   └── executor.py
│   ├── presets/
│   │   ├── base.py
│   │   ├── edit.py
│   │   ├── style.py
│   │   ├── controlnet.py
│   │   ├── inpaint.py
│   │   └── ltx_video.py
│   ├── comfyui/
│   │   ├── client.py
│   │   ├── workflow.py
│   │   └── supervisor.py
│   └── io/
│       ├── downloader.py
│       └── uploader.py
├── workflows/
│   ├── edit.json
│   ├── style.json
│   ├── controlnet.json
│   ├── inpaint.json
│   └── ltx_video.json
├── configs/
│   ├── models.yaml
│   ├── custom_nodes.yaml
│   └── comfyui_args.yaml
├── tests/
│   ├── test_api.py
│   ├── test_workflow_injection.py
│   ├── test_model_manager.py
│   ├── test_auth_hmac.py
│   └── fixtures/
└── docs/
    └── superpowers/specs/
        └── 2026-04-14-img-video-worker-design.md
```

### 9.7 Lifecycle

```
boot → bootstrap.sh (idempotent):
  1. load .env; generate missing secrets
  2. ensure COMFYUI_PATH exists (clone if not)
  3. install missing custom nodes (diff vs configs/custom_nodes.yaml)
  4. download missing models (diff vs configs/models.yaml)
  5. optional cloudflared if token present
  6. hand off to start.sh

start.sh → supervisord (PID 1 in container; a plain systemd unit in bare-metal mode):
  - program comfyui: main.py --listen 127.0.0.1 --port 8188 ...  (autostart, priority 100)
  - program worker:  uvicorn worker.main:app --host 0.0.0.0 --port $WORKER_PORT  (autostart, priority 200)
    (both start immediately; the worker itself waits on ComfyUI readiness before flipping ready=true)

worker startup sequence:
  a. poll ComfyUI /system_stats every 1 s, max 120 s; fail loudly if unreachable
  b. load & validate preset JSONs
  c. run warm_comfyui.py (dummy image + dummy video)
  d. /v1/health → ready=true

graceful shutdown (SIGTERM):
  a. stop accepting new jobs (return 503 on /v1/generate)
  b. wait up to 60 s for in-flight
  c. emit cancelled callbacks for unstarted jobs
  d. stop ComfyUI; exit
```

Crash recovery:
- Worker crash → supervisord restarts; running job becomes orphan; backend's timeout fires → backend reassigns.
- ComfyUI crash → supervisord restarts it; worker polls `/system_stats`; resumes accepting jobs when healthy.

## 10. Error handling matrix

| Condition | Worker action | `retryable` | Log level |
|---|---|---|---|
| `input_image_url` 404 / 403 | failed, `INPUT_FETCH_FAILED` | false | warn |
| `input_image_url` timeout | 3 retries (1/2/4 s backoff), then failed | true | warn |
| Input decode fails (not an image) | failed, `INPUT_DECODE_FAILED` | false | warn |
| Input max side > 4096 (hard cap) | failed, `INPUT_TOO_LARGE` | false | info |
| ComfyUI `/prompt` 500 | 1 retry, then failed | true | error |
| ComfyUI execution error (node) | failed, `INFERENCE_FAILED`, parsed error detail | true | error |
| CUDA OOM | failed, `OOM_CUDA`; model_manager forces full reload | true | error |
| Upload URL 403 (signature expired) | failed, `UPLOAD_AUTH_FAILED` | false | warn |
| Upload 5xx | 3 retries (1/2/4 s), then failed | true | warn |
| Callback 5xx | 5 retries (1/2/4/8/16 s), then drop | — | error |
| Callback 4xx | 1 retry, then drop | — | error |
| Job timeout (soft, default 300 s) | `/interrupt` ComfyUI, failed, `JOB_TIMEOUT` | true | warn |
| Worker SIGTERM during job | cancelled callback | true | info |

Worker never maintains its own retry queue. `retryable=true` only signals the backend; the backend decides whether to reassign.

## 11. Testing strategy

Three tiers:

**Unit (no GPU, every PR in CI):** ~40 tests covering workflow injection edge cases, auth & HMAC, preset validation, model-manager state transitions, async scheduler overlap (mocked GPU).

**Component (no GPU, every PR):** ~15 tests using FastAPI `TestClient` with mocked ComfyUI; full request → callback flow, retry behavior, idempotency, cancel.

**Integration (GPU required, nightly + manual):** 4–5 tests on a real GPU runner (manual initially; self-hosted GH Actions runner later):
- `edit` end-to-end with SSIM/CLIP checks on known input
- `ltx_video` end-to-end with duration & file-size checks
- Model swap timing (3-cycle image↔video)
- 50-job concurrent load → p95 latency + GPU utilization

Fixtures: 5–6 public-domain test images in `tests/fixtures/`.

## 12. Rollout plan (high-level; refined in implementation plan)

| Sprint | Focus | Est. |
|---|---|---|
| 0 — Foundation | Repo skeleton, Dockerfile, config/logging/auth/HMAC, bootstrap.sh, ComfyUI supervisor, `/v1/health` | 1–2 d |
| 1 — Image pipeline | `edit` + `style` presets, ComfyUI client, async scheduler, `/v1/generate` end-to-end, callback | 2–3 d |
| 2 — Video pipeline | LTX-Video node + model, `ltx_video` preset, model manager swap | 1–2 d |
| 3 — Remaining presets | `controlnet` + `inpaint`, preprocessors | 1–2 d |
| 4 — Hardening | Prometheus + Grafana dashboard, load test, SLO verification, CF Tunnel, CI → GHCR | 1–2 d |

Total ≈ 8–12 days single-developer equivalent. User's central backend may not be ready in parallel; a `scripts/mock_backend.py` is included so the worker can be validated end-to-end without it.

## 13. Open for Phase 2 (explicitly out of scope here)

- Premium "quality" mode via FLUX.1-Kontext (higher VRAM, more filtering).
- Raw-Diffusers + `torch.compile` port of the hottest preset.
- True batching (shared text encode + batched sampler) for same-preset same-resolution queues.
- Additional presets: `upscale`, `background_remove`, `face_restore`.
- Separate image-only and video-only worker classes (no swap overhead at all).
- Self-hosted GitHub Actions GPU runner.
