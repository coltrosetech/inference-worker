# CLAUDE.md — Project context for Claude Code

**Read this first.** If you are Claude Code starting a new session on this repo, this
document tells you what exists, what the user prefers, and where to resume.

## Project

Stateless GPU inference worker for image/video generation. The user's remote
frontend sends `prompt + image`, the worker revises/regenerates, uploads the
result to the user's storage, and POSTs an HMAC-signed callback.

- Deployment target: vast.ai GPU instances (starting with RTX 5080, 16 GB).
- Horizontal scale: one worker per GPU box; control plane (queue / DB /
  storage / frontend / auth) lives on the user's own server (not in this repo).
- "Maximum optimization" is a stated requirement.
- Minimal-filter models are preferred; no safety classifier is installed.

## User preferences (carry forward — do not re-ask)

- **Never use the ComfyUI web UI.** Author workflows as JSON, install custom
  nodes via CLI, manage ComfyUI as a service, test via HTTP. The user
  explicitly rejected UI-driven workflows during brainstorming.
- Confirm-with-"Onaylıyorum go" style of working: present options with a
  recommendation, the user picks.
- User responds in Turkish; answer in Turkish. Communicate briefly, ship
  fast, don't over-explain.
- When touching git: never amend, never skip hooks, never modify git config.
  Use `GIT_AUTHOR_NAME="Claude Code" GIT_AUTHOR_EMAIL="noreply@anthropic.com"`
  env vars for commits (repo has no user.name/email set).

## Current state (as of 2026-04-20, evening)

- Branch: `feat/phase-1-foundation`. `git log --oneline` shows the full trail.
- Tests: **196 pass, 1 skipped** (`tests/integration/*` requires live worker
  + env vars).
- **Ten presets now registered.** Full warm-up of all ten takes ~30 s on
  RTX 5090 (vast.ai). `/v1/health.ready` flips to `true` once every preset
  warms successfully.
  - `edit` — SDXL Lightning img2img + IP-Adapter
  - `style` — IP-Adapter style transfer
  - `controlnet` — ControlNet Union SDXL ProMax (canny/depth/pose/lineart/scribble)
  - `inpaint` — SDXL Lightning inpaint with optional two-pass (undress→redress)
    + structural refiner. **Python-side SegFormer-B2 auto-mask** keeps face /
    hair / skin / background intact by construction.
  - `inpaint_sdxl` — JuggernautXL Inpaint v9, balanced SDXL fine-tune
  - `inpaint_realvis` — RealVisXL V4 Inpaint, photorealistic portrait aesthetic
  - `inpaint_premium` — **FLUX.1-Fill-dev fp8** with optional ControlNet pose
    guide (FLUX.1-dev ControlNet Union Pro 2.0, openpose via DWPreprocessor)
  - `tryon` — IP-Adapter (garment photo as `reference_image_url`) +
    JuggernautXL Inpaint for reference-driven virtual try-on
  - `edit_premium` — FLUX.1-Kontext-dev fp8 prompt-driven semantic edit
  - `ltx_video` — LTX-Video 2B img→video
- **Webapp + Cloudflare quick tunnel** live for browser testing (see "Webapp"
  section).
- Skipped for this slice, kept as a one-line enable: `--use-sage-attention`
  in `/opt/supervisor-scripts/comfyui.sh` for ~20-40 % FLUX speed. Package
  `sageattention` is already pip-installed in the ComfyUI venv.
- Phase 2 Sprints 3–4 (observability, GHCR CI, load testing) and Phase 3
  (torch.compile, TensorRT, batching) remain unscheduled. Named Cloudflare
  tunnel (stable URL) is a separate mini-sprint.

## Architecture

Two stable docs (read these if you need full detail, do NOT re-derive):

- `docs/superpowers/specs/2026-04-14-img-video-worker-design.md` — design
  spec. API contract, error matrix, SLO targets, deployment layout, file
  structure. Approved by user before plan writing.
- `docs/superpowers/plans/2026-04-14-img-video-worker-phase-1.md` — 38-task
  TDD implementation plan for Phase 1. All 37 code tasks committed; task 38
  (manual smoke test) was executed on the original vast.ai instance.

## Workflow format (API format only)

Both `workflows/edit.json` and `workflows/style.json` use ComfyUI **API
format** (`{node_name: {class_type, inputs}}`). The legacy full-format
(`{nodes: [...], links: [...]}`) branch in `EditPreset.inject()` /
`StylePreset.inject()` is kept as defensive fallback for future workflows
but is not exercised in production; `Executor._run_inference` skips
`convert_workflow()` when the template is already API format.

Registered presets (`worker/presets/__init__.py`):
- `edit` — img2img + IP-Adapter (input as ref), weight_type `linear`,
  `preservation` knob.
- `style` — img2img + IP-Adapter (separate reference_image), weight_type
  `style transfer`, `style_strength` knob.
- `controlnet` — ControlNet Union SDXL ProMax via `ControlNetLoader +
  SetUnionControlNetType + ControlNetApplyAdvanced`; `AIO_Preprocessor`
  for input; `controlnet_type` enum: canny | depth | pose | lineart |
  scribble.
- `inpaint` — `VAEEncodeForInpaint + GrowMask + ImageCompositeMasked` on
  SDXL Lightning. Optional params: `two_pass` (undress→redress, sidesteps
  the "majority completion" bias by filling skin first then redressing —
  uses a separate `skin_prompt` for pass 1), `structural_refiner` (unsharp
  post-process), `auto_mask` + `auto_mask_categories`. Either
  `mask_image_url` or `auto_mask=true` is required.
- `inpaint_sdxl` / `inpaint_realvis` — subclasses of InpaintPreset sharing
  all its plumbing; only the checkpoint (`juggernaut_xl_inpaint.safetensors`
  / `realvisxl_v40_inpaint.safetensors`) and sampler defaults (dpmpp_2m +
  karras, 25 / 30 steps, cfg 7 / 6.5) differ.
- `inpaint_premium` — `Mode.IMAGE_PREMIUM`. **FLUX.1-Fill-dev fp8**
  (non-gated `dim/...` mirror) + `InpaintModelConditioning` + `FluxGuidance`
  (default 30.0). Optional `use_pose_guide` inserts `DWPreprocessor →
  ControlNetLoader(flux_controlnet_union_pro_2_fp8.safetensors) →
  SetUnionControlNetType("openpose") → ControlNetApplySD3` (the SD3-family
  node, needed because FLUX ControlNet requires a VAE input which the
  legacy `ControlNetApplyAdvanced` node can't supply). When the toggle is
  off, `inject()` deletes all four pose nodes to avoid paying the
  preprocessor + ControlNet load cost. Single-shot premium quality —
  `two_pass` is intentionally not supported here.
- `tryon` — virtual try-on: needs both `reference_image_url` (the garment
  photo) and a mask (usually via `auto_mask`). Graph: JuggernautXL Inpaint
  + `IPAdapterUnifiedLoader` (PLUS preset) + `IPAdapterAdvanced`
  (weight_type `linear`, embeds_scaling "V only") + the usual inpaint
  stack. `reference_weight` param (default 0.9) trades prompt freedom
  against garment fidelity.
- `ltx_video` — `Mode.VIDEO`, output `video/mp4`. Uses
  `CheckpointLoaderSimple` on the bundled LTX-Video 2B 0.9.8 distilled
  fp8 safetensors (placed in `checkpoints/` — contains UNET+VAE), plus
  `CLIPLoader(type="ltxv")` on T5-XXL fp8, then `LTXVImgToVideo +
  LTXVConditioning + LTXVScheduler + CFGGuider +
  SamplerCustomAdvanced + VAEDecode + VHS_VideoCombine`. Peak VRAM ~13
  GB; ModelManager triggers `ComfyUIClient.free()` on IMAGE↔VIDEO swap
  to fit the 16 GB budget.
- `edit_premium` — `Mode.IMAGE_PREMIUM`. **FLUX.1-Kontext-dev fp8 scaled**
  (Comfy-Org repackaged) for native prompt-driven semantic editing.
  Pipeline: `UNETLoader + DualCLIPLoader(type=flux, clip_l + t5xxl_fp8) +
  VAELoader(flux_ae) + FluxKontextImageScale + VAEEncode + ReferenceLatent
  + FluxGuidance(guidance=2.5) + KSampler(cfg=1.0, euler/simple, 20 steps)
  + VAEDecode + SaveImage`. Runs in ~10–30 s.

## Auto-mask pipeline

`inpaint`, `inpaint_sdxl`, `inpaint_realvis`, `inpaint_premium`, and
`tryon` all share one auto-mask mechanism. When `parameters.auto_mask` is
true, `InpaintPreset.orchestrate()` (or the subclass / tryon override)
shells out to `scripts/segment_clothing.py` via
`worker/io/segmentation.py` → `asyncio.create_subprocess_exec` against
`/venv/main/bin/python`. That script loads `mattmdjaga/segformer_b2_clothes`
from `$MODELS_PATH/segformer_b2_clothes/` and writes a binary PNG mask
where **only the user-requested clothing category indices** are 255 —
background, face, hair, skin, and any class the user didn't ask for stay
0. Why Python-side instead of the `StartHua/Comfyui_segformer_b2_clothes`
custom node: that node's `sample()` hard-codes `labels_to_keep = [0]`
(background), so the mask it produces always marks the background as
inpaint, which wrecked every early auto-mask run. The custom node is no
longer installed; `configs/custom_nodes.yaml` still has a comment
explaining why. Model weights (`model.safetensors`, `config.json`,
`preprocessor_config.json`) **are** kept in `configs/models.yaml`
because the Python script loads them.

After segmentation, `orchestrate()` writes the mask into ComfyUI's input
dir as `{job_id}_automask.png`, flips `params.auto_mask` off, and hands
off to the normal single- or two-pass inpaint flow so both passes read
the same mask.

Warm-up (`worker/pipeline/warmup.py`) iterates `PRESETS` (or
`WARMUP_PRESETS` env, comma-list). Per-preset overrides live on the
`Preset` base class: `needs_reference_image`, `needs_mask_image`,
`warmup_params() -> dict`, `warmup_timeout_sec`. `ltx_video` uses small
frames (25 @ 512×320) for warm-up; its timeout is raised to 600 s.
`worker_state.mark_ready()` fires only after ALL warmups succeed.
Standalone invocation remains via `scripts/warm_comfyui.py`.

## Completed open items (as of 2026-04-20)

- ~~Migrate `style.json` to API format + IP-Adapter (item #1)~~ — done.
- ~~IP-Adapter chain in `edit.json` (item #2)~~ — done.
- ~~Phase 2 plan~~ — at `docs/superpowers/plans/2026-04-20-img-video-worker-phase-2.md`.
- ~~Auto-warmup → `worker_state.ready` (item #4)~~ — done.
- Fixed a ws race in `worker/comfyui/client.py` (`wait_for_completion` now
  polls `/history/{prompt_id}` each 2 s as a fallback; prevents missed
  completion events when inference is sub-second on cached models).

## Open items (Phase 2 remainder)

See `docs/superpowers/plans/2026-04-20-img-video-worker-phase-2.md`. Sprint 2
(new presets) shipped. Remaining sprints:

1. Sprint 3 — Full Prometheus metrics wiring + `/v1/metrics` endpoint +
   optional Sentry.
2. Sprint 4 — Cloudflare Tunnel + GHCR CI pipeline + load-testing harness
   + SLO verification.

## Resuming on a new vast.ai instance

Once `CLAUDE.md` + repo are on a fresh GPU box:

```bash
cd /workspace
git clone <REPO_URL> works
cd works
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
bash scripts/gen_secrets.sh .env     # or paste the saved values from the previous run
# point at the instance's ComfyUI
sed -i 's|^COMFYUI_PATH=.*|COMFYUI_PATH=/workspace/ComfyUI|' .env
sed -i 's|^COMFYUI_INTERNAL_PORT=.*|COMFYUI_INTERNAL_PORT=18188|' .env   # if vast.ai template uses 18188
sed -i "s|^MODELS_PATH=.*|MODELS_PATH=/workspace/ComfyUI/models|" .env
sed -i "s|^WORKFLOWS_PATH=.*|WORKFLOWS_PATH=/workspace/works/workflows|" .env

# Install missing custom nodes into the existing ComfyUI
source /venv/main/bin/activate   # vast.ai template's ComfyUI venv
COMFYUI_PATH=/workspace/ComfyUI bash scripts/install_custom_nodes.sh configs/custom_nodes.yaml

# Download models (~70 GB total across all ten presets, ~15–25 min at 1 Gbps)
python scripts/download_models.py --models-dir /workspace/ComfyUI/models

# Restart ComfyUI so it loads new custom nodes
supervisorctl -c /etc/supervisor/supervisord.conf restart comfyui

# Install the worker (with webapp extra for the playground UI)
cd /workspace/works
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,webapp]"

# Start the worker
set -a && source .env && set +a
uvicorn worker.main:app --host 127.0.0.1 --port 8000 &

# Build + start the webapp + Cloudflare tunnel (optional — for browser testing)
bash scripts/build_frontend.sh        # npm install + vite build
bash scripts/start_webapp.sh &        # FastAPI on :8001
bash scripts/start_tunnel.sh          # cloudflared — prints the *.trycloudflare.com URL
```

Optional: to enable SageAttention (~20–40 % FLUX-Fill speedup), append
`--use-sage-attention` to `COMFYUI_ARGS` in `/opt/supervisor-scripts/comfyui.sh`
and `supervisorctl restart comfyui`. Package is already pip-installed in
the ComfyUI venv.

See `README.md` for the equivalent Docker path (not yet validated on a fresh
box; smoke test used the bare-metal path above).

## Webapp (React + shadcn/ui playground)

A browser UI for testing the worker lives in `webapp/`. It is a separate
FastAPI app on port 8001 that proxies `/v1/generate` to the worker, stores
uploads/outputs locally, and receives HMAC-verified callbacks. The worker
itself stays stateless.

```bash
# One-time: build the React bundle
bash scripts/build_frontend.sh

# Runtime: worker must already be up on :8000
bash scripts/start_webapp.sh          # :8001, serves the UI

# Optional: expose via Cloudflare quick tunnel (ephemeral trycloudflare.com URL)
bash scripts/start_tunnel.sh
```

The frontend is `webapp/frontend/` (Vite + React + TS + Tailwind v3 +
shadcn/ui); `npm run dev` inside that directory gives HMR against the
running webapp (proxy config in `vite.config.ts`). Routes summary:

- `GET /` — the SPA
- `POST /api/upload` — multipart → `{name, bytes}`
- `POST /api/generate` — proxy to worker `/v1/generate` with local URLs
- `PUT /o/{name}` — worker uploads output here
- `POST /api/callback` — worker HMAC callback (verified)
- `GET /api/jobs/{id}` — poll job state
- `GET /u/{name}`, `GET /o/{name}` — serve upload / output

## Repository layout

- `worker/` — FastAPI service (api/, core/, comfyui/, io/, pipeline/, presets/)
- `webapp/` — standalone FastAPI UI (port 8001) + React/Vite/shadcn SPA
- `workflows/` — ComfyUI workflow JSON templates (API format for edit, legacy
  for style — see Open items #1)
- `configs/` — model + custom-node YAML manifests
- `scripts/` — bootstrap, model downloader, warm-up, mock backend, secrets gen,
  webapp/tunnel launchers
- `tests/` — unit + component + integration tiers (integration skipped by
  default, requires `WORKER_API_KEY` env)
- `docs/superpowers/{specs,plans}/` — design artifacts

## What NOT to re-litigate

The following decisions were made during brainstorming and confirmed by the
user. Do not re-ask or re-propose alternatives without an explicit prompt:

- ComfyUI backend (not raw Diffusers) for Phase 1; Diffusers rewrite for a
  single hot pipeline deferred to Phase 2.
- JuggernautXL Lightning (primary) + RealVisXL 5.0 (alternate) for image;
  LTX-Video 2B for video (Phase 2). SDXL fine-tunes over FLUX for filter
  permissiveness.
- Async callback + signed upload URL (not synchronous binary response).
- HMAC-SHA256 signed callbacks (not mTLS, not unsigned).
- Bearer token auth between user's backend and worker.
- Docker-first deployment + bare-metal fallback; GHCR registry for Phase 2.
- Fat Docker image (ComfyUI baked in) instead of thin image with runtime setup.
- Cloudflare Tunnel for stable external URL (Phase 2 wiring).
