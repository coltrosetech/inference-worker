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

## Current state (as of 2026-04-20)

- Branch: `feat/phase-1-foundation`. `git log --oneline` shows the full trail.
- Tests: **163 pass, 1 skipped** (`tests/integration/*` requires live worker
  + env vars).
- **Phase 1 + Phase 2 Sprint 2 + FLUX Kontext premium shipped.** Six presets
  warm via the lifespan and `/v1/health.ready` flips to `true` on RTX 5080
  (vast.ai). Full warm-up of all six took ~28 s in verification (cold cache).
  VRAM settles at ~9.5 GB post-warm; ComfyUI's own memory manager evicts old
  model weights on mode swap, and `ModelManager.ensure()` additionally POSTs
  `/free` as belt-and-suspenders.
- Phase 2 Sprints 3–4 (observability, Cloudflare Tunnel, GHCR CI, load
  testing) remain in the plan. A Phase 3 (FLUX Fill/Redux/ControlNet premium
  variants, `torch.compile`, SageAttention, TensorRT, batching) is
  unscheduled.

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
- `inpaint` — `VAEEncodeForInpaint + GrowMask + ImageCompositeMasked`;
  requires `mask_image_url` (InputPaths.mask_image validated at inject
  time).
- `ltx_video` — `Mode.VIDEO`, output `video/mp4`. Uses
  `CheckpointLoaderSimple` on the bundled LTX-Video 2B 0.9.8 distilled
  fp8 safetensors (placed in `checkpoints/` — contains UNET+VAE), plus
  `CLIPLoader(type="ltxv")` on T5-XXL fp8, then `LTXVImgToVideo +
  LTXVConditioning + LTXVScheduler + CFGGuider +
  SamplerCustomAdvanced + VAEDecode + VHS_VideoCombine`. Peak VRAM ~13
  GB; ModelManager triggers `ComfyUIClient.free()` on IMAGE↔VIDEO swap
  to fit the 16 GB budget.
- `edit_premium` — `Mode.IMAGE_PREMIUM`. **FLUX.1-Kontext-dev fp8 scaled**
  (Comfy-Org repackaged) for native prompt-driven semantic editing. Pipeline:
  `UNETLoader + DualCLIPLoader(type=flux, clip_l + t5xxl_fp8) +
  VAELoader(flux_ae) + FluxKontextImageScale + VAEEncode + ReferenceLatent
  + FluxGuidance(guidance=2.5) + KSampler(cfg=1.0, euler/simple, 20 steps)
  + VAEDecode + SaveImage`. Runs in ~10–30 s on RTX 5080 (16 GB). Much higher
  quality than `edit` (Lightning img2img) at the cost of latency. User chooses
  speed vs. quality via preset selection.

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

# Download models (~10 GB, ~4 min at 1 Gbps)
python scripts/download_models.py --models-dir /workspace/ComfyUI/models

# Restart ComfyUI so it loads new custom nodes
supervisorctl -c /etc/supervisor/supervisord.conf restart comfyui

# Start the worker
source /workspace/works/.venv/bin/activate
set -a && source .env && set +a
uvicorn worker.main:app --host 127.0.0.1 --port 8000 &
```

See `README.md` for the equivalent Docker path (not yet validated on a fresh
box; smoke test used the bare-metal path above).

## Repository layout

- `worker/` — FastAPI service (api/, core/, comfyui/, io/, pipeline/, presets/)
- `workflows/` — ComfyUI workflow JSON templates (API format for edit, legacy
  for style — see Open items #1)
- `configs/` — model + custom-node YAML manifests
- `scripts/` — bootstrap, model downloader, warm-up, mock backend, secrets gen
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
