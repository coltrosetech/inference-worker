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

## Current state (as of 2026-04-14)

- Branch: `feat/phase-1-foundation` — **40 commits** ahead of origin/main (if
  origin exists). `git log --oneline` shows the full trail.
- Tests: **103 pass, 1 skipped** (`tests/integration/*` requires live worker
  + env vars). `.env` file isolation handled by `_isolate_cwd` fixture in
  `tests/test_config.py`.
- **Phase 1 is shippable.** Smoke-tested on RTX 5080: a 4-step JuggernautXL
  Lightning img2img request through `POST /v1/generate` returns a real
  image in ~1.6 s end-to-end (download + preprocess + GPU + upload +
  HMAC callback). Tested with 10 diverse prompts — all succeeded at
  1.0–1.5 s each. Sample outputs in `/workspace/ComfyUI/output/sample_*.png`
  on the original instance.

## Architecture

Two stable docs (read these if you need full detail, do NOT re-derive):

- `docs/superpowers/specs/2026-04-14-img-video-worker-design.md` — design
  spec. API contract, error matrix, SLO targets, deployment layout, file
  structure. Approved by user before plan writing.
- `docs/superpowers/plans/2026-04-14-img-video-worker-phase-1.md` — 38-task
  TDD implementation plan for Phase 1. All 37 code tasks committed; task 38
  (manual smoke test) was executed on the original vast.ai instance.

## Known deviations from the spec/plan

During smoke test (task 38) we discovered the original plan assumed full-format
ComfyUI workflows with valid `nodes + links` topology. Empty `links: []` is
rejected by ComfyUI's `/workflow/convert`. Pivot: use API format directly.

Commit `97c182c` contains the fix:

- `workflows/edit.json` is now ComfyUI **API format** (`{node_name: {class_type,
  inputs}}`) instead of full-format (`{nodes: [...], links: [...]}`).
- `WorkflowTemplate.is_api_format()` + `set_input(node_name, key, value)` added.
- `EditPreset.inject()` branches on format; the old `set_widget` path is still
  there for legacy full-format workflows.
- `Executor._run_inference` skips `convert_workflow()` when the template is
  already API format.
- `workflows/style.json` is still legacy format — tests pass but real inference
  would fail until it is migrated. See "Open items" below.

The `edit` preset is currently a **simple img2img pipeline** (JuggernautXL
Lightning + CLIP text encoders + KSampler + VAE decode). It does NOT yet use
IP-Adapter, despite the spec calling for IP-Adapter-driven editing. This was
intentional for smoke-test throughput; upgrading to IP-Adapter is tracked
under "Open items".

## Open items (in order)

1. Migrate `workflows/style.json` to API format and include reference image +
   IP-Adapter style-transfer mode. Update `StylePreset.inject()` analogously
   to `EditPreset.inject()`.
2. Extend `workflows/edit.json` to include IP-Adapter chain (CLIPVision,
   IPAdapterModelLoader, IPAdapter apply) so `edit` matches the "preserve
   input features while editing" intent from the spec.
3. Write Phase 2 plan — covers `ltx_video` (image-to-video with LTX-Video 2B),
   `controlnet` and `inpaint` presets, Prometheus metrics at `/v1/metrics`,
   Cloudflare Tunnel integration, GHCR CI, load testing. Spec §13 lists the
   backlog.
4. Auto-warmup is not wired to `worker_state.ready`. The worker reports
   `ready=false` indefinitely unless `worker_state.mark_ready()` is called.
   The integration test fixture `wait_for_worker_ready` expects `ready=true`,
   so it skips by default. Fix: call warm-up from the lifespan startup and
   flip the flag when it finishes.

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
