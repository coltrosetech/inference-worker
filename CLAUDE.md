# CLAUDE.md — Project context for Claude Code

**Read this first.** If you are Claude Code starting a new session on this repo, this
document tells you what exists, what the user prefers, and where to resume.

## Project

Stateless GPU inference worker for image/video generation. The user's remote
frontend sends `prompt + image`, the worker revises/regenerates, uploads the
result to the user's storage, and POSTs an HMAC-signed callback.

- Deployment target: vast.ai GPU instances (starting with RTX 5080, 16 GB).
- Horizontal scale: one worker per GPU box; control plane lives on the user's own server.
- "Maximum optimization" is a stated requirement.
- Minimal-filter models are preferred.

## User preferences (carry forward — do not re-ask)

- Never use the ComfyUI web UI. Author workflows as JSON, install custom nodes via CLI.
- Confirm with "Onaylıyorum go".
- User responds in Turkish; answer in Turkish. Communicate briefly.
- When touching git: never amend, never skip hooks. Use GIT_AUTHOR_NAME="Claude Code".

## Current state (as of 2026-05-23)

- Branch: `feat/phase-1-foundation`
- **Quality pack (RealVis)** shipped with advanced post-processing.
- Ten presets registered:
  - `edit`, `style`, `controlnet`
  - `inpaint`, `inpaint_sdxl`, `inpaint_realvis` (quality pack), `inpaint_premium` (FLUX.1-Fill-dev fp8)
  - `tryon`, `edit_premium` (FLUX.1-Kontext), `ltx_video`
- Auto-mask system with SegFormer-B2 is active.
- Hires fix, detailers, and preserve_composite techniques are implemented.

## Architecture

ComfyUI based inference worker with API format workflows. Presets are defined in `worker/presets/`.

Focus areas going forward:
- Further optimization
- New preset development
- Quality improvements
- Video generation enhancements