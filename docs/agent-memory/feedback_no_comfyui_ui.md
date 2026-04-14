---
name: No ComfyUI UI usage — all CLI/terminal
description: User doesn't want to touch the ComfyUI web interface; everything must be driven from terminal/code
type: feedback
originSessionId: a861bbe1-6a43-4919-842f-1562402b59b3
---
Do not ask the user to open or interact with the ComfyUI web UI. All ComfyUI work (installing custom nodes, downloading models, authoring workflows, starting/stopping the service, testing) must be done from the terminal through shell commands, Python scripts, and workflow JSON files authored directly.

**Why:** User explicitly stated they don't want to do anything in the ComfyUI UI, they want me to handle everything from the terminal. This keeps the system fully CLI/code-driven and avoids UI-dependent workflows.

**How to apply:** When building ComfyUI-based systems, author workflow JSON directly (not via UI export), install custom nodes with git clone + pip install, download models via huggingface-cli / wget, manage ComfyUI as a systemd/supervisor service, and test via HTTP API calls (curl / Python httpx) — never "go to the UI and click X".
