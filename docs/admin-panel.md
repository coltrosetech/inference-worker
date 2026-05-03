# Admin panel — `atelier · inference console`

Single-user admin panel for the worker. Lives in `webapp/frontend/` and is
served by the FastAPI webapp at `:8001`. Built with React 18 + TypeScript +
Vite + Tailwind v3 + shadcn/ui primitives + Radix UI.

## Aesthetic

A dark "atelier console" — pure black background with a single sodium-amber
accent (`#f5b942`). Geist Sans body + Geist Mono numerical readouts +
Instrument Serif italic display moments for a touch of editorial character.
Subtle radial gradient ornament in the background, page-load stagger reveal
for the four columns.

The visual restraint matches the use case (a private, technical tool) while
the typography pairing avoids the generic dashboard feel of pure Inter.

## Layout

Four columns under a sticky `SystemBar`:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ SystemBar       worker · ready  queue 0  RTX 5090 · 7.7/32GB  uptime …  │
├─────────┬────────────────────────────────────┬─────────────┬────────────┤
│ Preset  │ Inputs row (drop zones, bucket)    │  Output     │  Job queue │
│ Rail    │ ─────────────────────────────────  │  viewer     │  (live)    │
│         │ Composer (prompt + neg + params)   │             │            │
│  edit   │ ─────────────────────────────────  │  preview +  │  status    │
│  inp.   │ Action bar (seed lock + generate)  │  compare    │  dots,     │
│  realvis│                                    │  + actions  │  retry     │
│  …      │                                    │             │            │
└─────────┴────────────────────────────────────┴─────────────┴────────────┘
```

- **PresetRail** (`260px`) — vertical preset list, grouped by image/video,
  premium label on FLUX/LTX. Selected preset gets an amber left dot + glow.
- **Center canvas** (flex) — three sections:
  1. **Inputs** row: 1-3 drop zones (input, optional mask, optional reference).
     Each shows the original dimensions and the bucket destination
     ("768×1024 → 896×1152") inline.
  2. **Composer** body: prompt + negative_prompt textareas (token counters)
     and a preset-aware grouped accordion of parameters.
  3. **Action bar** (sticky bottom): seed control on the left, generate
     button on the right.
- **OutputViewer** (`360px`) — live job follow with status pill, click-to-
  open lightbox, copy-link / download / open / retry-with-new-seed buttons.
  Mini input thumbnail strip at the bottom.
- **JobQueue** (`360px`) — last 20 jobs with 2-second polling. Status dots
  pulse while running. Hover reveals a retry-with-new-seed shortcut.
- **CompareLightbox** — full-screen overlay opened by clicking the output
  preview. Drag-handle slider to reveal input under output, ←/→ keys move
  the split, ESC closes.

## Parameter accordions

Parameters are preset-aware and grouped. For `inpaint_realvis` the panel shows:

- **sampling** (steps, cfg, strength)
- **mask** (auto-mask toggle + category chips, grow_mask_px, feather_mask_px)
- **lora stack** (detail, skin, anatomy, **bust_emphasis**)
- **detailers · hires** (person/face/hand_detailer, hires_fix + strength + scale, preserve_face)
- **advanced** (two_pass, skin_prompt, structural_refiner)

For other presets, only the relevant groups appear. Sliders use a monospace
amber readout; toggles get a short rationale below the label.

## Keyboard shortcuts

- `⌘ + Enter` (or `Ctrl + Enter`) — submit the current composer.
- `← / →` — slide the compare lightbox handle.
- `ESC` — close the lightbox.

All interactive elements have visible focus rings (Radix primitives + Tailwind
`focus-visible:ring-1 focus-visible:ring-primary/60`).

## Worker health proxy

The webapp exposes `GET /api/worker-health` which proxies to the worker's
`/v1/health`. The SystemBar fetches it every 4 seconds and surfaces:

- `worker · {ready|warming|busy|offline}` (color-coded status dot, pulsing while live)
- `queue {N}`
- GPU model + VRAM `{used}/{total}GB`
- Uptime (counts up locally between poll cycles)
- Worker version

When the worker is offline the dot turns red and `state` reads `offline`.

## Building

```bash
cd webapp/frontend
npm install
npm run build      # emits dist/ which webapp/main.py serves
```

`npm run dev` gives Vite HMR with `/api`, `/u`, `/o` proxied to `:8001`
(see `vite.config.ts`).

## Files added in this redesign

```
webapp/frontend/src/
├── components/
│   ├── SystemBar.tsx          # top status bar + worker health probe
│   ├── PresetRail.tsx         # left preset list
│   ├── InputCard.tsx          # drag-drop + bucket info display
│   ├── PromptEditor.tsx       # prompt + negative_prompt
│   ├── PresetForm.tsx         # parameter accordion (preset-aware)
│   ├── SeedControl.tsx        # lock/random toggle, dice, manual input
│   ├── GenerateButton.tsx     # primary CTA
│   ├── JobQueue.tsx           # 2-second poll, status dots, retry
│   ├── OutputViewer.tsx       # live job follow + actions
│   └── CompareLightbox.tsx    # side-by-side compare slider
├── lib/api.ts                 # listJobs, getWorkerHealth, presetUrl, outputUrl
└── globals.css                # atelier theme tokens, font imports, motion
```

The previous `App.tsx` 2-column layout was rewritten to a 4-column dense
grid; the previous `JobProgress.tsx` and `ResultViewer.tsx` were folded
into `OutputViewer.tsx`.

## Endpoint reference

The webapp side (read-only summary):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Multipart upload. Bucket-fits image to nearest SDXL bucket. Returns `{name, bytes, bucket: [w,h]}`. Pass `?bucket=false` to skip. |
| `POST` | `/api/generate` | Proxies to worker `/v1/generate` with bearer + idempotency key. Returns `{job_id, status, out_name}`. |
| `GET` | `/api/jobs` | `?limit=20`. Returns recent job states. |
| `GET` | `/api/jobs/{id}` | One job state. |
| `POST` | `/api/callback` | HMAC-verified worker callback receiver. |
| `GET` | `/api/health` | Webapp self-health (jobs_active count). |
| `GET` | `/api/worker-health` | Proxies to worker `/v1/health`. |
| `GET` | `/u/{name}` | Serve uploaded image. |
| `PUT` | `/o/{name}` | Worker uploads result here. |
| `GET` | `/o/{name}` | Serve generated output. |
