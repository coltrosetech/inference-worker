# Quality Pack — `inpaint_realvis`

This document describes the post-Phase-2 enhancement layer wired into
`workflows/inpaint_realvis.json` and exposed through `worker.presets.inpaint_realvis`.
The pack adds five orthogonal stages on top of the base RealVisXL V4
inpaint sampler. Every stage is a knob: defaults aim for a natural,
non-AI-looking outfit replacement; each can be tuned or disabled.

```
                                  ┌─────────────────────────────────┐
checkpoint (RealVisXL V4 Inpaint) │ + LoRA chain (4)                │
        │                         │   detail · skin · anatomy ·     │
        │                         │   curvy (bust_emphasis)         │
        ▼                         └─────────────────────────────────┘
input_image ─┬──► VAEEncodeForInpaint ──► KSampler ──► VAEDecode ──► composite (feathered)
mask ────────┴──► GrowMask ──► FeatherMask ──► (used by composite + preserve)

composite ──► (1) PersonDetailer (YOLO26n + SEGS, 1024px guide_size)
           ──► (2) FaceDetailer  (optional, OFF by default — would change the face)
           ──► (3) HandDetailer  (yolov8s + 384px guide)
           ──► (4) 4×-UltraSharp upscale → ImageScaleBy(scale_by/4) → KSampler@hires_strength
           ──► (5) preserve_composite — pastes ORIGINAL face/skin/background back
                                        outside the feathered mask
           ──► save
```

## The five stages

### 1. LoRA stack (knobs)

Three LoRAs chained between the checkpoint and the sampler, plus an optional fourth
exposed as the `bust_emphasis` knob.

| LoRA | Default | Range | Purpose |
|---|---|---|---|
| `add_detail_xl` | `0.30` | 0..2 | Sharpens fabric, skin, hair texture |
| `realistic_skin_v5` | `0.25` | 0..2 | Pores, natural skin tone — counters waxy look |
| `body_details_xl` | `0.30` | 0..2 | Anatomy fidelity at large mask coverage |
| `curvy_body_xl` (`bust_emphasis`) | `0.0` | -1.5..+1.5 | Continuous bust/curve push (negative reduces) |

These default low (~0.3) on purpose. Earlier defaults of 0.5/0.45/0.45 produced a
recognizable "AI look" — over-sharpened detail + waxy skin. Half-strength is the
sweet spot for a casual / editorial finish.

### 2. PersonDetailer (`person_detailer: bool`, default `True`)

Solves the "subject is small in frame" failure mode. Pipeline:

1. `UltralyticsDetectorProvider` loads `yolo26n.pt` (Ultralytics/YOLO26, Jan 2026).
2. `BboxDetectorSEGS` runs detection with `labels="person"` — only person-class
   bboxes survive. crop_factor 2.5, drop_size 30 (skip noise detections).
3. `SEGSDetailer` crops the person area, scales to `guide_size=1024`, runs an
   inpaint pass at `denoise=person_detailer_strength` (default 0.32) using the
   same LoRA-stacked pipe + the same prompt/negative.
4. `SEGSPaste` composes the refined crop back into the base composite with
   `feather=8`.

**Effect**: a far-away subject occupying ~25% of the frame gets re-rendered at
near-1024px resolution within their bounding box, then pasted back. The clothing
region inside the bbox gains real detail; the rest of the image is untouched.

### 3. FaceDetailer (`face_detailer: bool`, default `False`)

The classic ADetailer pattern — `face_yolov8m.pt` bbox → inpaint at `guide_size=512`.
**Off by default** because, in this pipeline, the face is *outside* the auto-mask,
so the original face is already preserved — re-rendering it would change the face
identity. Re-enable only when you specifically want a "fix the face" pass on a
photo whose face needs improvement.

### 4. HandDetailer (`hand_detailer: bool`, default `True`)

Same pattern with `hand_yolov8s.pt` at `guide_size=384`. Hands are inside the
person bbox but tiny; this pass corrects the most common SDXL bug (extra/fused
fingers). Affects only the hand bbox, so cannot drift the rest of the image.

### 5. Hires-fix + face preservation (`hires_fix` + `preserve_face`)

After detailers, the composite is upscaled with **4×-UltraSharp** ESRGAN, then
scaled down via `ImageScaleBy(scale_by = hires_scale / 4)` so the final pixel
count is `hires_scale × bucket size` (default 1.5×). A short img2img pass at
`denoise=hires_strength` (default 0.22) on the upscaled latent sharpens the
clothing region.

The hires sampler operates on the **full image**, so without `preserve_face`
the face/skin/background drift slightly even at low denoise. The
`preserve_composite` node fixes this:

```
preserve_input_resize  ← input_image scaled by hires_scale (lanczos)
preserve_mask_to_img   ← FEATHERED grow_mask
preserve_mask_resize   ← scale to hires resolution (lanczos preserves the gradient)
preserve_mask_back     ← back to MASK
preserve_composite     ← destination = upscaled original
                       ← source      = hires_decode
                       ← mask        = upscaled feathered mask
```

Inside the feathered mask: hires-refined garment.
Outside: pixel-for-pixel original (just lanczos-resized).
At the edge: smooth gradient blend (no visible seam).

`preserve_face` defaults to `True`. Disable only if you actively want the
hires pass to influence the whole image (rare).

## Mask edges — `grow_mask_px` + `feather_mask_px`

The auto-mask from SegFormer-B2 is a hard binary. Two stages soften it:

- `GrowMask(expand=grow_mask_px, tapered_corners=true)` — extends mask outward
  by N px (default 16) so the new garment can spill over the original outfit's
  outer edge.
- `FeatherMask(left=top=right=bottom=feather_mask_px)` — gradient sönümleme
  on all four sides (default 16 px). The composite step uses this feathered
  mask, so the inpainted region blends smoothly into the original.

If you see a hard seam: bump `feather_mask_px` to 24-32. If you see leftover
fabric from the original outfit: bump `grow_mask_px` to 24-28.

## SDXL bucket pre-processor (webapp side)

`webapp.main.fit_to_sdxl_bucket` runs at upload. The webapp resizes every
upload to the closest SDXL native bucket while preserving aspect ratio:

```
1024×1024 · 1152×896 · 896×1152 · 1216×832 · 832×1216 · 1344×768 · 768×1344
```

`ImageOps.fit` does smart center-crop + lanczos resize. EXIF rotation is
applied (`ImageOps.exif_transpose`) so phone photos always land right-side up.
The output of fit_to_sdxl_bucket is what the worker actually sees — the model
operates at SDXL-native resolution regardless of what the user uploaded.

Pass `?bucket=false` to `/api/upload` to skip bucketing (rare).

## Mask coverage taxonomy (auto-mask categories)

`scripts/segment_clothing.py` (SegFormer-B2) understands these labels.
The webapp's chips UI exposes the first five by default; the rest are
opt-in via the param.

| Category | Typical use |
|---|---|
| `upper_clothes` | Shirt, top, jacket |
| `pants` | Trousers, leggings |
| `skirt` | Skirt |
| `dress` | Full dress |
| `belt` | Belt strap |
| `hat` | Hat / cap (extra) |
| `shoe` | Footwear (extra) |
| `scarf` | Scarf (extra) |
| `bag` | Handbag (extra) |
| `sunglasses` | Eyewear (extra) |

Face, skin, hair, and background are **never** included — by construction
inside the SegFormer script. This is why `preserve_face` works: the
auto-mask alone already excludes the face area; `preserve_composite` only
re-anchors it after the hires pass which would otherwise drift it.

## Tuning recipes

### "Output looks too AI / filtered"
- Drop LoRA weights: detail 0.3 → 0.2, skin 0.25 → 0.15, anatomy 0.3 → 0.2.
- Drop `cfg` from 5.5 → 5.0.
- Use a less-prescriptive prompt: replace "8k masterpiece ultra detailed"
  with "casual photo, snapshot, soft daylight".

### "Mask edge visible"
- `feather_mask_px`: 16 → 24 (or 32).
- `grow_mask_px`: 16 → 24.

### "Original outfit residue showing"
- Switch to `two_pass: true` (only available on `inpaint`/`inpaint_sdxl`/`inpaint_realvis`).
  Pass 1 fills with `skin_prompt`, pass 2 dresses with the user prompt.
  Removes the model's "majority completion" bias.

### "Subject is small in frame"
- Already handled by `person_detailer` (default ON).
- For very small subjects (<15% of frame), bump `person_detailer_strength`
  to 0.45-0.5 (default 0.32) — more aggressive re-render of the bbox.

### "Want a specific bust size"
- `bust_emphasis`: 0.0 baseline, +0.4 medium, +0.8 strong, +1.2 aggressive.
  Negative values reduce. Combine with prompt phrasing for finer control.

## Models & their HF sources

| File | Repo | Size |
|---|---|---|
| `realvisxl_v40_inpaint.safetensors` | `henrysasse193/realvisxlInpainting_v40-inpainting` | 6.9 GB |
| `add_detail_xl.safetensors` | `PvDeep/Add-Detail-XL` | 232 MB |
| `realistic_skin_v5.safetensors` | `Jonjew/RealisticSkin` | 672 MB |
| `body_details_xl.safetensors` | `NicoEmb/BodyEnhancement-LoRAs` | 307 MB |
| `body_proportion_xl.safetensors` | `NicoEmb/BodyEnhancement-LoRAs` | 10 MB |
| `curvy_body_xl.safetensors` | `G858/curvy-body-sdxl-lora` | 405 MB |
| `4x-UltraSharp.pth` | `lokCX/4x-Ultrasharp` | 67 MB |
| `face_yolov8m.pt` | `Bingsu/adetailer` | 52 MB |
| `hand_yolov8s.pt` | `Bingsu/adetailer` | 22 MB |
| `yolo26n.pt` | `Ultralytics/YOLO26` | 5.5 MB |

All entries live in `configs/models.yaml` and are pulled by
`scripts/download_models.py`.

## Custom node deps

The quality pack uses two ComfyUI custom nodes beyond the Phase-2 baseline:

- `ltdrdata/ComfyUI-Impact-Pack` — `FaceDetailer`, `BboxDetectorSEGS`,
  `SEGSDetailer`, `SEGSPaste`, `ImpactSEGSLabelFilter`, `ToBasicPipe`.
- `ltdrdata/ComfyUI-Impact-Subpack` — `UltralyticsDetectorProvider`.

Both are installed by `scripts/install_custom_nodes.sh` from
`configs/custom_nodes.yaml`. `ultralytics` PyPI package is a pip dep of
Impact-Subpack.
