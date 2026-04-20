#!/usr/bin/env python
"""Standalone CLI: run SegFormer-B2-Clothes on an image and write a binary
mask where *only* the requested clothing categories are white (255). Every
other label — including background, face, hair, skin — stays black (0), so
the resulting mask is safe to feed into an inpaint workflow without
destroying regions the user wants preserved.

Designed to be invoked as a subprocess from the worker (which does not have
torch/transformers in its own venv). Use the ComfyUI venv:

    /venv/main/bin/python scripts/segment_clothing.py \
        --input /path/in.png \
        --output /path/mask.png \
        --categories upper_clothes,pants \
        --model-dir /workspace/ComfyUI/models/segformer_b2_clothes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


LABEL_TO_IDX: dict[str, int] = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "shoe_l": 9,
    "shoe_r": 10,
    "face": 11,
    "leg_l": 12,
    "leg_r": 13,
    "arm_l": 14,
    "arm_r": 15,
    "bag": 16,
    "scarf": 17,
}

# Convenience aliases so callers don't have to split L/R themselves.
ALIASES: dict[str, list[str]] = {
    "shoe": ["shoe_l", "shoe_r"],
    "shoes": ["shoe_l", "shoe_r"],
    "leg": ["leg_l", "leg_r"],
    "legs": ["leg_l", "leg_r"],
    "arm": ["arm_l", "arm_r"],
    "arms": ["arm_l", "arm_r"],
}


def resolve_categories(raw: list[str]) -> list[int]:
    idxs: set[int] = set()
    for name in raw:
        key = name.strip().lower().replace("-", "_")
        if not key:
            continue
        if key in ALIASES:
            for a in ALIASES[key]:
                idxs.add(LABEL_TO_IDX[a])
            continue
        if key not in LABEL_TO_IDX:
            raise SystemExit(
                f"unknown category {name!r}; allowed: "
                f"{sorted(list(LABEL_TO_IDX.keys()) + list(ALIASES.keys()))}"
            )
        idxs.add(LABEL_TO_IDX[key])
    if not idxs:
        raise SystemExit("no categories specified")
    return sorted(idxs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--categories",
        required=True,
        help="comma-separated list (e.g. upper_clothes,pants,dress)",
    )
    ap.add_argument(
        "--model-dir",
        default="/workspace/ComfyUI/models/segformer_b2_clothes",
    )
    ap.add_argument(
        "--device",
        default="cuda" if _cuda_available() else "cpu",
    )
    args = ap.parse_args()

    # Import heavy deps only after argparse so --help stays snappy.
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

    cats = resolve_categories(args.categories.split(","))
    proc = SegformerImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_dir).to(args.device).eval()

    img = Image.open(args.input).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits  # (1, C, H', W')
    up = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )
    pred = up.argmax(dim=1)[0].cpu().numpy()  # (H, W), values 0..17

    mask = np.isin(pred, cats).astype("uint8") * 255
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_path, format="PNG")
    hit = int(mask.sum()) // 255
    total = mask.size
    print(
        f"wrote {out_path} ({hit}/{total} px = {100*hit/total:.1f}% inpainted)",
        file=sys.stderr,
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
