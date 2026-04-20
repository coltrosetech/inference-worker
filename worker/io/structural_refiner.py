from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter


def apply_unsharp(output_path: Path, strength: float = 0.3) -> None:
    """UR-VTON-style structural refiner: inject high-frequency detail back into the output.

    strength in [0.0, 1.0] maps to UnsharpMask percent in [50, 250].
    Re-saves `output_path` in-place with the same format.
    """
    if strength <= 0.0:
        return
    strength = min(max(strength, 0.0), 1.0)
    percent = int(round(50 + strength * 200))
    img = Image.open(output_path)
    fmt = img.format
    refined = img.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))
    refined.save(output_path, format=fmt)
