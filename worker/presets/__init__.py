from __future__ import annotations

from worker.presets.base import Mode, Preset
from worker.presets.controlnet import ControlnetPreset
from worker.presets.edit import EditPreset
from worker.presets.edit_premium import EditPremiumPreset
from worker.presets.inpaint import InpaintPreset
from worker.presets.ltx_video import LtxVideoPreset
from worker.presets.style import StylePreset


PRESETS: dict[str, Preset] = {
    p.name: p for p in (
        EditPreset(),
        StylePreset(),
        ControlnetPreset(),
        InpaintPreset(),
        EditPremiumPreset(),
        LtxVideoPreset(),
    )
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset: {name!r}")
    return PRESETS[name]


__all__ = ["Mode", "Preset", "PRESETS", "get_preset"]
