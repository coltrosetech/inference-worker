from __future__ import annotations

from worker.presets.base import Mode, Preset
from worker.presets.tryon import TryonPreset
from worker.presets.wan_flf2v import WanFlf2vPreset
from worker.presets.wan_i2v import WanI2vPreset


PRESETS: dict[str, Preset] = {
    p.name: p for p in (
        TryonPreset(),
        WanI2vPreset(),
        WanFlf2vPreset(),
    )
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset: {name!r}")
    return PRESETS[name]


__all__ = ["Mode", "Preset", "PRESETS", "get_preset"]
