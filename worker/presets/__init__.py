from __future__ import annotations

from worker.presets.base import Mode, Preset
from worker.presets.ltx_video import LtxVideoPreset
from worker.presets.tryon import TryonPreset
from worker.presets.wan_flf2v import WanFlf2vPreset


PRESETS: dict[str, Preset] = {
    p.name: p for p in (
        TryonPreset(),
        LtxVideoPreset(),
        WanFlf2vPreset(),
    )
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset: {name!r}")
    return PRESETS[name]


__all__ = ["Mode", "Preset", "PRESETS", "get_preset"]
