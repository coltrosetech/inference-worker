from __future__ import annotations

from worker.presets.base import Mode, Preset
from worker.presets.edit import EditPreset
from worker.presets.style import StylePreset


PRESETS: dict[str, Preset] = {
    p.name: p for p in (EditPreset(), StylePreset())
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset: {name!r}")
    return PRESETS[name]


__all__ = ["Mode", "Preset", "PRESETS", "get_preset"]
