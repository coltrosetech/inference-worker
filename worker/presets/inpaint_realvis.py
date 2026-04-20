from __future__ import annotations

from pydantic import ConfigDict, Field

from worker.presets.base import Mode
from worker.presets.inpaint import InpaintPreset


class InpaintRealvisPreset(InpaintPreset):
    """RealVisXL V4 Inpaint — photorealistic SDXL inpaint fine-tune.

    Leans toward naturalistic skin, lighting, and fabric (portrait / fashion
    photography aesthetic) compared to the more stylised Juggernaut variant.
    ~4-6 s on RTX 5090 at 30 steps. Shares auto-mask + two-pass with the
    base InpaintPreset.
    """

    name = "inpaint_realvis"
    mode = Mode.IMAGE
    template_filename = "inpaint_realvis.json"

    class Parameters(InpaintPreset.Parameters):
        model_config = ConfigDict(extra="forbid")
        steps: int = Field(default=30, ge=1, le=60)
        cfg: float = Field(default=6.5, ge=0.0, le=15.0)

    Parameters = Parameters  # type: ignore[misc]
