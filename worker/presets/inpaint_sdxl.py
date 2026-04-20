from __future__ import annotations

from pydantic import ConfigDict, Field

from worker.presets.base import Mode
from worker.presets.inpaint import InpaintPreset


class InpaintSdxlPreset(InpaintPreset):
    """SDXL inpaint-fine-tuned (JuggernautXL v9 Inpaint).

    Middle ground between the Lightning `inpaint` (fast, few-step) and
    FLUX-Fill `inpaint_premium` (slow, highest quality): full SDXL inpaint
    fine-tune, 25 steps, ~3-5 s on RTX 5090. Shares auto-mask + two-pass
    with the base InpaintPreset — only the checkpoint and sampler defaults
    differ.
    """

    name = "inpaint_sdxl"
    mode = Mode.IMAGE
    template_filename = "inpaint_sdxl.json"

    class Parameters(InpaintPreset.Parameters):
        model_config = ConfigDict(extra="forbid")
        # SDXL inpaint fine-tunes run at standard (non-Lightning) settings.
        steps: int = Field(default=25, ge=1, le=60)
        cfg: float = Field(default=7.0, ge=0.0, le=15.0)

    Parameters = Parameters  # type: ignore[misc]
