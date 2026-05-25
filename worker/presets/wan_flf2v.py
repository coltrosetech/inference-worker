from __future__ import annotations

import secrets

from pydantic import ConfigDict, Field, field_validator

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode, Preset


def random_seed() -> int:
    return secrets.randbelow(2**63)


class WanFlf2vPreset(Preset):
    """Wan 2.1 first-last-frame video (self-hosted, no external API).

    Interpolates a coherent transition between a START frame and an END frame.
    For the wedding/garment flow: START = the original person photo (before the
    garment) and END = the try-on result (wearing the garment).
    """

    name = "wan_flf2v"
    mode = Mode.VIDEO
    template_filename = "wan_flf2v.json"
    output_extension = "mp4"
    output_content_type = "video/mp4"
    needs_reference_image = True
    warmup_timeout_sec = 900.0

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(..., min_length=1, max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        length: int = Field(default=49, ge=5, le=205)
        fps: int = Field(default=12, ge=8, le=30)
        steps: int = Field(default=8, ge=1, le=60)
        cfg: float = Field(default=1.0, ge=0.0, le=15.0)
        shift: float = Field(default=8.0, ge=1.0, le=12.0)
        lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
        width: int = Field(default=480, ge=256, le=1280)
        height: int = Field(default=832, ge=256, le=1280)

        @field_validator("length")
        @classmethod
        def _validate_length(cls, v: int) -> int:
            if (v - 1) % 4 != 0:
                raise ValueError(f"length must be 4n+1 (e.g. 81, 113, 161), got {v}")
            return v

        @field_validator("width", "height")
        @classmethod
        def _validate_div16(cls, v: int) -> int:
            if v % 16 != 0:
                raise ValueError(f"width/height must be divisible by 16, got {v}")
            return v

    Parameters = Parameters  # type: ignore[misc]

    @classmethod
    def warmup_params(cls) -> dict:
        return dict(prompt="warmup", steps=4, cfg=1.0, width=320, height=320,
                    length=21, fps=16, seed=0)

    def inject(
        self,
        template: WorkflowTemplate,
        params: "WanFlf2vPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        if not input_paths.reference_image:
            raise ValueError("wan_flf2v requires reference_image (the END frame)")
        if not template.is_api_format():
            raise NotImplementedError("wan_flf2v only supports API-format workflows")

        seed = params.seed if params.seed is not None and params.seed >= 0 else random_seed()

        template.set_input("start_image", "image", input_paths.input_image)
        template.set_input("end_image", "image", input_paths.reference_image)
        template.set_input("positive_prompt", "text", params.prompt)
        template.set_input("negative_prompt", "text", params.negative_prompt)
        template.set_input("lora", "strength_model", params.lora_strength)
        template.set_input("model_sampling", "shift", params.shift)
        template.set_input("flf2v", "width", params.width)
        template.set_input("flf2v", "height", params.height)
        template.set_input("flf2v", "length", params.length)
        template.set_input("sampler", "seed", seed)
        template.set_input("sampler", "steps", params.steps)
        template.set_input("sampler", "cfg", params.cfg)
        template.set_input("save", "frame_rate", params.fps)
        return template
