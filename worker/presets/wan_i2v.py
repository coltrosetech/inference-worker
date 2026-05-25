from __future__ import annotations

import secrets

from pydantic import ConfigDict, Field, field_validator

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode, Preset


def random_seed() -> int:
    """A random non-negative seed within ComfyUI's accepted range."""
    return secrets.randbelow(2**63)


# A quality-oriented default negative; callers may append their own.
DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, distorted, deformed, "
    "static, frozen, watermark, text, oversaturated, flickering"
)


class WanI2vPreset(Preset):
    """Image-to-video animation (Wan 2.2 I2V 14B, MoE + LightX2V 4-step).

    Replaces the old LTX-Video preset. Takes a still image — typically the
    clothed try-on result — and animates it into a short MP4 with believable
    micro-motion (breathing, fabric, hair, gentle camera). It only animates the
    supplied image; it never alters clothing coverage.

    Wan 2.2 14B is a mixture-of-experts: a HIGH-noise expert denoises the early
    steps and a LOW-noise expert finishes. Each expert gets its matching LightX2V
    4-step distill LoRA so the whole thing runs at steps=4, cfg=1 (~seconds on a
    5090) instead of the 20-step / cfg-3.5 base config.
    """

    name = "wan_i2v"
    mode = Mode.VIDEO
    template_filename = "wan_i2v.json"
    output_extension = "mp4"
    output_content_type = "video/mp4"
    warmup_timeout_sec = 600.0

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(..., min_length=1, max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        # Wan frame counts are 4n+1. At 16 fps: 49≈3s, 81≈5s, 121≈7.5s.
        length: int = Field(default=81, ge=5, le=205)
        fps: int = Field(default=16, ge=8, le=30)
        steps: int = Field(default=4, ge=2, le=60)
        cfg: float = Field(default=1.0, ge=0.0, le=15.0)  # cfg-distilled LoRA → cfg≈1
        shift: float = Field(default=5.0, ge=1.0, le=12.0)
        lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
        # Wan dims must be divisible by 16; wedding/portrait photos are vertical.
        width: int = Field(default=720, ge=256, le=1280)
        height: int = Field(default=1280, ge=256, le=1280)

        @field_validator("length")
        @classmethod
        def _validate_length(cls, v: int) -> int:
            if (v - 1) % 4 != 0:
                raise ValueError(f"length must be 4n+1 (e.g. 49, 81, 121), got {v}")
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
        params: "WanI2vPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        if not template.is_api_format():
            raise NotImplementedError("wan_i2v only supports API-format workflows")

        seed = params.seed if params.seed is not None and params.seed >= 0 else random_seed()
        negative = f"{DEFAULT_NEGATIVE}, {params.negative_prompt}".rstrip(", ")
        # MoE handoff: high-noise expert runs [0, split), low-noise expert [split, steps).
        split = max(1, params.steps // 2)

        template.set_input("input_image", "image", input_paths.input_image)
        template.set_input("positive_prompt", "text", params.prompt)
        template.set_input("negative_prompt", "text", negative)
        template.set_input("img2video", "width", params.width)
        template.set_input("img2video", "height", params.height)
        template.set_input("img2video", "length", params.length)
        template.set_input("model_sampling_high", "shift", params.shift)
        template.set_input("model_sampling_low", "shift", params.shift)
        template.set_input("lora_high", "strength_model", params.lora_strength)
        template.set_input("lora_low", "strength_model", params.lora_strength)

        template.set_input("sampler_high", "noise_seed", seed)
        template.set_input("sampler_high", "steps", params.steps)
        template.set_input("sampler_high", "cfg", params.cfg)
        template.set_input("sampler_high", "start_at_step", 0)
        template.set_input("sampler_high", "end_at_step", split)

        template.set_input("sampler_low", "noise_seed", seed)
        template.set_input("sampler_low", "steps", params.steps)
        template.set_input("sampler_low", "cfg", params.cfg)
        template.set_input("sampler_low", "start_at_step", split)
        template.set_input("sampler_low", "end_at_step", params.steps)

        template.set_input("save", "frame_rate", params.fps)
        return template
