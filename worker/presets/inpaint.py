from __future__ import annotations

from pydantic import ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode, Preset
from worker.presets.edit import random_seed


class InpaintPreset(Preset):
    name = "inpaint"
    mode = Mode.IMAGE
    template_filename = "inpaint.json"
    output_extension = "png"
    output_content_type = "image/png"
    needs_mask_image = True

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(..., min_length=1, max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        grow_mask_px: int = Field(default=8, ge=0, le=128)
        strength: float = Field(default=0.9, ge=0.0, le=1.0)
        steps: int = Field(default=6, ge=1, le=50)
        cfg: float = Field(default=1.8, ge=0.0, le=15.0)

    Parameters = Parameters  # type: ignore[misc]

    def inject(
        self,
        template: WorkflowTemplate,
        params: "InpaintPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        if not input_paths.mask_image:
            raise ValueError("inpaint preset requires mask_image")

        seed = params.seed if params.seed is not None else random_seed()

        if template.is_api_format():
            template.set_input("input_image", "image", input_paths.input_image)
            template.set_input("mask_image", "image", input_paths.mask_image)
            template.set_input("grow_mask", "expand", params.grow_mask_px)
            template.set_input("positive_prompt", "text", params.prompt)
            template.set_input("negative_prompt", "text", params.negative_prompt)
            template.set_input("sampler", "seed", seed)
            template.set_input("sampler", "steps", params.steps)
            template.set_input("sampler", "cfg", params.cfg)
            template.set_input("sampler", "denoise", params.strength)
            return template

        # Legacy full-workflow format (widget-indexed)
        template.set_widget("input_image", 0, input_paths.input_image)
        template.set_widget("mask_image", 0, input_paths.mask_image)
        template.set_widget("grow_mask", 0, params.grow_mask_px)
        template.set_widget("positive_prompt", 0, params.prompt)
        template.set_widget("negative_prompt", 0, params.negative_prompt)
        template.set_widget("sampler", 0, seed)
        template.set_widget("sampler", 1, "fixed")
        template.set_widget("sampler", 2, params.steps)
        template.set_widget("sampler", 3, params.cfg)
        template.set_widget("sampler", 6, params.strength)
        return template
