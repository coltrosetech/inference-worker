from __future__ import annotations

import secrets

from pydantic import ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode, Preset


def random_seed() -> int:
    return secrets.randbits(63)


class EditPreset(Preset):
    name = "edit"
    mode = Mode.IMAGE
    template_filename = "edit.json"
    output_extension = "png"
    output_content_type = "image/png"

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(..., min_length=1, max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        strength: float = Field(default=0.7, ge=0.0, le=1.0)
        preservation: float = Field(default=0.5, ge=0.0, le=1.5)
        steps: int = Field(default=6, ge=1, le=50)
        cfg: float = Field(default=1.8, ge=0.0, le=15.0)

    Parameters = Parameters  # type: ignore[misc]

    def inject(
        self,
        template: WorkflowTemplate,
        params: "EditPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        seed = params.seed if params.seed is not None and params.seed >= 0 else random_seed()

        if template.is_api_format():
            template.set_input("input_image", "image", input_paths.input_image)
            template.set_input("positive_prompt", "text", params.prompt)
            template.set_input("negative_prompt", "text", params.negative_prompt)
            try:
                template.set_input("ipa_apply", "weight", params.preservation)
            except KeyError:
                pass
            template.set_input("sampler", "seed", seed)
            template.set_input("sampler", "steps", params.steps)
            template.set_input("sampler", "cfg", params.cfg)
            template.set_input("sampler", "denoise", params.strength)
            return template

        # Legacy full-workflow format (widget-indexed)
        template.set_widget("input_image", 0, input_paths.input_image)
        template.set_widget("positive_prompt", 0, params.prompt)
        template.set_widget("negative_prompt", 0, params.negative_prompt)
        template.set_widget("sampler", 0, seed)
        template.set_widget("sampler", 1, "fixed")
        template.set_widget("sampler", 2, params.steps)
        template.set_widget("sampler", 3, params.cfg)
        template.set_widget("sampler", 6, params.strength)
        return template
