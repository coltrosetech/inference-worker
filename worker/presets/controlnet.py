from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode, Preset
from worker.presets.edit import random_seed


ControlnetType = Literal["canny", "depth", "pose", "lineart", "scribble"]


# Map high-level controlnet_type → (preprocessor class_type, union type string)
_CN_TYPE_MAP: dict[str, tuple[str, str]] = {
    "canny":    ("CannyEdgePreprocessor",      "canny/lineart/anime_lineart/mlsd"),
    "depth":    ("DepthAnythingV2Preprocessor", "depth"),
    "pose":     ("DWPreprocessor",             "openpose"),
    "lineart":  ("AnyLineArtPreprocessor_aux", "canny/lineart/anime_lineart/mlsd"),
    "scribble": ("ScribblePreprocessor",       "hed/pidi/scribble/ted"),
}


class ControlnetPreset(Preset):
    name = "controlnet"
    mode = Mode.IMAGE
    template_filename = "controlnet.json"
    output_extension = "png"
    output_content_type = "image/png"

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(..., min_length=1, max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        controlnet_type: ControlnetType = "canny"
        controlnet_strength: float = Field(default=0.8, ge=0.0, le=2.0)
        strength: float = Field(default=0.7, ge=0.0, le=1.0)
        steps: int = Field(default=6, ge=1, le=50)
        cfg: float = Field(default=1.8, ge=0.0, le=15.0)

    Parameters = Parameters  # type: ignore[misc]

    def inject(
        self,
        template: WorkflowTemplate,
        params: "ControlnetPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        seed = params.seed if params.seed is not None else random_seed()
        preproc, union_type = _CN_TYPE_MAP[params.controlnet_type]

        if template.is_api_format():
            template.set_input("input_image", "image", input_paths.input_image)
            template.set_input("preprocessor", "preprocessor", preproc)
            template.set_input("cn_union_type", "type", union_type)
            template.set_input("cn_apply", "strength", params.controlnet_strength)
            template.set_input("positive_prompt", "text", params.prompt)
            template.set_input("negative_prompt", "text", params.negative_prompt)
            template.set_input("sampler", "seed", seed)
            template.set_input("sampler", "steps", params.steps)
            template.set_input("sampler", "cfg", params.cfg)
            template.set_input("sampler", "denoise", params.strength)
            return template

        template.set_widget("input_image", 0, input_paths.input_image)
        template.set_widget("preprocessor", 0, preproc)
        template.set_widget("cn_union_type", 0, union_type)
        template.set_widget("cn_apply", 0, params.controlnet_strength)
        template.set_widget("positive_prompt", 0, params.prompt)
        template.set_widget("negative_prompt", 0, params.negative_prompt)
        template.set_widget("sampler", 0, seed)
        template.set_widget("sampler", 1, "fixed")
        template.set_widget("sampler", 2, params.steps)
        template.set_widget("sampler", 3, params.cfg)
        template.set_widget("sampler", 6, params.strength)
        return template
