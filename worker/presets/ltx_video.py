from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode, Preset
from worker.presets.edit import random_seed


class LtxVideoPreset(Preset):
    name = "ltx_video"
    mode = Mode.VIDEO
    template_filename = "ltx_video.json"
    output_extension = "mp4"
    output_content_type = "video/mp4"
    warmup_timeout_sec = 600.0  # LTX video sampling is 30–120s

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(..., min_length=1, max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        num_frames: Literal[25, 49, 97, 121] = 97
        fps: int = Field(default=24, ge=8, le=60)
        cfg: float = Field(default=3.0, ge=0.0, le=20.0)
        steps: int = Field(default=8, ge=1, le=100)
        width: int = Field(default=768, ge=256, le=1216)
        height: int = Field(default=512, ge=256, le=704)

    Parameters = Parameters  # type: ignore[misc]

    @classmethod
    def warmup_params(cls) -> dict:
        return dict(
            prompt="warmup",
            steps=4,
            cfg=1.0,
            width=512,
            height=320,
            num_frames=25,
            fps=24,
            seed=0,
        )

    def inject(
        self,
        template: WorkflowTemplate,
        params: "LtxVideoPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        seed = params.seed if params.seed is not None else random_seed()

        if template.is_api_format():
            template.set_input("input_image", "image", input_paths.input_image)
            template.set_input("positive_prompt", "text", params.prompt)
            template.set_input("negative_prompt", "text", params.negative_prompt)
            template.set_input("img2video", "width", params.width)
            template.set_input("img2video", "height", params.height)
            template.set_input("img2video", "length", params.num_frames)
            template.set_input("conditioning", "frame_rate", float(params.fps))
            template.set_input("scheduler", "steps", params.steps)
            template.set_input("guider", "cfg", params.cfg)
            template.set_input("noise", "noise_seed", seed)
            template.set_input("save", "frame_rate", params.fps)
            return template

        raise NotImplementedError("ltx_video has no legacy full-format workflow")
