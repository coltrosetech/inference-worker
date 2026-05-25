from __future__ import annotations

import secrets
from typing import Literal

from pydantic import ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode, Preset


def random_seed() -> int:
    """A random non-negative seed within ComfyUI's accepted range."""
    return secrets.randbelow(2**63)


class LtxVideoPreset(Preset):
    """Image-to-video animation (LTX-Video 2B distilled).

    Takes a still image — typically the clothed try-on result — and animates
    it into a short MP4 (subtle motion, fabric movement, gentle turn). It only
    animates the supplied image; it never alters clothing coverage.
    """

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
        # LTX frame counts must be 8n+1. At 24 fps: 97≈4s, 121≈5s, 169≈7s, 241≈10s.
        num_frames: Literal[25, 49, 97, 121, 169, 241] = 169  # default ≈7s @ 24fps
        fps: int = Field(default=24, ge=8, le=60)
        cfg: float = Field(default=1.0, ge=0.0, le=20.0)  # distilled LTX → cfg≈1
        steps: int = Field(default=8, ge=1, le=100)
        # LTX accepts dims divisible by 32; allow portrait too (wedding photos
        # are usually vertical, so height can exceed the old landscape-only 704).
        width: int = Field(default=768, ge=256, le=1216)
        height: int = Field(default=512, ge=256, le=1216)

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
        if not template.is_api_format():
            raise NotImplementedError("ltx_video has no legacy full-format workflow")

        seed = params.seed if params.seed is not None and params.seed >= 0 else random_seed()

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
