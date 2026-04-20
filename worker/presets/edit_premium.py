from __future__ import annotations

from pydantic import ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode, Preset
from worker.presets.edit import random_seed


class EditPremiumPreset(Preset):
    """FLUX.1-Kontext-dev — premium prompt-driven image editing.

    Higher quality than Lightning img2img: native semantic editing, better
    prompt adherence, superior anatomy + text rendering. Trade-off: 20–40 s
    per image on a 16 GB GPU instead of 1.5 s.
    """

    name = "edit_premium"
    mode = Mode.IMAGE_PREMIUM
    template_filename = "edit_premium.json"
    output_extension = "png"
    output_content_type = "image/png"
    warmup_timeout_sec = 600.0

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(..., min_length=1, max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        guidance: float = Field(default=2.5, ge=0.0, le=10.0)
        steps: int = Field(default=20, ge=4, le=50)
        cfg: float = Field(default=1.0, ge=0.0, le=10.0)

    Parameters = Parameters  # type: ignore[misc]

    @classmethod
    def warmup_params(cls) -> dict:
        # Minimal warm-up to touch kernels without burning VRAM or time.
        return dict(prompt="warmup", steps=4, cfg=1.0, guidance=2.5,
                    width=512, height=512, seed=0)

    def inject(
        self,
        template: WorkflowTemplate,
        params: "EditPremiumPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        seed = params.seed if params.seed is not None else random_seed()

        if template.is_api_format():
            template.set_input("input_image", "image", input_paths.input_image)
            template.set_input("positive_prompt", "text", params.prompt)
            template.set_input("negative_prompt", "text", params.negative_prompt)
            template.set_input("flux_guidance", "guidance", params.guidance)
            template.set_input("sampler", "seed", seed)
            template.set_input("sampler", "steps", params.steps)
            template.set_input("sampler", "cfg", params.cfg)
            return template

        raise NotImplementedError("edit_premium has no legacy full-format workflow")
