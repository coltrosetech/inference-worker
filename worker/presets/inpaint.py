from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate
from worker.io.structural_refiner import apply_unsharp
from worker.presets.base import InputPaths, Mode, Preset
from worker.presets.edit import random_seed


DEFAULT_SKIN_PROMPT = "bare natural skin, torso, arms, body, soft even lighting, anatomy"
UNDRESS_NEGATIVE = "clothing, fabric, garment, shirt, sleeve, dress, jacket, coat, pattern, logo"


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
        two_pass: bool = False
        skin_prompt: str = Field(default=DEFAULT_SKIN_PROMPT, max_length=4000)
        structural_refiner: bool = False
        refiner_strength: float = Field(default=0.3, ge=0.0, le=1.0)

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

    async def orchestrate(
        self,
        single_pass,
        params: "InpaintPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
        *,
        job_id: str,
        timeout_sec: float,
        cu_input_dir: Path,
    ) -> Path:
        """Undress-to-Redress two-pass inpainting (UR-VTON-inspired).

        Pass 1 ("undress") fills the masked region with skin using `skin_prompt`,
        avoiding the "majority completion" bias that makes models re-draw clothing.
        Pass 2 ("redress") runs the user's target prompt on the skin-filled image.
        """
        async def _post(path: Path) -> Path:
            if params.structural_refiner:
                apply_unsharp(path, strength=params.refiner_strength)
            return path

        if not params.two_pass:
            out = await single_pass(params, input_paths, job_id, timeout_sec)
            return await _post(out)

        base = params.model_dump()
        for k in ("prompt", "negative_prompt", "strength", "steps",
                  "two_pass", "structural_refiner"):
            base.pop(k, None)

        undress_params = self.Parameters(
            **base,
            prompt=params.skin_prompt,
            negative_prompt=UNDRESS_NEGATIVE,
            strength=1.0,
            steps=max(4, params.steps // 2 + 2),
            two_pass=False,
            structural_refiner=False,
        )

        p1_out = await single_pass(undress_params, input_paths, f"{job_id}_p1", timeout_sec / 2)

        # Bridge pass 1's output into ComfyUI's input dir for pass 2.
        p2_input_name = f"{job_id}_p2_in.png"
        (cu_input_dir / p2_input_name).write_bytes(Path(p1_out).read_bytes())
        p2_paths = InputPaths(
            input_image=p2_input_name,
            mask_image=input_paths.mask_image,
            reference_image=input_paths.reference_image,
        )

        redress_params = self.Parameters(
            **base,
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            strength=params.strength,
            steps=params.steps,
            two_pass=False,
            structural_refiner=False,  # refiner applied once at end below
        )

        p2_out = await single_pass(redress_params, p2_paths, f"{job_id}_p2", timeout_sec / 2)

        Path(p1_out).unlink(missing_ok=True)
        (cu_input_dir / p2_input_name).unlink(missing_ok=True)
        return await _post(p2_out)
