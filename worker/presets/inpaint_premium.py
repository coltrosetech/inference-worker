from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict, Field, field_validator

from worker.comfyui.workflow import WorkflowTemplate
from worker.io.segmentation import segment_clothing_to_mask
from worker.presets.base import InputPaths, Mode, Preset
from worker.presets.edit import random_seed
from worker.presets.inpaint import ALLOWED_CATEGORIES, DEFAULT_AUTO_CATEGORIES


class InpaintPremiumPreset(Preset):
    """FLUX.1-Fill-dev — premium mask-guided regeneration.

    Much higher anatomical and fabric quality than SDXL-Lightning inpaint at
    the cost of 15–30 s per image (20 steps on RTX 5090). Shares the same
    auto-mask plumbing as `inpaint`: Python-side SegFormer produces the mask
    before the ComfyUI graph runs, so face/hair/background are preserved by
    construction.
    """

    name = "inpaint_premium"
    mode = Mode.IMAGE_PREMIUM
    template_filename = "inpaint_premium.json"
    output_extension = "png"
    output_content_type = "image/png"
    needs_mask_image = True
    warmup_timeout_sec = 600.0

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(..., min_length=1, max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        grow_mask_px: int = Field(default=12, ge=0, le=128)
        guidance: float = Field(default=30.0, ge=0.0, le=100.0)
        steps: int = Field(default=20, ge=4, le=50)
        cfg: float = Field(default=1.0, ge=0.0, le=10.0)
        auto_mask: bool = False
        auto_mask_categories: list[str] = Field(default_factory=lambda: list(DEFAULT_AUTO_CATEGORIES))

        @field_validator("auto_mask_categories")
        @classmethod
        def _validate_categories(cls, v: list[str]) -> list[str]:
            bad = [c for c in v if c not in ALLOWED_CATEGORIES]
            if bad:
                raise ValueError(
                    f"unknown auto_mask_categories {bad}; allowed: {sorted(ALLOWED_CATEGORIES)}"
                )
            if not v:
                raise ValueError("auto_mask_categories must not be empty")
            return v

    Parameters = Parameters  # type: ignore[misc]

    @classmethod
    def warmup_params(cls) -> dict:
        return dict(prompt="warmup", steps=4, cfg=1.0, guidance=30.0,
                    width=512, height=512, seed=0)

    def inject(
        self,
        template: WorkflowTemplate,
        params: "InpaintPremiumPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        if not input_paths.mask_image:
            raise ValueError(
                "inpaint_premium preset requires mask_image (or auto_mask=true, which synthesises one)"
            )

        seed = params.seed if params.seed is not None else random_seed()

        if template.is_api_format():
            template.set_input("input_image", "image", input_paths.input_image)
            template.set_input("mask_image", "image", input_paths.mask_image)
            template.set_input("grow_mask", "expand", params.grow_mask_px)
            template.set_input("positive_prompt", "text", params.prompt)
            template.set_input("negative_prompt", "text", params.negative_prompt)
            template.set_input("flux_guidance", "guidance", params.guidance)
            template.set_input("sampler", "seed", seed)
            template.set_input("sampler", "steps", params.steps)
            template.set_input("sampler", "cfg", params.cfg)
            return template

        raise NotImplementedError("inpaint_premium has no legacy full-format workflow")

    async def orchestrate(
        self,
        single_pass,
        params: "InpaintPremiumPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
        *,
        job_id: str,
        timeout_sec: float,
        cu_input_dir: Path,
        run_workflow=None,
        segment_fn=None,
    ) -> Path:
        """When `auto_mask` is on, synthesise the mask via Python-side SegFormer
        (same path as `inpaint`), then run the FLUX-Fill graph once with a
        regular manual-style mask. FLUX-Fill is single-shot — no undress /
        redress two-pass here."""
        seg = segment_fn or segment_clothing_to_mask

        if params.auto_mask:
            mask_name = f"{job_id}_automask.png"
            await seg(
                cu_input_dir / input_paths.input_image,
                cu_input_dir / mask_name,
                list(params.auto_mask_categories),
            )
            input_paths = InputPaths(
                input_image=input_paths.input_image,
                mask_image=mask_name,
                reference_image=input_paths.reference_image,
            )
            params = params.model_copy(update={"auto_mask": False})
            try:
                return await single_pass(params, input_paths, job_id, timeout_sec)
            finally:
                (cu_input_dir / mask_name).unlink(missing_ok=True)

        return await single_pass(params, input_paths, job_id, timeout_sec)
