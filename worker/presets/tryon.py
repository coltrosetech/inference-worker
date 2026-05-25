from __future__ import annotations

import secrets
from pathlib import Path

from pydantic import ConfigDict, Field, field_validator

from worker.comfyui.workflow import WorkflowTemplate
from worker.io.segmentation import segment_clothing_to_mask
from worker.presets.base import InputPaths, Mode, Preset


def random_seed() -> int:
    """A random non-negative seed within ComfyUI's accepted range."""
    return secrets.randbelow(2**63)


# Baked into every try-on job's negative prompt and not removable by callers.
# Try-on always *replaces* clothing with a target garment; it must never be
# steerable toward producing nudity or removing a subject's clothing.
SAFETY_NEGATIVE = (
    "nude, nudity, naked, undressed, topless, bottomless, bare skin, "
    "exposed breasts, exposed genitals, lingerie, underwear, sexual, nsfw"
)

# SegFormer clothing-parser labels we allow as auto-mask targets. Background,
# face, hair, and skin are excluded by construction in segment_clothing.py, so
# they are never marked for replacement.
ALLOWED_CATEGORIES: frozenset[str] = frozenset({
    "upper_clothes", "pants", "skirt", "dress", "belt",
    "hat", "shoe", "scarf", "bag", "sunglasses",
})
DEFAULT_AUTO_CATEGORIES: list[str] = ["upper_clothes", "pants", "skirt", "dress", "belt"]

# Nodes that make up the optional hires-fix refine tail in tryon.json.
_HIRES_NODES = (
    "upscale_loader", "hires_upscale", "hires_scale_down",
    "hires_encode", "hires_sampler", "hires_decode",
)


class TryonPreset(Preset):
    """Reference-image-driven virtual try-on (the only image preset shipped).

    Combines JuggernautXL Inpaint with IP-Adapter Plus SDXL: the caller
    supplies a garment photo as `reference_image_url`, SegFormer (via the
    shared auto-mask pipeline) carves out the wearer's current clothing
    region, and the diffusion is conditioned on the reference garment so the
    new outfit visually resembles it — not just what the prompt says.

    An optional hires-fix refine pass (4x-UltraSharp upscale → low-denoise
    resample) sharpens fabric/detail without changing identity. Output is
    always clothed: the nudity-exclusion negative is baked in and the garment
    reference is required, so there is no "undress" path.
    """

    name = "tryon"
    mode = Mode.IMAGE
    template_filename = "tryon.json"
    output_extension = "png"
    output_content_type = "image/png"
    needs_mask_image = True
    needs_reference_image = True

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(default="wearing the reference garment, natural lighting, photorealistic",
                            max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        grow_mask_px: int = Field(default=12, ge=0, le=128)
        strength: float = Field(default=0.9, ge=0.0, le=1.0)
        steps: int = Field(default=32, ge=1, le=60)
        cfg: float = Field(default=7.0, ge=0.0, le=15.0)
        reference_weight: float = Field(default=0.9, ge=0.0, le=2.0)
        auto_mask: bool = True
        auto_mask_categories: list[str] = Field(default_factory=lambda: list(DEFAULT_AUTO_CATEGORIES))
        # Quality: hires-fix refine pass (core upscale nodes, no extra deps).
        hires_fix: bool = True
        hires_strength: float = Field(default=0.25, ge=0.0, le=0.6)
        hires_scale: float = Field(default=1.5, ge=1.0, le=2.0)

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
        return dict(prompt="warmup", steps=4, cfg=2.0, width=512, height=512, seed=0)

    def inject(
        self,
        template: WorkflowTemplate,
        params: "TryonPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        # The garment reference is required (no "empty garment" path) and the
        # nudity-exclusion negative is baked in and cannot be overridden.
        if not input_paths.reference_image:
            raise ValueError("tryon preset requires reference_image (the target garment photo)")
        if not input_paths.mask_image:
            raise ValueError("tryon preset requires mask_image (or auto_mask=true)")

        if not template.is_api_format():
            raise NotImplementedError("tryon has no legacy full-format workflow")

        seed = params.seed if params.seed is not None and params.seed >= 0 else random_seed()
        negative = f"{SAFETY_NEGATIVE}, {params.negative_prompt}".rstrip(", ")

        template.set_input("input_image", "image", input_paths.input_image)
        template.set_input("reference_image", "image", input_paths.reference_image)
        template.set_input("mask_image", "image", input_paths.mask_image)
        template.set_input("grow_mask", "expand", params.grow_mask_px)
        template.set_input("positive_prompt", "text", params.prompt)
        template.set_input("negative_prompt", "text", negative)
        template.set_input("ipadapter_apply", "weight", params.reference_weight)
        template.set_input("sampler", "seed", seed)
        template.set_input("sampler", "steps", params.steps)
        template.set_input("sampler", "cfg", params.cfg)
        template.set_input("sampler", "denoise", params.strength)

        # Hires-fix refine tail: upscale the composited result then re-sample at
        # low denoise so fabric/detail sharpen while identity is preserved.
        if params.hires_fix:
            template.set_input("hires_upscale", "image", ["composite", 0])
            template.set_input("hires_scale_down", "scale_by", params.hires_scale / 4.0)
            template.set_input("hires_sampler", "seed", seed)
            template.set_input("hires_sampler", "steps", max(8, params.steps // 2))
            template.set_input("hires_sampler", "cfg", params.cfg)
            template.set_input("hires_sampler", "denoise", params.hires_strength)
            template.set_input("save", "images", ["hires_decode", 0])
        else:
            for node in _HIRES_NODES:
                template.remove_node(node)
            template.set_input("save", "images", ["composite", 0])

        return template

    async def orchestrate(
        self,
        single_pass,
        params: "TryonPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
        *,
        job_id: str,
        timeout_sec: float,
        cu_input_dir: Path,
        run_workflow=None,
        segment_fn=None,
    ) -> Path:
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
