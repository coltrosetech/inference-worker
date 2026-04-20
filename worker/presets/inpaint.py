from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from worker.comfyui.workflow import WorkflowTemplate
from worker.io.structural_refiner import apply_unsharp
from worker.presets.base import InputPaths, Mode, Preset
from worker.presets.edit import random_seed


DEFAULT_SKIN_PROMPT = "bare natural skin, torso, arms, body, soft even lighting, anatomy"
UNDRESS_NEGATIVE = "clothing, fabric, garment, shirt, sleeve, dress, jacket, coat, pattern, logo"

# Map canonical lowercase category keys to the mixed-case input names the
# `segformer_b2_clothes` ComfyUI node actually uses. Keep in lockstep with
# workflows/inpaint.json and workflows/_segment_clothing.json.
CATEGORY_TO_NODE_KEY: dict[str, str] = {
    "face": "Face",
    "hat": "Hat",
    "hair": "Hair",
    "upper_clothes": "Upper_clothes",
    "skirt": "Skirt",
    "pants": "Pants",
    "dress": "Dress",
    "belt": "Belt",
    "shoe": "shoe",
    "leg": "leg",
    "arm": "arm",
    "bag": "Bag",
    "scarf": "Scarf",
}
ALLOWED_CATEGORIES: frozenset[str] = frozenset(CATEGORY_TO_NODE_KEY.keys())
DEFAULT_AUTO_CATEGORIES: list[str] = ["upper_clothes", "pants", "skirt", "dress", "belt"]


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

    def inject(
        self,
        template: WorkflowTemplate,
        params: "InpaintPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        if not params.auto_mask and not input_paths.mask_image:
            raise ValueError("inpaint preset requires mask_image or auto_mask=true")

        seed = params.seed if params.seed is not None else random_seed()

        if template.is_api_format():
            template.set_input("input_image", "image", input_paths.input_image)
            template.set_input("grow_mask", "expand", params.grow_mask_px)
            template.set_input("positive_prompt", "text", params.prompt)
            template.set_input("negative_prompt", "text", params.negative_prompt)
            template.set_input("sampler", "seed", seed)
            template.set_input("sampler", "steps", params.steps)
            template.set_input("sampler", "cfg", params.cfg)
            template.set_input("sampler", "denoise", params.strength)

            if params.auto_mask:
                # Enable every category (=> keep that region), then disable the
                # user-selected ones so they become inpaint targets.
                for cat, node_key in CATEGORY_TO_NODE_KEY.items():
                    template.set_input("auto_mask_segmenter", node_key, True)
                for cat in params.auto_mask_categories:
                    template.set_input("auto_mask_segmenter", CATEGORY_TO_NODE_KEY[cat], False)
                # Route the segmenter's mask through ImageToMask into grow_mask.
                template.set_input("grow_mask", "mask", ["auto_mask_to_mask", 0])
                # Drop the manual-mask loader so ComfyUI doesn't try to read a
                # placeholder file that doesn't exist.
                template.remove_node("mask_image")
            else:
                # Manual-mask path: use the uploaded mask, drop the auto branch.
                template.set_input("mask_image", "image", input_paths.mask_image)
                template.remove_node("auto_mask_segmenter")
                template.remove_node("auto_mask_to_mask")
            return template

        # Legacy full-workflow format (widget-indexed) — auto-mask not supported here.
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
        run_workflow=None,
    ) -> Path:
        """Inpaint orchestration:

        - Single-pass (manual or auto-mask): `inject()` handles everything in
          one ComfyUI call.
        - Two-pass manual: undress → redress, both runs read the same uploaded
          mask (existing behavior).
        - Two-pass + auto-mask: run the segment-only side workflow once, save
          the mask to ComfyUI's input dir, then re-enter the two-pass manual
          path with that mask so both passes agree on what to inpaint.
        """
        async def _post(path: Path) -> Path:
            if params.structural_refiner:
                apply_unsharp(path, strength=params.refiner_strength)
            return path

        if not params.two_pass:
            out = await single_pass(params, input_paths, job_id, timeout_sec)
            return await _post(out)

        if params.auto_mask:
            if run_workflow is None:
                raise RuntimeError("executor did not supply run_workflow; cannot pre-segment")
            seg_wf = self._build_segment_workflow(
                input_paths.input_image, params.auto_mask_categories
            )
            seg_out = await run_workflow(seg_wf, f"{job_id}_seg", min(timeout_sec, 120.0), "png")
            mask_name = f"{job_id}_automask.png"
            (cu_input_dir / mask_name).write_bytes(Path(seg_out).read_bytes())
            Path(seg_out).unlink(missing_ok=True)
            input_paths = InputPaths(
                input_image=input_paths.input_image,
                mask_image=mask_name,
                reference_image=input_paths.reference_image,
            )
            params = params.model_copy(update={"auto_mask": False})
            try:
                return await self._two_pass_manual(
                    single_pass, params, input_paths, job_id, timeout_sec, cu_input_dir, _post
                )
            finally:
                (cu_input_dir / mask_name).unlink(missing_ok=True)

        return await self._two_pass_manual(
            single_pass, params, input_paths, job_id, timeout_sec, cu_input_dir, _post
        )

    async def _two_pass_manual(
        self,
        single_pass,
        params: "InpaintPreset.Parameters",
        input_paths: InputPaths,
        job_id: str,
        timeout_sec: float,
        cu_input_dir: Path,
        _post,
    ) -> Path:
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
            structural_refiner=False,
        )
        p2_out = await single_pass(redress_params, p2_paths, f"{job_id}_p2", timeout_sec / 2)

        Path(p1_out).unlink(missing_ok=True)
        (cu_input_dir / p2_input_name).unlink(missing_ok=True)
        return await _post(p2_out)

    @staticmethod
    def _build_segment_workflow(input_image: str, categories: list[str]) -> dict[str, Any]:
        """Inline the tiny _segment_clothing graph so we don't need a file read here."""
        flags = {node_key: True for node_key in CATEGORY_TO_NODE_KEY.values()}
        for cat in categories:
            flags[CATEGORY_TO_NODE_KEY[cat]] = False
        return {
            "input_image": {
                "class_type": "LoadImage",
                "inputs": {"image": input_image},
            },
            "auto_mask_segmenter": {
                "class_type": "segformer_b2_clothes",
                "inputs": {"image": ["input_image", 0], **flags},
            },
            "save": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "automask", "images": ["auto_mask_segmenter", 0]},
            },
        }
