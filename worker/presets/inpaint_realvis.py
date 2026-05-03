from __future__ import annotations

from pydantic import ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode
from worker.presets.inpaint import InpaintPreset


class InpaintRealvisPreset(InpaintPreset):
    """RealVisXL V4 Inpaint with quality pack B stack.

    Pipeline: base inpaint (auto-mask preserves face/skin/background) →
    optional Face/Hand detailer → optional 4x-UltraSharp + 1.5x hires-fix →
    final mask-aware composite that paints the original face/skin/background
    back over the hires output (so hires shifts only inside the mask).
    """

    name = "inpaint_realvis"
    mode = Mode.IMAGE
    template_filename = "inpaint_realvis.json"

    class Parameters(InpaintPreset.Parameters):
        model_config = ConfigDict(extra="forbid")
        steps: int = Field(default=32, ge=1, le=60)
        cfg: float = Field(default=5.5, ge=0.0, le=15.0)
        strength: float = Field(default=0.92, ge=0.0, le=1.0)
        grow_mask_px: int = Field(default=16, ge=0, le=128)
        feather_mask_px: int = Field(default=16, ge=0, le=64)

        lora_detail_weight: float = Field(default=0.3, ge=0.0, le=2.0)
        lora_skin_weight: float = Field(default=0.25, ge=0.0, le=2.0)
        lora_anatomy_weight: float = Field(default=0.3, ge=0.0, le=2.0)
        bust_emphasis: float = Field(default=0.0, ge=-1.5, le=1.5)

        face_detailer: bool = False
        hand_detailer: bool = True
        person_detailer: bool = True
        person_detailer_strength: float = Field(default=0.32, ge=0.0, le=0.8)
        hires_fix: bool = True
        hires_strength: float = Field(default=0.22, ge=0.0, le=0.6)
        hires_scale: float = Field(default=1.5, ge=1.0, le=2.0)
        preserve_face: bool = True

    Parameters = Parameters  # type: ignore[misc]

    def inject(
        self,
        template: WorkflowTemplate,
        params: "InpaintRealvisPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        super().inject(template, params, input_paths)

        template.set_input("lora_detail", "strength_model", params.lora_detail_weight)
        template.set_input("lora_detail", "strength_clip", params.lora_detail_weight)
        template.set_input("lora_skin", "strength_model", params.lora_skin_weight)
        template.set_input("lora_skin", "strength_clip", params.lora_skin_weight)
        template.set_input("lora_anatomy", "strength_model", params.lora_anatomy_weight)
        template.set_input("lora_anatomy", "strength_clip", params.lora_anatomy_weight)
        template.set_input("lora_curvy", "strength_model", params.bust_emphasis)
        template.set_input("lora_curvy", "strength_clip", params.bust_emphasis)

        template.set_input("feather_mask", "left", params.feather_mask_px)
        template.set_input("feather_mask", "top", params.feather_mask_px)
        template.set_input("feather_mask", "right", params.feather_mask_px)
        template.set_input("feather_mask", "bottom", params.feather_mask_px)

        last = "composite"
        if params.person_detailer:
            template.set_input("person_segs", "image", [last, 0])
            template.set_input("person_segs_detailer", "image", [last, 0])
            template.set_input("person_segs_detailer", "denoise", params.person_detailer_strength)
            template.set_input("person_paste", "image", [last, 0])
            last = "person_paste"
        else:
            for node in (
                "person_provider", "person_basic_pipe", "person_segs",
                "person_segs_detailer", "person_paste",
            ):
                template.remove_node(node)

        if params.face_detailer:
            template.set_input("face_detailer", "image", [last, 0])
            last = "face_detailer"
        else:
            template.remove_node("face_detailer")
            template.remove_node("bbox_face")

        if params.hand_detailer:
            template.set_input("hand_detailer", "image", [last, 0])
            last = "hand_detailer"
        else:
            template.remove_node("hand_detailer")
            template.remove_node("bbox_hand")

        if params.hires_fix:
            template.set_input("upscale_image", "image", [last, 0])
            template.set_input("scale_down", "scale_by", params.hires_scale / 4.0)
            template.set_input("hires_sampler", "denoise", params.hires_strength)
            preserve_scale = params.hires_scale
            hires_source = "hires_decode"
        else:
            for node in (
                "upscale_loader", "upscale_image", "scale_down",
                "hires_encode", "hires_sampler", "hires_decode",
            ):
                template.remove_node(node)
            preserve_scale = 1.0
            hires_source = last

        if params.preserve_face:
            template.set_input("preserve_input_resize", "scale_by", preserve_scale)
            template.set_input("preserve_mask_resize", "scale_by", preserve_scale)
            template.set_input("preserve_composite", "source", [hires_source, 0])
            template.set_input("save", "images", ["preserve_composite", 0])
        else:
            for node in (
                "preserve_input_resize", "preserve_mask_to_img",
                "preserve_mask_resize", "preserve_mask_back",
                "preserve_composite",
            ):
                template.remove_node(node)
            template.set_input("save", "images", [hires_source, 0])

        return template
