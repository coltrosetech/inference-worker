from pathlib import Path

import pytest
from PIL import Image

from worker.presets.base import InputPaths
from worker.presets.inpaint_premium import InpaintPremiumPreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_params_require_prompt():
    with pytest.raises(Exception):
        InpaintPremiumPreset.Parameters()
    p = InpaintPremiumPreset.Parameters(prompt="x")
    assert p.steps == 20
    assert p.cfg == 1.0
    assert p.guidance == 30.0
    assert p.grow_mask_px == 12
    assert p.auto_mask is False


def test_params_guidance_bounds():
    with pytest.raises(Exception):
        InpaintPremiumPreset.Parameters(prompt="x", guidance=-1)
    with pytest.raises(Exception):
        InpaintPremiumPreset.Parameters(prompt="x", guidance=101)


def test_params_auto_mask_categories_validation():
    with pytest.raises(Exception):
        InpaintPremiumPreset.Parameters(prompt="x", auto_mask_categories=["tie"])
    with pytest.raises(Exception):
        InpaintPremiumPreset.Parameters(prompt="x", auto_mask_categories=[])


def test_inject_requires_mask():
    preset = InpaintPremiumPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = InpaintPremiumPreset.Parameters(prompt="x")
    with pytest.raises(ValueError):
        preset.inject(tpl, params, InputPaths(input_image="in.png", mask_image=None))


def test_inject_default_drops_pose_branch():
    preset = InpaintPremiumPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = InpaintPremiumPreset.Parameters(prompt="x", use_pose_guide=False)
    out = preset.inject(tpl, params, InputPaths(input_image="in.png", mask_image="m.png")).to_dict()
    for node in ("pose_preprocessor", "pose_controlnet_loader", "pose_type_setter", "pose_apply"):
        assert node not in out, f"{node} should be removed when use_pose_guide=False"
    # inpaint_cond still reads directly from the prompts
    assert out["inpaint_cond"]["inputs"]["positive"] == ["positive_prompt", 0]
    assert out["inpaint_cond"]["inputs"]["negative"] == ["negative_prompt", 0]


def test_inject_with_pose_guide_keeps_and_wires_branch():
    preset = InpaintPremiumPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = InpaintPremiumPreset.Parameters(
        prompt="x", use_pose_guide=True, pose_strength=0.7,
    )
    out = preset.inject(tpl, params, InputPaths(input_image="in.png", mask_image="m.png")).to_dict()
    for node in ("pose_preprocessor", "pose_controlnet_loader", "pose_type_setter", "pose_apply"):
        assert node in out, f"{node} should be retained when use_pose_guide=True"
    assert out["pose_apply"]["inputs"]["strength"] == 0.7
    # inpaint_cond now sources through pose_apply
    assert out["inpaint_cond"]["inputs"]["positive"] == ["pose_apply", 0]
    assert out["inpaint_cond"]["inputs"]["negative"] == ["pose_apply", 1]


def test_inject_sets_flux_fill_nodes():
    preset = InpaintPremiumPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = InpaintPremiumPreset.Parameters(
        prompt="red dress", negative_prompt="ugly",
        guidance=35.0, steps=25, cfg=1.0, grow_mask_px=16, seed=42,
    )
    paths = InputPaths(input_image="in.png", mask_image="m.png")
    out = preset.inject(tpl, params, paths).to_dict()

    assert out["input_image"]["inputs"]["image"] == "in.png"
    assert out["mask_image"]["inputs"]["image"] == "m.png"
    assert out["grow_mask"]["inputs"]["expand"] == 16
    assert out["positive_prompt"]["inputs"]["text"] == "red dress"
    assert out["negative_prompt"]["inputs"]["text"] == "ugly"
    assert out["flux_guidance"]["inputs"]["guidance"] == 35.0
    sampler = out["sampler"]["inputs"]
    assert sampler["seed"] == 42
    assert sampler["steps"] == 25
    assert sampler["cfg"] == 1.0
    # InpaintModelConditioning wired into KSampler
    assert sampler["positive"] == ["flux_guidance", 0]
    assert sampler["negative"] == ["inpaint_cond", 1]
    assert sampler["latent_image"] == ["inpaint_cond", 2]


def test_output_extension():
    assert InpaintPremiumPreset.output_extension == "png"
    assert InpaintPremiumPreset.output_content_type == "image/png"


async def test_orchestrate_passthrough_when_manual_mask(tmp_path):
    preset = InpaintPremiumPreset()
    calls: list = []

    async def fake_single(params, paths, jid, ts):
        calls.append((params.prompt, paths.mask_image, params.auto_mask))
        out = tmp_path / f"{jid}_out.png"
        Image.new("RGB", (8, 8), "gold").save(out)
        return out

    params = InpaintPremiumPreset.Parameters(prompt="sky", auto_mask=False)
    paths = InputPaths(input_image="in.png", mask_image="m.png")
    result = await preset.orchestrate(
        fake_single, params, paths,
        job_id="jP", timeout_sec=60.0, cu_input_dir=tmp_path,
    )
    assert len(calls) == 1
    assert calls[0] == ("sky", "m.png", False)
    assert result.exists()


async def test_orchestrate_auto_mask_synthesises_then_falls_through(tmp_path):
    preset = InpaintPremiumPreset()
    seg_calls: list = []
    single_calls: list = []

    async def fake_segment(in_path, out_path, categories):
        seg_calls.append({"in": in_path.name, "cats": list(categories)})
        Image.new("L", (8, 8), 255).save(out_path)
        return out_path

    async def fake_single(params, paths, jid, ts):
        single_calls.append({"mask": paths.mask_image, "auto": params.auto_mask})
        out = tmp_path / f"{jid}_out.png"
        Image.new("RGB", (8, 8), "red").save(out)
        return out

    (tmp_path / "orig.png").write_bytes(b"fake")
    params = InpaintPremiumPreset.Parameters(
        prompt="emerald gown", auto_mask=True,
        auto_mask_categories=["upper_clothes", "dress"],
    )
    paths = InputPaths(input_image="orig.png", mask_image=None)
    result = await preset.orchestrate(
        fake_single, params, paths,
        job_id="jQ", timeout_sec=120.0, cu_input_dir=tmp_path,
        segment_fn=fake_segment,
    )

    assert len(seg_calls) == 1
    assert seg_calls[0]["cats"] == ["upper_clothes", "dress"]
    assert len(single_calls) == 1
    assert single_calls[0]["mask"] == "jQ_automask.png"
    assert single_calls[0]["auto"] is False
    assert result.exists()
    # Temp mask cleaned up.
    assert not (tmp_path / "jQ_automask.png").exists()
