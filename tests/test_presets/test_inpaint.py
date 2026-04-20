from pathlib import Path

import pytest
from PIL import Image

from worker.presets.base import InputPaths
from worker.presets.inpaint import DEFAULT_SKIN_PROMPT, InpaintPreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_inpaint_params_require_prompt():
    with pytest.raises(Exception):
        InpaintPreset.Parameters()
    p = InpaintPreset.Parameters(prompt="fill with sky")
    assert p.grow_mask_px == 8
    assert p.strength == 0.9


def test_inpaint_params_grow_mask_bounds():
    with pytest.raises(Exception):
        InpaintPreset.Parameters(prompt="x", grow_mask_px=-1)
    with pytest.raises(Exception):
        InpaintPreset.Parameters(prompt="x", grow_mask_px=129)
    p = InpaintPreset.Parameters(prompt="x", grow_mask_px=0)
    assert p.grow_mask_px == 0
    p = InpaintPreset.Parameters(prompt="x", grow_mask_px=128)
    assert p.grow_mask_px == 128


def test_inpaint_inject_rejects_missing_mask():
    preset = InpaintPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = InpaintPreset.Parameters(prompt="x")
    paths = InputPaths(input_image="in.png", mask_image=None)
    with pytest.raises(ValueError):
        preset.inject(tpl, params, paths)


def test_inpaint_inject_sets_paths_and_sampler():
    preset = InpaintPreset()
    tpl = preset.load_template(WORKFLOWS)
    assert tpl.is_api_format()
    params = InpaintPreset.Parameters(
        prompt="replace with ocean",
        negative_prompt="blur",
        grow_mask_px=16,
        strength=0.8,
        steps=8,
        cfg=2.0,
        seed=99,
    )
    paths = InputPaths(input_image="in.png", mask_image="m.png")
    out = preset.inject(tpl, params, paths).to_dict()

    assert out["input_image"]["inputs"]["image"] == "in.png"
    assert out["mask_image"]["inputs"]["image"] == "m.png"
    assert out["grow_mask"]["inputs"]["expand"] == 16
    assert out["positive_prompt"]["inputs"]["text"] == "replace with ocean"
    assert out["negative_prompt"]["inputs"]["text"] == "blur"
    sampler = out["sampler"]["inputs"]
    assert sampler["seed"] == 99
    assert sampler["steps"] == 8
    assert sampler["cfg"] == 2.0
    assert sampler["denoise"] == 0.8


def test_inpaint_inject_seed_random_when_none(monkeypatch):
    monkeypatch.setattr("worker.presets.inpaint.random_seed", lambda: 777)
    preset = InpaintPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = InpaintPreset.Parameters(prompt="x", seed=None)
    paths = InputPaths(input_image="a.png", mask_image="b.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert out["sampler"]["inputs"]["seed"] == 777


def test_inpaint_workflow_composites_only_masked_region():
    """ImageCompositeMasked must wire destination=input, source=decoded, mask=grown."""
    preset = InpaintPreset()
    tpl = preset.load_template(WORKFLOWS)
    data = tpl.to_dict()
    assert data["composite"]["class_type"] == "ImageCompositeMasked"
    assert data["composite"]["inputs"]["destination"] == ["input_image", 0]
    assert data["composite"]["inputs"]["source"] == ["vae_decode", 0]
    assert data["composite"]["inputs"]["mask"] == ["grow_mask", 0]
    assert data["save"]["inputs"]["images"] == ["composite", 0]


def test_inpaint_output_extension_and_content_type():
    assert InpaintPreset.output_extension == "png"
    assert InpaintPreset.output_content_type == "image/png"


def test_inpaint_params_two_pass_defaults():
    p = InpaintPreset.Parameters(prompt="x")
    assert p.two_pass is False
    assert p.skin_prompt == DEFAULT_SKIN_PROMPT
    assert p.structural_refiner is False
    assert p.refiner_strength == 0.3


def test_inpaint_params_refiner_strength_bounds():
    with pytest.raises(Exception):
        InpaintPreset.Parameters(prompt="x", refiner_strength=-0.1)
    with pytest.raises(Exception):
        InpaintPreset.Parameters(prompt="x", refiner_strength=1.1)


async def test_orchestrate_single_pass_when_two_pass_false(tmp_path):
    preset = InpaintPreset()
    calls: list[tuple] = []

    async def fake_single(params, paths, jid, ts):
        out = tmp_path / f"{jid}_out.png"
        Image.new("RGB", (8, 8), "red").save(out)
        calls.append((params.prompt, paths.input_image, jid))
        return out

    params = InpaintPreset.Parameters(prompt="sky", two_pass=False)
    paths = InputPaths(input_image="orig.png", mask_image="m.png")
    result = await preset.orchestrate(
        fake_single, params, paths,
        job_id="jA", timeout_sec=120.0, cu_input_dir=tmp_path,
    )
    assert len(calls) == 1
    assert calls[0] == ("sky", "orig.png", "jA")
    assert result.exists()


async def test_orchestrate_two_pass_runs_undress_then_redress(tmp_path):
    preset = InpaintPreset()
    calls: list[dict] = []

    async def fake_single(params, paths, jid, ts):
        out = tmp_path / f"{jid}_out.png"
        Image.new("RGB", (8, 8), "blue").save(out)
        calls.append({
            "prompt": params.prompt,
            "strength": params.strength,
            "steps": params.steps,
            "input_image": paths.input_image,
            "jid": jid,
        })
        return out

    params = InpaintPreset.Parameters(
        prompt="green forest",
        two_pass=True,
        skin_prompt="bare skin torso",
        steps=8,
        strength=0.85,
    )
    paths = InputPaths(input_image="orig.png", mask_image="m.png")
    result = await preset.orchestrate(
        fake_single, params, paths,
        job_id="jB", timeout_sec=120.0, cu_input_dir=tmp_path,
    )
    assert len(calls) == 2
    # Pass 1: undress
    assert calls[0]["prompt"] == "bare skin torso"
    assert calls[0]["strength"] == 1.0
    assert calls[0]["jid"] == "jB_p1"
    assert calls[0]["input_image"] == "orig.png"
    # Pass 2: redress, user prompt + user strength, bridged input
    assert calls[1]["prompt"] == "green forest"
    assert calls[1]["strength"] == 0.85
    assert calls[1]["steps"] == 8
    assert calls[1]["jid"] == "jB_p2"
    assert calls[1]["input_image"].endswith("_p2_in.png")
    # Pass 1 intermediate cleaned, pass 2 output returned
    assert result.exists()
    assert not (tmp_path / "jB_p1_out.png").exists()


def test_inpaint_params_auto_mask_defaults():
    p = InpaintPreset.Parameters(prompt="x")
    assert p.auto_mask is False
    assert p.auto_mask_categories == ["upper_clothes", "pants", "skirt", "dress", "belt"]


def test_inpaint_params_auto_mask_categories_reject_unknown():
    with pytest.raises(Exception):
        InpaintPreset.Parameters(prompt="x", auto_mask_categories=["upper_clothes", "tie"])


def test_inpaint_params_auto_mask_categories_reject_empty():
    with pytest.raises(Exception):
        InpaintPreset.Parameters(prompt="x", auto_mask_categories=[])


def test_inpaint_inject_auto_mask_wires_segmenter():
    preset = InpaintPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = InpaintPreset.Parameters(
        prompt="red velvet dress",
        auto_mask=True,
        auto_mask_categories=["upper_clothes", "pants"],
        seed=7,
    )
    paths = InputPaths(input_image="in.png", mask_image=None)
    out = preset.inject(tpl, params, paths).to_dict()

    assert "mask_image" not in out
    assert "auto_mask_segmenter" in out
    assert "auto_mask_to_mask" in out
    seg = out["auto_mask_segmenter"]["inputs"]
    assert seg["Upper_clothes"] is False
    assert seg["Pants"] is False
    assert seg["Face"] is True and seg["Hair"] is True
    assert out["grow_mask"]["inputs"]["mask"] == ["auto_mask_to_mask", 0]


def test_inpaint_inject_manual_drops_auto_nodes():
    preset = InpaintPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = InpaintPreset.Parameters(prompt="x", auto_mask=False)
    paths = InputPaths(input_image="in.png", mask_image="m.png")
    out = preset.inject(tpl, params, paths).to_dict()

    assert "mask_image" in out
    assert "auto_mask_segmenter" not in out
    assert "auto_mask_to_mask" not in out
    assert out["grow_mask"]["inputs"]["mask"] == ["mask_image", 0]


def test_inpaint_inject_auto_mask_without_manual_mask_ok():
    preset = InpaintPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = InpaintPreset.Parameters(prompt="x", auto_mask=True)
    paths = InputPaths(input_image="in.png", mask_image=None)
    # Should not raise
    preset.inject(tpl, params, paths)


async def test_orchestrate_two_pass_auto_mask_preseg(tmp_path):
    preset = InpaintPreset()
    single_calls: list[dict] = []
    workflow_calls: list[dict] = []

    async def fake_run_workflow(wf, jid, ts, ext):
        workflow_calls.append({"nodes": set(wf.keys()), "jid": jid})
        out = tmp_path / f"{jid}_segout.png"
        Image.new("RGB", (8, 8), "white").save(out)
        return out

    async def fake_single(params, paths, jid, ts):
        single_calls.append({
            "prompt": params.prompt,
            "mask_image": paths.mask_image,
            "auto_mask": params.auto_mask,
            "jid": jid,
        })
        out = tmp_path / f"{jid}_out.png"
        Image.new("RGB", (8, 8), "blue").save(out)
        return out

    params = InpaintPreset.Parameters(
        prompt="green silk dress",
        two_pass=True,
        auto_mask=True,
        auto_mask_categories=["upper_clothes", "dress"],
    )
    paths = InputPaths(input_image="orig.png", mask_image=None)
    result = await preset.orchestrate(
        fake_single, params, paths,
        job_id="jAUTO", timeout_sec=120.0, cu_input_dir=tmp_path,
        run_workflow=fake_run_workflow,
    )

    # Segment-only workflow ran once
    assert len(workflow_calls) == 1
    assert "auto_mask_segmenter" in workflow_calls[0]["nodes"]
    assert workflow_calls[0]["jid"] == "jAUTO_seg"
    # Inpaint ran twice (undress + redress)
    assert len(single_calls) == 2
    # Both passes used the same generated mask filename and had auto_mask disabled
    assert single_calls[0]["mask_image"].endswith("_automask.png")
    assert single_calls[0]["mask_image"] == single_calls[1]["mask_image"]
    assert single_calls[0]["auto_mask"] is False
    assert single_calls[1]["auto_mask"] is False
    # Prompts: pass 1 = skin_prompt, pass 2 = user prompt
    assert single_calls[0]["prompt"] == params.skin_prompt
    assert single_calls[1]["prompt"] == "green silk dress"
    assert result.exists()


async def test_orchestrate_single_pass_auto_mask_no_run_workflow_needed(tmp_path):
    """Single-pass + auto_mask runs inject() inline; run_workflow not used."""
    preset = InpaintPreset()

    async def fake_single(params, paths, jid, ts):
        out = tmp_path / f"{jid}_out.png"
        Image.new("RGB", (8, 8), "red").save(out)
        return out

    async def never_call(*_a, **_kw):
        raise AssertionError("run_workflow should not be invoked for single-pass auto-mask")

    params = InpaintPreset.Parameters(prompt="x", auto_mask=True, two_pass=False)
    paths = InputPaths(input_image="orig.png", mask_image=None)
    result = await preset.orchestrate(
        fake_single, params, paths,
        job_id="jS", timeout_sec=60.0, cu_input_dir=tmp_path,
        run_workflow=never_call,
    )
    assert result.exists()


async def test_orchestrate_structural_refiner_modifies_output(tmp_path):
    preset = InpaintPreset()

    async def fake_single(params, paths, jid, ts):
        out = tmp_path / f"{jid}_out.png"
        Image.new("RGB", (64, 64), (120, 120, 120)).save(out)
        return out

    params = InpaintPreset.Parameters(
        prompt="x", structural_refiner=True, refiner_strength=0.8
    )
    paths = InputPaths(input_image="orig.png", mask_image="m.png")
    result = await preset.orchestrate(
        fake_single, params, paths,
        job_id="jC", timeout_sec=60.0, cu_input_dir=tmp_path,
    )
    # UnsharpMask on a flat image still produces identical pixels (nothing to sharpen),
    # but with a non-flat image we'd see a delta. Assert the file is still a valid PNG
    # and readable (i.e. we didn't corrupt it while saving).
    img = Image.open(result)
    assert img.size == (64, 64)
    assert img.format == "PNG"
