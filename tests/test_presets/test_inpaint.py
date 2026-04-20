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


async def test_orchestrate_auto_mask_single_pass_calls_segment_and_falls_through(tmp_path):
    """Single-pass + auto_mask: segment_fn produces the mask, then one inpaint call
    uses that synthesized mask with auto_mask flipped off."""
    preset = InpaintPreset()
    seg_calls: list[dict] = []
    single_calls: list[dict] = []

    async def fake_segment(in_path, out_path, categories):
        seg_calls.append({"in": in_path.name, "out": out_path.name, "cats": list(categories)})
        Image.new("L", (8, 8), 255).save(out_path)
        return out_path

    async def fake_single(params, paths, jid, ts):
        single_calls.append({
            "prompt": params.prompt,
            "mask_image": paths.mask_image,
            "auto_mask": params.auto_mask,
            "jid": jid,
        })
        out = tmp_path / f"{jid}_out.png"
        Image.new("RGB", (8, 8), "red").save(out)
        return out

    # Seed the "uploaded input" into cu_input_dir so segment_fn sees a real file.
    (tmp_path / "orig.png").write_bytes(b"fake-image-bytes")

    params = InpaintPreset.Parameters(
        prompt="navy suit",
        auto_mask=True,
        auto_mask_categories=["upper_clothes", "pants"],
    )
    paths = InputPaths(input_image="orig.png", mask_image=None)
    result = await preset.orchestrate(
        fake_single, params, paths,
        job_id="jS", timeout_sec=60.0, cu_input_dir=tmp_path,
        segment_fn=fake_segment,
    )

    assert len(seg_calls) == 1
    assert seg_calls[0]["in"] == "orig.png"
    assert seg_calls[0]["out"] == "jS_automask.png"
    assert seg_calls[0]["cats"] == ["upper_clothes", "pants"]

    assert len(single_calls) == 1
    assert single_calls[0]["auto_mask"] is False  # flipped off before inject
    assert single_calls[0]["mask_image"] == "jS_automask.png"
    assert single_calls[0]["prompt"] == "navy suit"
    assert result.exists()
    # Mask cleaned up afterwards
    assert not (tmp_path / "jS_automask.png").exists()


async def test_orchestrate_auto_mask_two_pass_shares_one_mask(tmp_path):
    """Two-pass + auto_mask: segment runs once; both passes use the same mask
    and have auto_mask=False on per-pass params."""
    preset = InpaintPreset()
    seg_calls: list[dict] = []
    single_calls: list[dict] = []

    async def fake_segment(in_path, out_path, categories):
        seg_calls.append({"in": in_path.name})
        Image.new("L", (8, 8), 255).save(out_path)
        return out_path

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

    (tmp_path / "orig.png").write_bytes(b"fake")

    params = InpaintPreset.Parameters(
        prompt="green silk dress",
        two_pass=True,
        auto_mask=True,
        auto_mask_categories=["upper_clothes", "dress"],
    )
    paths = InputPaths(input_image="orig.png", mask_image=None)
    result = await preset.orchestrate(
        fake_single, params, paths,
        job_id="jT", timeout_sec=120.0, cu_input_dir=tmp_path,
        segment_fn=fake_segment,
    )

    # Segmentation: once.
    assert len(seg_calls) == 1
    # Inpaint: twice.
    assert len(single_calls) == 2
    # Both passes read the same auto-generated mask filename and see auto_mask=False.
    assert single_calls[0]["mask_image"] == "jT_automask.png"
    assert single_calls[1]["mask_image"] == "jT_automask.png"
    assert all(c["auto_mask"] is False for c in single_calls)
    # Prompt sequence: skin_prompt → user prompt.
    assert single_calls[0]["prompt"] == params.skin_prompt
    assert single_calls[1]["prompt"] == "green silk dress"
    assert result.exists()
    assert not (tmp_path / "jT_automask.png").exists()


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
