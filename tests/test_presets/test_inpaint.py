from pathlib import Path

import pytest

from worker.presets.base import InputPaths
from worker.presets.inpaint import InpaintPreset


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
