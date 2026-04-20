"""Smoke tests for the SDXL inpaint fine-tune variants. Both inherit the
auto-mask + two-pass behaviour from InpaintPreset; we just verify the
overrides (name, template, defaults) and that inject() still wires the
same node graph."""
from pathlib import Path

import pytest

from worker.presets.base import InputPaths
from worker.presets.inpaint_realvis import InpaintRealvisPreset
from worker.presets.inpaint_sdxl import InpaintSdxlPreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


@pytest.mark.parametrize(
    "cls,expected_name,expected_steps,expected_cfg",
    [
        (InpaintSdxlPreset, "inpaint_sdxl", 25, 7.0),
        (InpaintRealvisPreset, "inpaint_realvis", 30, 6.5),
    ],
)
def test_defaults(cls, expected_name, expected_steps, expected_cfg):
    assert cls.name == expected_name
    p = cls.Parameters(prompt="x")
    assert p.steps == expected_steps
    assert p.cfg == expected_cfg
    # Inherited fields still work
    assert p.grow_mask_px == 8
    assert p.auto_mask is False


@pytest.mark.parametrize("cls", [InpaintSdxlPreset, InpaintRealvisPreset])
def test_inject_wires_graph(cls):
    preset = cls()
    tpl = preset.load_template(WORKFLOWS)
    params = cls.Parameters(prompt="sky", seed=7)
    paths = InputPaths(input_image="in.png", mask_image="m.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert out["input_image"]["inputs"]["image"] == "in.png"
    assert out["mask_image"]["inputs"]["image"] == "m.png"
    assert out["sampler"]["inputs"]["seed"] == 7
    # auto_mask off → no leftover auto-mask nodes in these SDXL templates
    # (they never had them; this just sanity-checks template load)
    assert out["grow_mask"]["inputs"]["mask"] == ["mask_image", 0]


@pytest.mark.parametrize("cls", [InpaintSdxlPreset, InpaintRealvisPreset])
def test_inject_requires_mask(cls):
    preset = cls()
    tpl = preset.load_template(WORKFLOWS)
    params = cls.Parameters(prompt="x")
    with pytest.raises(ValueError):
        preset.inject(tpl, params, InputPaths(input_image="in.png", mask_image=None))
