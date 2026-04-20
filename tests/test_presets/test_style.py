from pathlib import Path

import pytest

from worker.presets.base import InputPaths
from worker.presets.style import StylePreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_style_params_require_prompt_and_reference():
    with pytest.raises(Exception):
        StylePreset.Parameters()
    p = StylePreset.Parameters(prompt="oil painting of a river")
    assert p.style_strength == 0.7


def test_style_inject_sets_prompts_and_reference():
    preset = StylePreset()
    tpl = preset.load_template(WORKFLOWS)
    assert tpl.is_api_format()
    params = StylePreset.Parameters(
        prompt="oil painting of a river",
        negative_prompt="flat",
        style_strength=0.85,
        steps=8, cfg=2.0, seed=7, width=768, height=768,
        strength=0.5,
    )
    paths = InputPaths(input_image="in.png", reference_image="ref.png")
    out = preset.inject(tpl, params, paths).to_dict()

    assert out["positive_prompt"]["inputs"]["text"] == "oil painting of a river"
    assert out["negative_prompt"]["inputs"]["text"] == "flat"
    assert out["input_image"]["inputs"]["image"] == "in.png"
    assert out["reference_image"]["inputs"]["image"] == "ref.png"
    assert out["ipa_apply"]["inputs"]["weight"] == 0.85
    sampler = out["sampler"]["inputs"]
    assert sampler["seed"] == 7
    assert sampler["steps"] == 8
    assert sampler["cfg"] == 2.0
    assert sampler["denoise"] == 0.5


def test_style_inject_rejects_missing_reference():
    preset = StylePreset()
    tpl = preset.load_template(WORKFLOWS)
    params = StylePreset.Parameters(prompt="x")
    paths = InputPaths(input_image="in.png", reference_image=None)
    with pytest.raises(ValueError):
        preset.inject(tpl, params, paths)


def test_style_inject_seed_random_when_none(monkeypatch):
    monkeypatch.setattr("worker.presets.style.random_seed", lambda: 42)
    preset = StylePreset()
    tpl = preset.load_template(WORKFLOWS)
    params = StylePreset.Parameters(prompt="x", seed=None)
    paths = InputPaths(input_image="a.png", reference_image="b.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert out["sampler"]["inputs"]["seed"] == 42


def test_style_workflow_wires_ipa_through_model_chain():
    """The sampler must receive the IP-Adapter-patched MODEL, not the bare checkpoint."""
    preset = StylePreset()
    tpl = preset.load_template(WORKFLOWS)
    data = tpl.to_dict()
    assert data["ipa_apply"]["class_type"] == "IPAdapterAdvanced"
    assert data["ipa_apply"]["inputs"]["model"] == ["checkpoint", 0]
    assert data["ipa_apply"]["inputs"]["ipadapter"] == ["ipa_loader", 0]
    assert data["ipa_apply"]["inputs"]["image"] == ["reference_image", 0]
    assert data["ipa_apply"]["inputs"]["clip_vision"] == ["clip_vision_loader", 0]
    assert data["sampler"]["inputs"]["model"] == ["ipa_apply", 0]
