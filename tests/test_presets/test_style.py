from pathlib import Path

import pytest

from worker.comfyui.workflow import find_node
from worker.presets.base import InputPaths
from worker.presets.style import StylePreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_style_params_require_prompt_and_reference():
    with pytest.raises(Exception):
        StylePreset.Parameters()
    p = StylePreset.Parameters(prompt="oil painting of a river")
    assert p.style_strength == 0.7


def test_style_inject_sets_prompts_and_reference(tmp_path):
    preset = StylePreset()
    tpl = preset.load_template(WORKFLOWS)
    params = StylePreset.Parameters(
        prompt="oil painting of a river",
        negative_prompt="flat",
        style_strength=0.85,
        steps=8, cfg=2.0, seed=7, width=768, height=768,
    )
    paths = InputPaths(input_image="in.png", reference_image="ref.png")
    out = preset.inject(tpl, params, paths).to_dict()

    assert find_node(out, "positive_prompt")["widgets_values"][0] == "oil painting of a river"
    assert find_node(out, "negative_prompt")["widgets_values"][0] == "flat"
    assert find_node(out, "input_image")["widgets_values"][0] == "in.png"
    assert find_node(out, "reference_image")["widgets_values"][0] == "ref.png"
    assert find_node(out, "ipa_apply")["widgets_values"][0] == 0.85
    sampler = find_node(out, "sampler")["widgets_values"]
    assert sampler[0] == 7
    assert sampler[2] == 8
    assert sampler[3] == 2.0


def test_style_inject_rejects_missing_reference():
    preset = StylePreset()
    tpl = preset.load_template(WORKFLOWS)
    params = StylePreset.Parameters(prompt="x")
    paths = InputPaths(input_image="in.png", reference_image=None)
    with pytest.raises(ValueError):
        preset.inject(tpl, params, paths)
