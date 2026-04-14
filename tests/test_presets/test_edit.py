from pathlib import Path

import pytest

from worker.comfyui.workflow import find_node
from worker.presets.base import InputPaths
from worker.presets.edit import EditPreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_edit_params_defaults():
    p = EditPreset.Parameters(prompt="a cat")
    assert p.prompt == "a cat"
    assert p.negative_prompt == ""
    assert p.strength == 0.7
    assert p.steps == 6
    assert p.cfg == 1.8


def test_edit_params_require_prompt():
    with pytest.raises(Exception):
        EditPreset.Parameters()


def test_edit_inject_patches_prompt_seed_steps_strength_dims():
    preset = EditPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = EditPreset.Parameters(
        prompt="a dog on a skateboard",
        negative_prompt="bad anatomy",
        steps=10, cfg=2.0, strength=0.6, width=768, height=768, seed=42,
    )
    paths = InputPaths(input_image="job1_input.png")
    out = preset.inject(tpl, params, paths).to_dict()

    assert find_node(out, "positive_prompt")["widgets_values"][0] == "a dog on a skateboard"
    assert find_node(out, "negative_prompt")["widgets_values"][0] == "bad anatomy"

    sampler = find_node(out, "sampler")["widgets_values"]
    assert sampler[0] == 42
    assert sampler[1] == "fixed"
    assert sampler[2] == 10
    assert sampler[3] == 2.0
    assert sampler[6] == 0.6

    assert find_node(out, "input_image")["widgets_values"][0] == "job1_input.png"


def test_edit_seed_random_when_none(monkeypatch):
    monkeypatch.setattr("worker.presets.edit.random_seed", lambda: 99999)
    preset = EditPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = EditPreset.Parameters(prompt="x", seed=None)
    paths = InputPaths(input_image="a.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert find_node(out, "sampler")["widgets_values"][0] == 99999


def test_edit_output_extension_and_content_type():
    assert EditPreset.output_extension == "png"
    assert EditPreset.output_content_type == "image/png"
