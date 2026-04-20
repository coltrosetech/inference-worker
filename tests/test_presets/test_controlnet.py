from pathlib import Path

import pytest

from worker.presets.base import InputPaths
from worker.presets.controlnet import ControlnetPreset, _CN_TYPE_MAP


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_controlnet_params_require_prompt():
    with pytest.raises(Exception):
        ControlnetPreset.Parameters()
    p = ControlnetPreset.Parameters(prompt="a portrait")
    assert p.controlnet_type == "canny"
    assert p.controlnet_strength == 0.8


def test_controlnet_params_reject_unknown_type():
    with pytest.raises(Exception):
        ControlnetPreset.Parameters(prompt="x", controlnet_type="hologram")


def test_controlnet_params_controlnet_strength_bounds():
    with pytest.raises(Exception):
        ControlnetPreset.Parameters(prompt="x", controlnet_strength=-0.1)
    with pytest.raises(Exception):
        ControlnetPreset.Parameters(prompt="x", controlnet_strength=2.1)


@pytest.mark.parametrize("cn_type,expected_preproc", [
    ("canny",    "CannyEdgePreprocessor"),
    ("depth",    "DepthAnythingV2Preprocessor"),
    ("pose",     "DWPreprocessor"),
    ("lineart",  "AnyLineArtPreprocessor_aux"),
    ("scribble", "ScribblePreprocessor"),
])
def test_controlnet_inject_maps_type_to_preprocessor(cn_type, expected_preproc):
    preset = ControlnetPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = ControlnetPreset.Parameters(prompt="x", controlnet_type=cn_type)
    paths = InputPaths(input_image="in.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert out["preprocessor"]["inputs"]["preprocessor"] == expected_preproc
    expected_union = _CN_TYPE_MAP[cn_type][1]
    assert out["cn_union_type"]["inputs"]["type"] == expected_union


def test_controlnet_inject_sets_strength_and_sampler():
    preset = ControlnetPreset()
    tpl = preset.load_template(WORKFLOWS)
    assert tpl.is_api_format()
    params = ControlnetPreset.Parameters(
        prompt="rainbow sky",
        negative_prompt="gray",
        controlnet_type="depth",
        controlnet_strength=1.2,
        strength=0.6,
        steps=8,
        cfg=2.0,
        seed=11,
    )
    paths = InputPaths(input_image="in.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert out["input_image"]["inputs"]["image"] == "in.png"
    assert out["positive_prompt"]["inputs"]["text"] == "rainbow sky"
    assert out["negative_prompt"]["inputs"]["text"] == "gray"
    assert out["cn_apply"]["inputs"]["strength"] == 1.2
    sampler = out["sampler"]["inputs"]
    assert sampler["seed"] == 11
    assert sampler["steps"] == 8
    assert sampler["cfg"] == 2.0
    assert sampler["denoise"] == 0.6


def test_controlnet_workflow_wires_preprocessor_through_cn_to_sampler():
    preset = ControlnetPreset()
    tpl = preset.load_template(WORKFLOWS)
    data = tpl.to_dict()
    assert data["cn_loader"]["class_type"] == "ControlNetLoader"
    assert data["cn_union_type"]["class_type"] == "SetUnionControlNetType"
    assert data["cn_union_type"]["inputs"]["control_net"] == ["cn_loader", 0]
    assert data["preprocessor"]["class_type"] == "AIO_Preprocessor"
    assert data["preprocessor"]["inputs"]["image"] == ["input_image", 0]
    assert data["cn_apply"]["class_type"] == "ControlNetApplyAdvanced"
    assert data["cn_apply"]["inputs"]["control_net"] == ["cn_union_type", 0]
    assert data["cn_apply"]["inputs"]["image"] == ["preprocessor", 0]
    assert data["sampler"]["inputs"]["positive"] == ["cn_apply", 0]
    assert data["sampler"]["inputs"]["negative"] == ["cn_apply", 1]


def test_controlnet_inject_seed_random_when_none(monkeypatch):
    monkeypatch.setattr("worker.presets.controlnet.random_seed", lambda: 321)
    preset = ControlnetPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = ControlnetPreset.Parameters(prompt="x", seed=None)
    paths = InputPaths(input_image="a.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert out["sampler"]["inputs"]["seed"] == 321


def test_controlnet_output_extension_and_content_type():
    assert ControlnetPreset.output_extension == "png"
    assert ControlnetPreset.output_content_type == "image/png"
