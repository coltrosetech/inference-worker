from pathlib import Path

import pytest

from worker.presets.base import InputPaths, Mode
from worker.presets.wan_i2v import DEFAULT_NEGATIVE, WanI2vPreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_defaults():
    p = WanI2vPreset.Parameters(prompt="x")
    assert p.steps == 4
    assert p.cfg == 1.0
    assert p.shift == 5.0
    assert p.length == 81
    assert p.fps == 16
    assert (p.width, p.height) == (720, 1280)


def test_mode_and_output():
    p = WanI2vPreset()
    assert p.mode is Mode.VIDEO
    assert p.output_extension == "mp4"
    assert p.output_content_type == "video/mp4"


def test_length_must_be_4n_plus_1():
    with pytest.raises(ValueError, match="4n"):
        WanI2vPreset.Parameters(prompt="x", length=80)
    WanI2vPreset.Parameters(prompt="x", length=81)  # ok


def test_dims_must_be_div16():
    with pytest.raises(ValueError, match="divisible by 16"):
        WanI2vPreset.Parameters(prompt="x", width=700)
    WanI2vPreset.Parameters(prompt="x", width=720)  # ok


def test_inject_wires_moe_split_and_shift():
    preset = WanI2vPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = WanI2vPreset.Parameters(prompt="she turns gently", steps=4, shift=6.0, seed=7)
    out = preset.inject(tpl, params, InputPaths(input_image="in.png")).to_dict()

    assert out["input_image"]["inputs"]["image"] == "in.png"
    assert out["positive_prompt"]["inputs"]["text"] == "she turns gently"
    # MoE handoff at steps//2: high runs [0,2), low runs [2,4)
    assert out["sampler_high"]["inputs"]["start_at_step"] == 0
    assert out["sampler_high"]["inputs"]["end_at_step"] == 2
    assert out["sampler_low"]["inputs"]["start_at_step"] == 2
    assert out["sampler_low"]["inputs"]["end_at_step"] == 4
    # both experts share the seed and the shift
    assert out["sampler_high"]["inputs"]["noise_seed"] == 7
    assert out["sampler_low"]["inputs"]["noise_seed"] == 7
    assert out["model_sampling_high"]["inputs"]["shift"] == 6.0
    assert out["model_sampling_low"]["inputs"]["shift"] == 6.0
    # low expert consumes the high expert's leftover-noise latent
    assert out["sampler_low"]["inputs"]["latent_image"] == ["sampler_high", 0]


def test_inject_bakes_quality_negative():
    preset = WanI2vPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = WanI2vPreset.Parameters(prompt="x", negative_prompt="ugly")
    out = preset.inject(tpl, params, InputPaths(input_image="in.png")).to_dict()
    neg = out["negative_prompt"]["inputs"]["text"]
    assert neg.startswith(DEFAULT_NEGATIVE)
    assert "ugly" in neg
