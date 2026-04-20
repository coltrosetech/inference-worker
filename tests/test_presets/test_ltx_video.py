from pathlib import Path

import pytest

from worker.presets.base import InputPaths, Mode
from worker.presets.ltx_video import LtxVideoPreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_ltx_video_params_defaults():
    p = LtxVideoPreset.Parameters(prompt="a bird flies")
    assert p.num_frames == 97
    assert p.fps == 24
    assert p.cfg == 3.0
    assert p.width == 768
    assert p.height == 512


def test_ltx_video_params_require_prompt():
    with pytest.raises(Exception):
        LtxVideoPreset.Parameters()


def test_ltx_video_num_frames_is_literal_enum():
    for n in (25, 49, 97, 121):
        p = LtxVideoPreset.Parameters(prompt="x", num_frames=n)
        assert p.num_frames == n
    with pytest.raises(Exception):
        LtxVideoPreset.Parameters(prompt="x", num_frames=50)


def test_ltx_video_resolution_bounds():
    with pytest.raises(Exception):
        LtxVideoPreset.Parameters(prompt="x", width=1217)
    with pytest.raises(Exception):
        LtxVideoPreset.Parameters(prompt="x", height=705)
    p = LtxVideoPreset.Parameters(prompt="x", width=1216, height=704)
    assert p.width == 1216 and p.height == 704


def test_ltx_video_is_video_mode():
    assert LtxVideoPreset.mode is Mode.VIDEO
    assert LtxVideoPreset.output_extension == "mp4"
    assert LtxVideoPreset.output_content_type == "video/mp4"


def test_ltx_video_inject_sets_all_knobs():
    preset = LtxVideoPreset()
    tpl = preset.load_template(WORKFLOWS)
    assert tpl.is_api_format()
    params = LtxVideoPreset.Parameters(
        prompt="a cat in a spaceship",
        negative_prompt="blurry",
        num_frames=49,
        fps=30,
        cfg=5.0,
        steps=12,
        width=1024,
        height=576,
        seed=1234,
    )
    paths = InputPaths(input_image="frame0.png")
    out = preset.inject(tpl, params, paths).to_dict()

    assert out["input_image"]["inputs"]["image"] == "frame0.png"
    assert out["positive_prompt"]["inputs"]["text"] == "a cat in a spaceship"
    assert out["negative_prompt"]["inputs"]["text"] == "blurry"
    assert out["img2video"]["inputs"]["width"] == 1024
    assert out["img2video"]["inputs"]["height"] == 576
    assert out["img2video"]["inputs"]["length"] == 49
    assert out["conditioning"]["inputs"]["frame_rate"] == 30.0
    assert out["scheduler"]["inputs"]["steps"] == 12
    assert out["guider"]["inputs"]["cfg"] == 5.0
    assert out["noise"]["inputs"]["noise_seed"] == 1234
    assert out["save"]["inputs"]["frame_rate"] == 30


def test_ltx_video_inject_seed_random_when_none(monkeypatch):
    monkeypatch.setattr("worker.presets.ltx_video.random_seed", lambda: 5555)
    preset = LtxVideoPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = LtxVideoPreset.Parameters(prompt="x", seed=None)
    paths = InputPaths(input_image="a.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert out["noise"]["inputs"]["noise_seed"] == 5555


def test_ltx_video_workflow_uses_ltxv_clip_and_checkpoint_vae():
    preset = LtxVideoPreset()
    tpl = preset.load_template(WORKFLOWS)
    data = tpl.to_dict()
    assert data["clip_loader"]["inputs"]["type"] == "ltxv"
    assert data["clip_loader"]["inputs"]["clip_name"] == "t5xxl_fp8_e4m3fn.safetensors"
    assert data["img2video"]["inputs"]["vae"] == ["checkpoint", 2]
    assert data["vae_decode"]["inputs"]["vae"] == ["checkpoint", 2]
    assert data["save"]["class_type"] == "VHS_VideoCombine"
    assert data["save"]["inputs"]["format"] == "video/h264-mp4"


def test_ltx_video_warmup_params_use_small_resolution_and_frames():
    kw = LtxVideoPreset.warmup_params()
    assert kw["width"] <= 512
    assert kw["height"] <= 512
    assert kw["num_frames"] == 25


def test_ltx_video_has_extended_warmup_timeout():
    assert LtxVideoPreset.warmup_timeout_sec >= 300.0
