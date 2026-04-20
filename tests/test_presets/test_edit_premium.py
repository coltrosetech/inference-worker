from pathlib import Path

import pytest

from worker.presets.base import InputPaths, Mode
from worker.presets.edit_premium import EditPremiumPreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_edit_premium_params_defaults():
    p = EditPremiumPreset.Parameters(prompt="make it sunset")
    assert p.guidance == 2.5
    assert p.steps == 20
    assert p.cfg == 1.0


def test_edit_premium_params_require_prompt():
    with pytest.raises(Exception):
        EditPremiumPreset.Parameters()


def test_edit_premium_params_guidance_bounds():
    with pytest.raises(Exception):
        EditPremiumPreset.Parameters(prompt="x", guidance=-0.1)
    with pytest.raises(Exception):
        EditPremiumPreset.Parameters(prompt="x", guidance=10.1)


def test_edit_premium_is_image_premium_mode():
    assert EditPremiumPreset.mode is Mode.IMAGE_PREMIUM


def test_edit_premium_warmup_timeout_elevated():
    assert EditPremiumPreset.warmup_timeout_sec >= 300.0


def test_edit_premium_inject_sets_prompts_guidance_and_sampler():
    preset = EditPremiumPreset()
    tpl = preset.load_template(WORKFLOWS)
    assert tpl.is_api_format()
    params = EditPremiumPreset.Parameters(
        prompt="change the sky to golden hour",
        negative_prompt="overexposed",
        guidance=4.0,
        steps=28,
        cfg=1.0,
        seed=123,
    )
    paths = InputPaths(input_image="in.png")
    out = preset.inject(tpl, params, paths).to_dict()

    assert out["input_image"]["inputs"]["image"] == "in.png"
    assert out["positive_prompt"]["inputs"]["text"] == "change the sky to golden hour"
    assert out["negative_prompt"]["inputs"]["text"] == "overexposed"
    assert out["flux_guidance"]["inputs"]["guidance"] == 4.0
    sampler = out["sampler"]["inputs"]
    assert sampler["seed"] == 123
    assert sampler["steps"] == 28
    assert sampler["cfg"] == 1.0


def test_edit_premium_inject_seed_random_when_none(monkeypatch):
    monkeypatch.setattr("worker.presets.edit_premium.random_seed", lambda: 888)
    preset = EditPremiumPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = EditPremiumPreset.Parameters(prompt="x", seed=None)
    paths = InputPaths(input_image="a.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert out["sampler"]["inputs"]["seed"] == 888


def test_edit_premium_workflow_wires_flux_kontext_pipeline():
    """UNETLoader → model; DualCLIPLoader → clip; reference_latent from input VAEEncode."""
    preset = EditPremiumPreset()
    tpl = preset.load_template(WORKFLOWS)
    data = tpl.to_dict()
    assert data["unet_loader"]["class_type"] == "UNETLoader"
    assert data["unet_loader"]["inputs"]["unet_name"] == "flux_kontext_dev_fp8_scaled.safetensors"
    assert data["clip_loader"]["class_type"] == "DualCLIPLoader"
    assert data["clip_loader"]["inputs"]["type"] == "flux"
    assert data["vae_loader"]["inputs"]["vae_name"] == "flux_ae.safetensors"
    assert data["scale_input"]["class_type"] == "FluxKontextImageScale"
    assert data["reference_cond"]["class_type"] == "ReferenceLatent"
    assert data["reference_cond"]["inputs"]["latent"] == ["vae_encode", 0]
    assert data["flux_guidance"]["class_type"] == "FluxGuidance"
    assert data["sampler"]["inputs"]["positive"] == ["flux_guidance", 0]
    assert data["sampler"]["inputs"]["model"] == ["unet_loader", 0]


def test_edit_premium_output_extension_and_content_type():
    assert EditPremiumPreset.output_extension == "png"
    assert EditPremiumPreset.output_content_type == "image/png"
