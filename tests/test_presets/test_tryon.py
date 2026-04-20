from pathlib import Path

import pytest
from PIL import Image

from worker.presets.base import InputPaths
from worker.presets.tryon import TryonPreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_defaults():
    p = TryonPreset.Parameters()
    assert p.steps == 25
    assert p.cfg == 7.0
    assert p.reference_weight == 0.9
    assert p.auto_mask is True  # tryon defaults ON — auto-mask is the usual path


def test_inject_requires_reference_image():
    preset = TryonPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = TryonPreset.Parameters()
    with pytest.raises(ValueError, match="reference_image"):
        preset.inject(tpl, params, InputPaths(input_image="in.png", mask_image="m.png", reference_image=None))


def test_inject_requires_mask():
    preset = TryonPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = TryonPreset.Parameters()
    with pytest.raises(ValueError, match="mask_image"):
        preset.inject(tpl, params, InputPaths(input_image="in.png", mask_image=None, reference_image="r.png"))


def test_inject_wires_ip_adapter_and_inpaint():
    preset = TryonPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = TryonPreset.Parameters(prompt="white shirt", reference_weight=0.85, seed=7)
    paths = InputPaths(input_image="in.png", mask_image="m.png", reference_image="r.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert out["input_image"]["inputs"]["image"] == "in.png"
    assert out["reference_image"]["inputs"]["image"] == "r.png"
    assert out["mask_image"]["inputs"]["image"] == "m.png"
    assert out["ipadapter_apply"]["inputs"]["weight"] == 0.85
    assert out["sampler"]["inputs"]["seed"] == 7
    # Sampler consumes IP-Adapter-conditioned model
    assert out["sampler"]["inputs"]["model"] == ["ipadapter_apply", 0]


async def test_orchestrate_auto_mask_uses_segment_fn(tmp_path):
    preset = TryonPreset()
    seg_calls: list = []
    single_calls: list = []

    async def fake_segment(in_path, out_path, categories):
        seg_calls.append(list(categories))
        Image.new("L", (8, 8), 255).save(out_path)
        return out_path

    async def fake_single(params, paths, jid, ts):
        single_calls.append({"auto": params.auto_mask, "mask": paths.mask_image, "ref": paths.reference_image})
        out = tmp_path / f"{jid}_out.png"
        Image.new("RGB", (8, 8), "tan").save(out)
        return out

    (tmp_path / "orig.png").write_bytes(b"fake")
    params = TryonPreset.Parameters(
        prompt="wearing the reference garment", auto_mask=True,
        auto_mask_categories=["upper_clothes"],
    )
    paths = InputPaths(input_image="orig.png", mask_image=None, reference_image="ref.png")
    result = await preset.orchestrate(
        fake_single, params, paths,
        job_id="jT", timeout_sec=60.0, cu_input_dir=tmp_path,
        segment_fn=fake_segment,
    )
    assert seg_calls == [["upper_clothes"]]
    assert len(single_calls) == 1
    assert single_calls[0]["auto"] is False
    assert single_calls[0]["mask"] == "jT_automask.png"
    assert single_calls[0]["ref"] == "ref.png"  # reference preserved through
    assert result.exists()
    assert not (tmp_path / "jT_automask.png").exists()
