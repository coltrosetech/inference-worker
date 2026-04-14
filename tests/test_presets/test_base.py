import pytest

from worker.presets.base import Preset, Mode


class _FakePreset(Preset):
    name = "fake"
    mode = Mode.IMAGE
    template_filename = "fake.json"
    output_extension = "png"
    output_content_type = "image/png"

    class Parameters(Preset.BaseParameters):
        steps: int = 6
        cfg: float = 1.8

    def inject(self, template, params, input_paths):
        return template


def test_preset_name_is_class_attr():
    p = _FakePreset()
    assert p.name == "fake"
    assert p.mode is Mode.IMAGE


def test_parameters_apply_defaults():
    params = _FakePreset.Parameters()
    assert params.steps == 6
    assert params.cfg == 1.8
    assert params.seed is None


def test_parameters_reject_extra_fields():
    with pytest.raises(Exception):
        _FakePreset.Parameters(steps=1, cfg=1.0, unknown_field="x")


def test_parameters_validate_ranges():
    with pytest.raises(Exception):
        _FakePreset.Parameters(steps=-1)
    with pytest.raises(Exception):
        _FakePreset.Parameters(cfg=-0.1)


def test_mode_enum_values():
    assert Mode.IMAGE.value == "image"
    assert Mode.VIDEO.value == "video"
