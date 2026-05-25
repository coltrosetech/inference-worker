import pytest

from worker.presets import PRESETS, get_preset
from worker.presets.base import Mode


def test_registry_presets():
    assert set(PRESETS.keys()) == {"tryon", "ltx_video", "wan_flf2v"}


def test_get_preset_returns_instance():
    p = get_preset("tryon")
    assert p.name == "tryon"
    assert p.mode is Mode.IMAGE


def test_ltx_video_is_video_mode():
    p = get_preset("ltx_video")
    assert p.name == "ltx_video"
    assert p.mode is Mode.VIDEO
    assert p.output_extension == "mp4"


def test_get_preset_unknown_raises():
    with pytest.raises(KeyError):
        get_preset("nonexistent")
