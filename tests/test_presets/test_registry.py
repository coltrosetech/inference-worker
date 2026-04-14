import pytest

from worker.presets import PRESETS, get_preset
from worker.presets.base import Mode


def test_registry_has_edit_and_style():
    assert set(PRESETS.keys()) >= {"edit", "style"}


def test_get_preset_returns_instance():
    p = get_preset("edit")
    assert p.name == "edit"
    assert p.mode is Mode.IMAGE


def test_get_preset_unknown_raises():
    with pytest.raises(KeyError):
        get_preset("nonexistent")
