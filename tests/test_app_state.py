from pathlib import Path

import pytest

from worker.app_state import AppState


def test_app_state_exposes_components(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))
    from worker.core.config import get_settings
    get_settings.cache_clear()

    state = AppState.build()
    assert state.queue is not None
    assert state.pools is not None
    assert state.model_manager is not None
    assert state.comfyui is not None
    assert state.http is not None
    assert state.executor is not None
    assert state.runner is not None
