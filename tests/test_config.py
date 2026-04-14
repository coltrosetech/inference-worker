import os
from pathlib import Path

import pytest

from worker.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolate_cwd(monkeypatch, tmp_path):
    """Prevent pydantic-settings from picking up a stray .env in the repo root."""
    monkeypatch.chdir(tmp_path)


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))

    s = Settings()
    assert s.worker_port == 8000
    assert s.comfyui_internal_port == 8188
    assert s.max_queue_depth == 8
    assert s.job_timeout_sec_default == 300
    assert s.gpu_device == 0
    assert s.log_level == "info"
    assert s.log_format == "json"
    assert s.metrics_port == 9090
    assert len(s.worker_api_key) == 32


def test_settings_requires_secrets(monkeypatch, tmp_path):
    monkeypatch.delenv("WORKER_API_KEY", raising=False)
    monkeypatch.delenv("CALLBACK_HMAC_SECRET", raising=False)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))
    with pytest.raises(Exception) as exc:
        Settings()
    assert "worker_api_key" in str(exc.value).lower() or "hmac" in str(exc.value).lower()


def test_settings_worker_id_auto(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("WORKER_ID", "wk-auto")
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))
    s = Settings()
    assert s.resolved_worker_id().startswith("wk-")
    assert s.resolved_worker_id() != "wk-auto"


def test_get_settings_is_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
