from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))
    from worker.core.config import get_settings
    get_settings.cache_clear()
    from worker.main import create_app
    return TestClient(create_app())


def test_health_includes_gpu_and_comfyui_fields(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    body = c.get("/v1/health").json()
    for key in ("gpu", "vram_used_gb", "vram_total_gb", "comfyui_alive", "queue_depth"):
        assert key in body
    # comfyui_alive is False because no real server
    assert body["comfyui_alive"] is False
