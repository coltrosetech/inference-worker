from fastapi.testclient import TestClient


def _client(monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", "/tmp")
    monkeypatch.setenv("MODELS_PATH", "/tmp")
    monkeypatch.setenv("CACHE_PATH", "/tmp")
    monkeypatch.setenv("WORKFLOWS_PATH", "/tmp")
    from worker.core.config import get_settings
    get_settings.cache_clear()
    from worker.main import create_app
    return TestClient(create_app())


def test_health_basic_shape(monkeypatch):
    client = _client(monkeypatch)
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["ready"] is False
    assert "version" in body
    assert "uptime_sec" in body
    assert body["queue_depth"] == 0
    assert "worker_id" in body


def test_health_marks_ready(monkeypatch):
    client = _client(monkeypatch)
    from worker.state import worker_state
    worker_state.mark_ready()
    r = client.get("/v1/health")
    assert r.json()["ready"] is True
    worker_state.mark_not_ready()
