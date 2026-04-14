from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from worker.core.auth import require_bearer


def _app(monkeypatch, key="secret1234567890"):
    monkeypatch.setenv("WORKER_API_KEY", key)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", "/tmp")
    monkeypatch.setenv("MODELS_PATH", "/tmp")
    monkeypatch.setenv("CACHE_PATH", "/tmp")
    monkeypatch.setenv("WORKFLOWS_PATH", "/tmp")
    from worker.core.config import get_settings
    get_settings.cache_clear()
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_bearer)])
    def protected():
        return {"ok": True}
    return TestClient(app)


def test_missing_header_returns_401(monkeypatch):
    client = _app(monkeypatch)
    r = client.get("/protected")
    assert r.status_code == 401


def test_wrong_scheme_returns_401(monkeypatch):
    client = _app(monkeypatch)
    r = client.get("/protected", headers={"Authorization": "Basic abc"})
    assert r.status_code == 401


def test_wrong_token_returns_401(monkeypatch):
    client = _app(monkeypatch, key="correctkey123456")
    r = client.get("/protected", headers={"Authorization": "Bearer wrongkey123456"})
    assert r.status_code == 401


def test_correct_token_passes(monkeypatch):
    client = _app(monkeypatch, key="correctkey123456")
    r = client.get("/protected", headers={"Authorization": "Bearer correctkey123456"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
