from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    (tmp_path / "workflows").mkdir(exist_ok=True)
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path / "workflows"))
    from worker.core.config import get_settings
    get_settings.cache_clear()
    from worker.main import create_app
    return TestClient(create_app())


def test_cancel_queued_job(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    hdrs = {"Authorization": f"Bearer {'k'*32}", "Idempotency-Key": "j"}
    payload = {
        "job_id": "j", "preset": "tryon", "prompt": "x",
        "input_image_url": "https://x/i.png", "reference_image_url": "https://x/g.png",
        "parameters": {"auto_mask": True},
        "callback_url": "https://b/cb",
        "upload_url": "https://s/o", "upload_method": "PUT",
    }
    assert c.post("/v1/generate", json=payload, headers=hdrs).status_code == 202
    r = c.post("/v1/cancel", json={"job_id": "j"}, headers={"Authorization": f"Bearer {'k'*32}"})
    assert r.status_code == 200
    assert r.json()["cancelled"] is True


def test_cancel_unknown_returns_404(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/cancel", json={"job_id": "nope"}, headers={"Authorization": f"Bearer {'k'*32}"})
    assert r.status_code == 404


def test_cancel_requires_auth(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/cancel", json={"job_id": "j"})
    assert r.status_code == 401
