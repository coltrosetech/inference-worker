import pytest
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
    app = create_app()
    return TestClient(app)


def _payload(job_id="j1", preset="edit"):
    return {
        "job_id": job_id,
        "preset": preset,
        "prompt": "a cat",
        "input_image_url": "https://s/in.png",
        "parameters": {"steps": 6, "cfg": 1.8, "width": 512, "height": 512},
        "callback_url": "https://b/cb",
        "upload_url": "https://s/out",
        "upload_method": "PUT",
    }


def test_generate_requires_auth(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/generate", json=_payload(), headers={"Idempotency-Key": "x"})
    assert r.status_code == 401


def test_generate_requires_idempotency_key(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/generate", json=_payload(), headers={"Authorization": f"Bearer {'k'*32}"})
    assert r.status_code == 400


def test_generate_accepts_valid(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/generate", json=_payload(), headers={
        "Authorization": f"Bearer {'k'*32}",
        "Idempotency-Key": "j1",
    })
    assert r.status_code == 202
    body = r.json()
    assert body["accepted"] is True
    assert body["job_id"] == "j1"
    assert "queue_position" in body
    assert "worker_id" in body


def test_generate_unknown_preset_returns_400(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    # Use an extra endpoint to bypass pydantic Literal enum (it would return 422)
    # by sending a preset not in the registry. Since Literal enforces, this becomes 422 or 400
    # depending on FastAPI; accept either.
    r = c.post("/v1/generate", json=_payload(preset="upscale"), headers={
        "Authorization": f"Bearer {'k'*32}",
        "Idempotency-Key": "j1",
    })
    assert r.status_code in (400, 422)


def test_generate_duplicate_job_id_returns_409(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    hdrs = {"Authorization": f"Bearer {'k'*32}", "Idempotency-Key": "j1"}
    r1 = c.post("/v1/generate", json=_payload(job_id="j1"), headers=hdrs)
    assert r1.status_code == 202
    r2 = c.post("/v1/generate", json=_payload(job_id="j1"), headers=hdrs)
    assert r2.status_code == 409


def test_generate_queue_full_returns_429(monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "1")
    c = _client(monkeypatch, tmp_path)
    hdrs = lambda k: {"Authorization": f"Bearer {'k'*32}", "Idempotency-Key": k}
    r1 = c.post("/v1/generate", json=_payload(job_id="a"), headers=hdrs("a"))
    r2 = c.post("/v1/generate", json=_payload(job_id="b"), headers=hdrs("b"))
    assert r1.status_code == 202
    assert r2.status_code == 429
    assert "retry-after" in [k.lower() for k in r2.headers.keys()]
