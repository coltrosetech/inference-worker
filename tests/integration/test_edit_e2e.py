import time
import uuid
from pathlib import Path

import httpx
import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _upload_input(mock_url: str, payload: bytes, name: str) -> str:
    r = httpx.put(f"{mock_url}/storage/in/{name}", content=payload, timeout=30)
    r.raise_for_status()
    return f"{mock_url}/storage/in/{name}"


def test_edit_preset_end_to_end(wait_for_worker_ready, worker_base_url, worker_api_key, mock_backend_url):
    img_bytes = Path("tests/fixtures/input_512.png").read_bytes()
    name = f"in_{uuid.uuid4().hex}.png"
    input_url = _upload_input(mock_backend_url, img_bytes, name)
    job_id = f"j_{uuid.uuid4().hex[:12]}"

    out_name = f"out_{job_id}.png"
    resp = httpx.post(
        f"{worker_base_url}/v1/generate",
        headers={"Authorization": f"Bearer {worker_api_key}", "Idempotency-Key": job_id},
        json={
            "job_id": job_id, "preset": "edit", "prompt": "a cozy cat",
            "input_image_url": input_url,
            "parameters": {"steps": 6, "cfg": 1.8, "width": 512, "height": 512, "seed": 1234},
            "callback_url": f"{mock_backend_url}/cb/{job_id}",
            "upload_url": f"{mock_backend_url}/storage/out/{out_name}",
            "upload_method": "PUT",
            "timeout_sec": 120,
        },
        timeout=10,
    )
    assert resp.status_code == 202, resp.text

    # Poll for callback (worker writes it after upload)
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            r = httpx.get(f"{mock_backend_url}/cb_peek/{job_id}", timeout=3)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(2)
    else:
        pytest.fail("no callback within 240s")

    cb = r.json()
    assert cb["verified"] is True
    assert cb["payload"]["status"] == "success"

    # Verify output exists
    r = httpx.get(f"{mock_backend_url}/storage/out/{out_name}", timeout=5)
    assert r.status_code == 200
    assert len(r.content) > 1024
