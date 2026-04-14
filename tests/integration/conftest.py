import os
import time

import httpx
import pytest


@pytest.fixture(scope="session")
def worker_base_url() -> str:
    return os.environ.get("WORKER_URL", "http://127.0.0.1:8000")


@pytest.fixture(scope="session")
def worker_api_key() -> str:
    key = os.environ.get("WORKER_API_KEY")
    if not key:
        pytest.skip("WORKER_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def mock_backend_url() -> str:
    return os.environ.get("MOCK_BACKEND_URL", "http://127.0.0.1:9100")


@pytest.fixture(scope="session")
def wait_for_worker_ready(worker_base_url):
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            r = httpx.get(f"{worker_base_url}/v1/health", timeout=3.0)
            if r.status_code == 200 and r.json().get("ready"):
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    pytest.skip("worker did not become ready in 240s")
