import httpx
import pytest

from worker.core.callback import send_callback
from worker.core.hmac_sign import verify


@pytest.mark.asyncio
async def test_callback_success_signs_and_posts(respx_mock):
    recorded = {}

    def handler(req: httpx.Request) -> httpx.Response:
        recorded["body"] = bytes(req.content)
        recorded["sig"] = req.headers.get("x-worker-signature", "")
        return httpx.Response(200)

    respx_mock.post("https://b/cb").mock(side_effect=handler)
    async with httpx.AsyncClient() as client:
        await send_callback(
            client,
            url="https://b/cb",
            secret="s" * 32,
            payload={"job_id": "j1", "status": "success"},
            max_retries=0,
        )
    assert verify(recorded["body"], recorded["sig"], secret="s" * 32) is True


@pytest.mark.asyncio
async def test_callback_5xx_retries_then_drops(respx_mock):
    route = respx_mock.post("https://b/cb")
    route.side_effect = [httpx.Response(500)] * 5 + [httpx.Response(500)]
    async with httpx.AsyncClient() as client:
        result = await send_callback(
            client, url="https://b/cb", secret="s" * 32,
            payload={"job_id": "j", "status": "failed"},
            max_retries=2, backoff_base=0.01,
        )
    assert result is False
    assert route.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_callback_2xx_returns_true(respx_mock):
    respx_mock.post("https://b/cb").respond(202)
    async with httpx.AsyncClient() as client:
        ok = await send_callback(
            client, url="https://b/cb", secret="s" * 32,
            payload={"job_id": "j"}, max_retries=0,
        )
    assert ok is True


@pytest.mark.asyncio
async def test_callback_4xx_retries_once(respx_mock):
    route = respx_mock.post("https://b/cb")
    route.side_effect = [httpx.Response(400), httpx.Response(400)]
    async with httpx.AsyncClient() as client:
        ok = await send_callback(
            client, url="https://b/cb", secret="s" * 32,
            payload={"job_id": "j"}, max_retries=5, backoff_base=0.01,
        )
    assert ok is False
    assert route.call_count == 2  # 4xx short-circuits to 1 retry
