import httpx
import pytest

from worker.comfyui.readiness import wait_for_comfyui, is_alive


@pytest.mark.asyncio
async def test_is_alive_true_when_200(respx_mock):
    respx_mock.get("http://x:8188/system_stats").respond(200, json={"ok": True})
    async with httpx.AsyncClient() as client:
        assert await is_alive(client, "http://x:8188") is True


@pytest.mark.asyncio
async def test_is_alive_false_on_http_error(respx_mock):
    respx_mock.get("http://x:8188/system_stats").respond(500)
    async with httpx.AsyncClient() as client:
        assert await is_alive(client, "http://x:8188") is False


@pytest.mark.asyncio
async def test_wait_for_comfyui_returns_when_ready(respx_mock):
    route = respx_mock.get("http://x:8188/system_stats")
    route.side_effect = [httpx.Response(500), httpx.Response(500), httpx.Response(200, json={})]
    async with httpx.AsyncClient() as client:
        await wait_for_comfyui(client, "http://x:8188", timeout_sec=5, interval_sec=0.01)
    assert route.call_count == 3


@pytest.mark.asyncio
async def test_wait_for_comfyui_raises_on_timeout(respx_mock):
    respx_mock.get("http://x:8188/system_stats").respond(500)
    async with httpx.AsyncClient() as client:
        with pytest.raises(TimeoutError):
            await wait_for_comfyui(client, "http://x:8188", timeout_sec=0.05, interval_sec=0.01)
