import json
from unittest.mock import AsyncMock

import httpx
import pytest

from worker.comfyui.client import ComfyUIClient
from worker.core.errors import AppError, ErrorCode


@pytest.mark.asyncio
async def test_convert_workflow_posts_and_returns_api_format(respx_mock):
    api_fmt = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}}}
    respx_mock.post("http://cu:8188/workflow/convert").respond(200, json={"api_prompt": api_fmt})
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        out = await c.convert_workflow({"nodes": []})
    assert out == api_fmt


@pytest.mark.asyncio
async def test_convert_workflow_error_raises_app_error(respx_mock):
    respx_mock.post("http://cu:8188/workflow/convert").respond(500)
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        with pytest.raises(AppError) as exc:
            await c.convert_workflow({"nodes": []})
    assert exc.value.code is ErrorCode.COMFYUI_UNAVAILABLE


@pytest.mark.asyncio
async def test_submit_prompt_returns_prompt_id(respx_mock):
    respx_mock.post("http://cu:8188/prompt").respond(200, json={"prompt_id": "pid-42", "number": 1})
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        pid = await c.submit_prompt({"1": {}}, client_id="cid-1")
    assert pid == "pid-42"


@pytest.mark.asyncio
async def test_interrupt_posts(respx_mock):
    route = respx_mock.post("http://cu:8188/interrupt").respond(200)
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        await c.interrupt()
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_get_history_returns_outputs(respx_mock):
    hist = {"pid-42": {"outputs": {"9": {"images": [{"filename": "r.png", "subfolder": "", "type": "output"}]}}}}
    respx_mock.get("http://cu:8188/history/pid-42").respond(200, json=hist)
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        outs = await c.get_history_outputs("pid-42")
    assert outs["9"]["images"][0]["filename"] == "r.png"


@pytest.mark.asyncio
async def test_get_history_raises_when_absent(respx_mock):
    respx_mock.get("http://cu:8188/history/pid-42").respond(200, json={})
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        with pytest.raises(KeyError):
            await c.get_history_outputs("pid-42")
