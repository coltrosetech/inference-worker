from pathlib import Path

import httpx
import pytest

from worker.core.errors import AppError, ErrorCode
from worker.io.uploader import upload_file


@pytest.mark.asyncio
async def test_upload_put_success(respx_mock, tmp_path):
    f = tmp_path / "r.png"
    f.write_bytes(b"IMG")
    route = respx_mock.put("https://s/out/r").respond(200)
    async with httpx.AsyncClient() as client:
        size = await upload_file(client, "https://s/out/r", f, method="PUT", content_type="image/png")
    assert size == 3
    assert route.calls.last.request.content == b"IMG"
    assert route.calls.last.request.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_upload_403_non_retryable(respx_mock, tmp_path):
    f = tmp_path / "r.png"; f.write_bytes(b"IMG")
    respx_mock.put("https://s/out/r").respond(403)
    async with httpx.AsyncClient() as client:
        with pytest.raises(AppError) as exc:
            await upload_file(client, "https://s/out/r", f, method="PUT")
    assert exc.value.code is ErrorCode.UPLOAD_AUTH_FAILED
    assert exc.value.retryable is False


@pytest.mark.asyncio
async def test_upload_500_retries(respx_mock, tmp_path):
    f = tmp_path / "r.png"; f.write_bytes(b"IMG")
    route = respx_mock.put("https://s/out/r")
    route.side_effect = [httpx.Response(500), httpx.Response(500), httpx.Response(200)]
    async with httpx.AsyncClient() as client:
        await upload_file(client, "https://s/out/r", f, method="PUT", max_retries=3, backoff_base=0.01)
    assert route.call_count == 3


@pytest.mark.asyncio
async def test_upload_post_method(respx_mock, tmp_path):
    f = tmp_path / "r.png"; f.write_bytes(b"IMG")
    route = respx_mock.post("https://s/out/r").respond(200)
    async with httpx.AsyncClient() as client:
        await upload_file(client, "https://s/out/r", f, method="POST")
    assert route.call_count == 1
