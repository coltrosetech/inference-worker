from pathlib import Path

import httpx
import pytest

from worker.core.errors import AppError, ErrorCode
from worker.io.downloader import download_to_file


@pytest.mark.asyncio
async def test_download_writes_file(respx_mock, tmp_path):
    respx_mock.get("https://x/a.png").respond(200, content=b"PNGBYTES")
    out = tmp_path / "a.png"
    async with httpx.AsyncClient() as client:
        size = await download_to_file(client, "https://x/a.png", out)
    assert out.read_bytes() == b"PNGBYTES"
    assert size == len(b"PNGBYTES")


@pytest.mark.asyncio
async def test_download_404_raises_non_retryable(respx_mock, tmp_path):
    respx_mock.get("https://x/a.png").respond(404)
    async with httpx.AsyncClient() as client:
        with pytest.raises(AppError) as exc:
            await download_to_file(client, "https://x/a.png", tmp_path / "a.png")
    assert exc.value.code is ErrorCode.INPUT_FETCH_FAILED
    assert exc.value.retryable is False


@pytest.mark.asyncio
async def test_download_503_retries_then_succeeds(respx_mock, tmp_path):
    route = respx_mock.get("https://x/a.png")
    route.side_effect = [httpx.Response(503), httpx.Response(503), httpx.Response(200, content=b"ok")]
    async with httpx.AsyncClient() as client:
        await download_to_file(client, "https://x/a.png", tmp_path / "a.png", max_retries=3, backoff_base=0.01)
    assert route.call_count == 3


@pytest.mark.asyncio
async def test_download_exhausts_retries(respx_mock, tmp_path):
    respx_mock.get("https://x/a.png").respond(503)
    async with httpx.AsyncClient() as client:
        with pytest.raises(AppError) as exc:
            await download_to_file(client, "https://x/a.png", tmp_path / "a.png", max_retries=2, backoff_base=0.01)
    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_download_too_large_rejected(respx_mock, tmp_path):
    big = b"x" * (2 * 1024 * 1024)
    respx_mock.get("https://x/a.png").respond(200, content=big)
    async with httpx.AsyncClient() as client:
        with pytest.raises(AppError) as exc:
            await download_to_file(client, "https://x/a.png", tmp_path / "a.png", max_bytes=1024 * 1024)
    assert exc.value.code is ErrorCode.INPUT_TOO_LARGE
