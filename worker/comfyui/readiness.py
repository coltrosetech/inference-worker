from __future__ import annotations

import asyncio

import httpx


async def is_alive(client: httpx.AsyncClient, base_url: str) -> bool:
    """One-shot check: is ComfyUI /system_stats responding 200?"""
    try:
        r = await client.get(f"{base_url}/system_stats", timeout=2.0)
        return r.status_code == 200
    except (httpx.HTTPError, asyncio.TimeoutError):
        return False


async def wait_for_comfyui(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    timeout_sec: float = 120.0,
    interval_sec: float = 1.0,
) -> None:
    """Block until ComfyUI responds 200 on /system_stats, or raise TimeoutError."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec
    while True:
        if await is_alive(client, base_url):
            return
        if loop.time() >= deadline:
            raise TimeoutError(f"ComfyUI not ready on {base_url} after {timeout_sec}s")
        await asyncio.sleep(interval_sec)
