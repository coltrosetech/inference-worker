from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import httpx

from worker.core.errors import AppError, ErrorCode


async def upload_file(
    client: httpx.AsyncClient,
    url: str,
    src: Path,
    *,
    method: Literal["PUT", "POST"] = "PUT",
    content_type: str = "application/octet-stream",
    max_retries: int = 3,
    backoff_base: float = 1.0,
    timeout_sec: float = 60.0,
) -> int:
    """Upload `src` to `url` via PUT or POST. Retries on 5xx/network errors only."""
    body = src.read_bytes()
    headers = {"content-type": content_type, "content-length": str(len(body))}
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.request(method, url, content=body, headers=headers, timeout=timeout_sec)
            if resp.status_code in (401, 403):
                raise AppError(
                    ErrorCode.UPLOAD_AUTH_FAILED,
                    f"HTTP {resp.status_code}: signed URL rejected",
                    retryable=False,
                )
            if 400 <= resp.status_code < 500:
                raise AppError(
                    ErrorCode.UPLOAD_FAILED,
                    f"HTTP {resp.status_code} from upload endpoint",
                    retryable=False,
                )
            if resp.status_code >= 500:
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                if attempt == max_retries:
                    break
                await asyncio.sleep(backoff_base * (2**attempt))
                continue
            return len(body)
        except AppError:
            raise
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            last_err = e
            if attempt == max_retries:
                break
            await asyncio.sleep(backoff_base * (2**attempt))
    raise AppError(
        ErrorCode.UPLOAD_FAILED,
        f"upload failed after {max_retries + 1} attempts: {last_err}",
        retryable=True,
    )
