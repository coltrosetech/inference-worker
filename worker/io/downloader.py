from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from worker.core.errors import AppError, ErrorCode

DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50 MB hard cap per input


async def download_to_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout_sec: float = 30.0,
) -> int:
    """Download `url` to `dest`, returning the byte count.

    Retryable on network / 5xx / timeout. Non-retryable on 4xx or too-large payloads.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with client.stream("GET", url, timeout=timeout_sec, follow_redirects=True) as resp:
                if 400 <= resp.status_code < 500:
                    raise AppError(
                        ErrorCode.INPUT_FETCH_FAILED,
                        f"HTTP {resp.status_code} for {url}",
                        retryable=False,
                    )
                if resp.status_code >= 500:
                    raise _RetryableHTTP(f"HTTP {resp.status_code} for {url}")
                written = 0
                with dest.open("wb") as f:
                    async for chunk in resp.aiter_bytes():
                        written += len(chunk)
                        if written > max_bytes:
                            dest.unlink(missing_ok=True)
                            raise AppError(
                                ErrorCode.INPUT_TOO_LARGE,
                                f"input exceeds {max_bytes} bytes",
                                retryable=False,
                            )
                        f.write(chunk)
                return written
        except AppError:
            raise
        except (_RetryableHTTP, httpx.HTTPError, asyncio.TimeoutError) as e:
            last_err = e
            if attempt == max_retries:
                break
            await asyncio.sleep(backoff_base * (2**attempt))
    raise AppError(
        ErrorCode.INPUT_FETCH_FAILED,
        f"download failed after {max_retries + 1} attempts: {last_err}",
        retryable=True,
    )


class _RetryableHTTP(Exception):
    pass
