from __future__ import annotations

import asyncio
import json

import httpx

from worker.core.hmac_sign import sign
from worker.core.logging import get_logger


log = get_logger("worker.callback")


async def send_callback(
    client: httpx.AsyncClient,
    *,
    url: str,
    secret: str,
    payload: dict,
    max_retries: int = 5,
    backoff_base: float = 1.0,
    timeout_sec: float = 20.0,
) -> bool:
    """POST payload to url with HMAC signature. Returns True on 2xx, False otherwise.

    5xx and network errors: retry with exponential backoff up to `max_retries`.
    4xx: retry at most once (the URL or auth is likely bad).
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    attempt = 0
    max_client_errors = 1
    client_errors = 0
    while True:
        sig = sign(body, secret=secret)
        try:
            r = await client.post(
                url,
                content=body,
                headers={"content-type": "application/json", "x-worker-signature": sig},
                timeout=timeout_sec,
            )
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            log.warning("callback.network_error", url=url, attempt=attempt, error=str(e))
            r = None

        if r is not None and 200 <= r.status_code < 300:
            return True

        status = r.status_code if r is not None else 0
        if r is not None and 400 <= r.status_code < 500:
            client_errors += 1
            log.warning("callback.4xx", url=url, status=status, attempt=attempt)
            if client_errors > max_client_errors:
                return False
        else:
            log.warning("callback.5xx_or_net", url=url, status=status, attempt=attempt)

        if attempt >= max_retries:
            return False
        await asyncio.sleep(backoff_base * (2**attempt))
        attempt += 1
