from __future__ import annotations

import hashlib
import hmac
import time


def sign(body: bytes, *, secret: str, now: int | None = None) -> str:
    """Compute the `X-Worker-Signature` header value for `body`.

    Header format: `t=<unix-ts>,v1=<hex-sha256>`
    The signed payload is `<ts>.<body-bytes>`.
    """
    ts = int(now if now is not None else time.time())
    signed = f"{ts}.".encode() + body
    mac = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify(
    body: bytes,
    header: str,
    *,
    secret: str,
    now: int | None = None,
    skew: int = 300,
) -> bool:
    """Return True iff `header` is a valid signature of `body` under `secret`."""
    try:
        parts = dict(p.strip().split("=", 1) for p in header.split(","))
        ts = int(parts["t"])
        received_mac = parts["v1"]
    except (KeyError, ValueError):
        return False
    current = int(now if now is not None else time.time())
    if abs(current - ts) > skew:
        return False
    signed = f"{ts}.".encode() + body
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return hmac.compare_digest(received_mac, expected)
