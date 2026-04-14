from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from worker.core.config import get_settings


def require_bearer(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency that enforces `Authorization: Bearer <WORKER_API_KEY>`."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "missing or malformed Authorization header"},
        )
    token = authorization.split(" ", 1)[1].strip()
    expected = get_settings().worker_api_key
    if not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "invalid api key"},
        )
