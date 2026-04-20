from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from worker.core.auth import require_bearer
from worker.core.config import get_settings
from worker.core.errors import AppError, ErrorCode
from worker.pipeline.queue import Job, JobQueue
from worker.presets import PRESETS, get_preset


router = APIRouter(tags=["generate"])

# Module-level fallback queue used when the full AppState lifespan hasn't run
# (e.g. in unit tests using TestClient without lifespan context).
_fallback_queue: JobQueue | None = None


def _get_queue_and_settings(request: Request):
    """Return (queue, settings) from app.state.app_state if available, else fallback."""
    global _fallback_queue
    try:
        state = request.app.state.app_state
        return state.queue, state.settings
    except AttributeError:
        pass
    # Fallback: use get_settings() and a lazily-created queue per app instance.
    # Cache it on the app object so duplicate-detection works within a test.
    s = get_settings()
    if not hasattr(request.app.state, "_fallback_queue"):
        request.app.state._fallback_queue = JobQueue(max_depth=s.max_queue_depth)
    return request.app.state._fallback_queue, s


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(..., min_length=1, max_length=128)
    preset: Literal["edit", "style", "controlnet", "inpaint", "edit_premium", "ltx_video"]
    prompt: str = Field("", max_length=4000)
    negative_prompt: str = Field("", max_length=4000)
    input_image_url: str
    mask_image_url: str | None = None
    reference_image_url: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    callback_url: str
    upload_url: str
    upload_method: Literal["PUT", "POST"] = "PUT"
    timeout_sec: int = Field(default=300, ge=10, le=1800)


@router.post("/v1/generate", status_code=202, dependencies=[Depends(require_bearer)])
async def generate(
    req: GenerateRequest,
    request: Request,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
):
    if not idempotency_key:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_PARAMETERS", "message": "Idempotency-Key header required"},
        )

    if req.preset not in PRESETS:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_PRESET", "message": f"unknown preset {req.preset!r}"},
        )

    preset = get_preset(req.preset)

    # Preset-specific parameter validation
    try:
        extra = {"prompt": req.prompt}
        if "negative_prompt" in preset.Parameters.model_fields:
            extra["negative_prompt"] = req.negative_prompt
        preset.Parameters(**extra, **{k: v for k, v in req.parameters.items() if v is not None})
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_PARAMETERS", "errors": e.errors()},
        ) from e

    if req.preset == "style" and not req.reference_image_url:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_PARAMETERS", "message": "style preset requires reference_image_url"},
        )
    if req.preset == "inpaint" and not req.mask_image_url and not req.parameters.get("auto_mask"):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_PARAMETERS",
                "message": "inpaint preset requires mask_image_url or parameters.auto_mask=true",
            },
        )

    queue, settings = _get_queue_and_settings(request)
    job = Job(job_id=req.job_id, preset=req.preset, raw_request=req.model_dump())
    try:
        await queue.put(job)
    except AppError as ae:
        if ae.code is ErrorCode.INVALID_PARAMETERS and "duplicate" in ae.message:
            raise HTTPException(status_code=409, detail={"code": "DUPLICATE", "message": ae.message}) from ae
        if ae.code is ErrorCode.INVALID_PARAMETERS and "queue full" in ae.message:
            return _retry_after_response(queue.depth())
        raise HTTPException(status_code=400, detail=ae.to_dict()) from ae

    return {
        "accepted": True,
        "job_id": req.job_id,
        "queue_position": queue.peek_position(req.job_id) or 0,
        "worker_id": settings.resolved_worker_id(),
    }


def _retry_after_response(depth: int):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"code": "QUEUE_FULL", "message": "worker queue is full"},
        headers={"Retry-After": "5"},
    )
