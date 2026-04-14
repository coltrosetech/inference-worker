from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from worker.api.generate import _get_queue_and_settings
from worker.core.auth import require_bearer


router = APIRouter(tags=["cancel"])


class CancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(..., min_length=1, max_length=128)


@router.post("/v1/cancel", dependencies=[Depends(require_bearer)])
async def cancel(req: CancelRequest, request: Request):
    queue, _settings = _get_queue_and_settings(request)
    if queue.remove_if_queued(req.job_id):
        return {"cancelled": True, "stage": "queued"}

    # Try executor cancel path if app_state exists
    try:
        state = request.app.state.app_state
    except AttributeError:
        state = None
    if state is not None and req.job_id in state.executor._active_prompt_ids:  # type: ignore[attr-defined]
        await state.executor.cancel(req.job_id)
        return {"cancelled": True, "stage": "running"}

    raise HTTPException(
        status_code=404,
        detail={"code": "UNKNOWN_JOB", "message": "job not in queue or running"},
    )
