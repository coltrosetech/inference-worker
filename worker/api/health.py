from __future__ import annotations

from fastapi import APIRouter

from worker import __version__
from worker.core.config import get_settings
from worker.state import worker_state

router = APIRouter(tags=["health"])


@router.get("/v1/health")
async def health() -> dict:
    s = get_settings()
    return {
        "ok": True,
        "ready": worker_state.ready,
        "worker_id": s.resolved_worker_id(),
        "current_job": worker_state.current_job_id,
        "model_loaded": worker_state.model_loaded,
        "comfyui_alive": worker_state.comfyui_alive,
        "queue_depth": worker_state.queue_depth,
        "uptime_sec": worker_state.uptime_sec(),
        "version": __version__,
    }
