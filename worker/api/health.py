from __future__ import annotations

from fastapi import APIRouter, Request

from worker import __version__
from worker.core.config import get_settings
from worker.gpu_info import safe_gpu_info
from worker.state import worker_state


router = APIRouter(tags=["health"])


@router.get("/v1/health")
async def health(request: Request) -> dict:
    s = get_settings()
    state = getattr(request.app.state, "app_state", None)
    comfyui_alive = False
    queue_depth = 0
    current_job = None
    model_loaded = None
    if state is not None:
        try:
            from worker.comfyui.readiness import is_alive
            comfyui_alive = await is_alive(state.http, s.comfyui_base_url)
        except Exception:
            comfyui_alive = False
        queue_depth = state.queue.depth()
        current_job = worker_state.current_job_id
        mode = state.model_manager.current_mode()
        model_loaded = mode.value if mode else None
    gpu = safe_gpu_info(gpu_device=s.gpu_device)
    return {
        "ok": True,
        "ready": worker_state.ready,
        "worker_id": s.resolved_worker_id(),
        "gpu": gpu.name,
        "vram_used_gb": round(gpu.vram_used_gb, 2),
        "vram_total_gb": round(gpu.vram_total_gb, 2),
        "gpu_util_ratio": round(gpu.util_ratio, 2),
        "queue_depth": queue_depth,
        "current_job": current_job,
        "model_loaded": model_loaded,
        "comfyui_alive": comfyui_alive,
        "uptime_sec": worker_state.uptime_sec(),
        "version": __version__,
    }
