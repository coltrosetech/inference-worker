from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from worker.api import cancel, generate, health
from worker.app_state import AppState
from worker.core.config import get_settings
from worker.core.logging import configure_logging, get_logger
from worker.pipeline.warmup import warm_all
from worker.state import worker_state


log = get_logger("worker.main")


async def _warmup_and_mark_ready(state: AppState) -> None:
    try:
        results = await warm_all(state.http, state.comfyui, state.settings)
    except Exception:
        log.exception("worker.warmup.exception")
        return
    log.info("worker.warmup.complete", results=results)
    if results and all(results.values()):
        worker_state.mark_ready()
        log.info("worker.ready")
    else:
        log.error("worker.not_ready", results=results)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state = AppState.build()
    app.state.app_state = state
    log.info("worker.startup", worker_id=state.settings.resolved_worker_id(),
             port=state.settings.worker_port)
    runner_task = asyncio.create_task(state.runner.run_forever())
    warmup_task = asyncio.create_task(_warmup_and_mark_ready(state))
    try:
        yield
    finally:
        log.info("worker.shutdown.begin")
        warmup_task.cancel()
        try:
            await warmup_task
        except (asyncio.CancelledError, Exception):
            pass
        await state.aclose()
        try:
            await asyncio.wait_for(runner_task, timeout=60.0)
        except asyncio.TimeoutError:
            log.warning("worker.shutdown.runner_timeout")
        log.info("worker.shutdown.done")


def create_app() -> FastAPI:
    s = get_settings()
    configure_logging(level=s.log_level, fmt=s.log_format)
    app = FastAPI(title="Inference Worker", version="0.1.0", lifespan=_lifespan)
    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(cancel.router)
    return app


app = create_app()
