from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from worker.api import health
from worker.app_state import AppState
from worker.core.config import get_settings
from worker.core.logging import configure_logging, get_logger


log = get_logger("worker.main")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state = AppState.build()
    app.state.app_state = state
    log.info("worker.startup", worker_id=state.settings.resolved_worker_id(),
             port=state.settings.worker_port)
    runner_task = asyncio.create_task(state.runner.run_forever())
    try:
        yield
    finally:
        log.info("worker.shutdown.begin")
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
    return app


app = create_app()
