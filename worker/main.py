from __future__ import annotations

from fastapi import FastAPI

from worker.api import health
from worker.core.config import get_settings
from worker.core.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    s = get_settings()
    configure_logging(level=s.log_level, fmt=s.log_format)
    log = get_logger("worker.main")
    log.info("worker.startup", worker_id=s.resolved_worker_id(), port=s.worker_port)

    app = FastAPI(title="Inference Worker", version="0.1.0")
    app.include_router(health.router)
    return app


app = create_app()
