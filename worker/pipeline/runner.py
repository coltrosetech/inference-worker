from __future__ import annotations

import asyncio

from worker.core.logging import get_logger


log = get_logger("worker.runner")


class Runner:
    """Drains JobQueue into Executor.run_job in a loop until asked to stop."""

    def __init__(self, queue, executor) -> None:
        self._queue = queue
        self._executor = executor
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        while not self._stop.is_set():
            get_task = asyncio.create_task(self._queue.get())
            stop_task = asyncio.create_task(self._stop.wait())
            done, pending = await asyncio.wait(
                {get_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if stop_task in done and get_task not in done:
                return
            try:
                job = get_task.result()
            except asyncio.CancelledError:
                return
            try:
                await self._executor.run_job(job)
            except Exception:  # noqa: BLE001
                log.exception("runner.job_crashed", job_id=getattr(job, "job_id", "?"))

    async def stop(self) -> None:
        self._stop.set()
