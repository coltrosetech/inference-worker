from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass

from worker.core.errors import AppError, ErrorCode


@dataclass
class Job:
    job_id: str
    preset: str
    raw_request: dict


class JobQueue:
    """Bounded FIFO job queue with idempotency by job_id."""

    def __init__(self, max_depth: int = 8) -> None:
        self._max = max_depth
        self._items: OrderedDict[str, Job] = OrderedDict()
        self._event = asyncio.Event()

    def depth(self) -> int:
        return len(self._items)

    async def put(self, job: Job) -> None:
        if job.job_id in self._items:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                f"duplicate job_id {job.job_id!r}",
                retryable=False,
            )
        if len(self._items) >= self._max:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                f"queue full (max_depth={self._max})",
                retryable=True,
            )
        self._items[job.job_id] = job
        self._event.set()

    async def get(self) -> Job:
        while not self._items:
            self._event.clear()
            await self._event.wait()
        job_id, job = next(iter(self._items.items()))
        self._items.pop(job_id)
        if not self._items:
            self._event.clear()
        return job

    def peek_position(self, job_id: str) -> int | None:
        for i, jid in enumerate(self._items.keys()):
            if jid == job_id:
                return i
        return None

    def remove_if_queued(self, job_id: str) -> bool:
        if job_id in self._items:
            self._items.pop(job_id)
            if not self._items:
                self._event.clear()
            return True
        return False

    def contains(self, job_id: str) -> bool:
        return job_id in self._items
