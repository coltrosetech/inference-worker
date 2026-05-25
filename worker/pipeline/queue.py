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
    """Bounded job queue with idempotency by job_id and preset-affinity dequeue.

    Insertion order is FIFO, but `get()` keeps the GPU's current model set hot:
    it prefers the oldest queued job whose preset matches the last one dequeued,
    for up to `affinity_run_limit` jobs in a row, then falls back to the FIFO head
    so no preset can be starved. Set `affinity_run_limit<=1` for pure FIFO.

    Rationale: switching preset (e.g. wan_i2v↔wan_flf2v) forces ComfyUI to evict
    and reload a 14–28 GB model set (~85s measured). Batching same-preset jobs
    turns N swaps into 1.
    """

    def __init__(self, max_depth: int = 8, affinity_run_limit: int = 4) -> None:
        self._max = max_depth
        self._affinity_run_limit = affinity_run_limit
        self._items: OrderedDict[str, Job] = OrderedDict()
        self._event = asyncio.Event()
        self._last_preset: str | None = None
        self._run_len = 0

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

    def _select_next_id(self) -> str:
        """Pick the next job_id under preset-affinity with bounded fairness.

        - affinity off (limit<=1) or nothing run yet → FIFO head.
        - within the run limit → oldest job of the current (hot) preset, if any.
        - run limit reached → force a switch to the oldest *different*-preset job
          so no preset is starved; if only the hot preset remains, keep going.
        """
        head = next(iter(self._items))
        if self._affinity_run_limit <= 1 or self._last_preset is None:
            return head
        same = next((jid for jid, j in self._items.items() if j.preset == self._last_preset), None)
        diff = next((jid for jid, j in self._items.items() if j.preset != self._last_preset), None)
        if self._run_len < self._affinity_run_limit and same is not None:
            return same
        if diff is not None:
            return diff
        return same if same is not None else head

    async def get(self) -> Job:
        while not self._items:
            self._event.clear()
            await self._event.wait()
        job_id = self._select_next_id()
        job = self._items.pop(job_id)
        if not self._items:
            self._event.clear()
        if job.preset == self._last_preset:
            self._run_len += 1
        else:
            self._last_preset = job.preset
            self._run_len = 1
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
