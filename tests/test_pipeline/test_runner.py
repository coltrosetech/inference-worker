import asyncio
from unittest.mock import AsyncMock

import pytest

from worker.pipeline.queue import Job, JobQueue
from worker.pipeline.runner import Runner


@pytest.mark.asyncio
async def test_runner_consumes_jobs_and_stops():
    q = JobQueue(max_depth=4)
    exe = AsyncMock()
    exe.run_job = AsyncMock()
    await q.put(Job("a", "edit", {"job_id": "a"}))
    await q.put(Job("b", "edit", {"job_id": "b"}))

    r = Runner(q, exe)
    task = asyncio.create_task(r.run_forever())
    for _ in range(50):
        if exe.run_job.await_count >= 2:
            break
        await asyncio.sleep(0.01)
    await r.stop()
    await task
    assert exe.run_job.await_count == 2


@pytest.mark.asyncio
async def test_runner_survives_executor_exception():
    q = JobQueue(max_depth=4)
    exe = AsyncMock()
    exe.run_job = AsyncMock(side_effect=[RuntimeError("boom"), None])
    await q.put(Job("a", "edit", {"job_id": "a"}))
    await q.put(Job("b", "edit", {"job_id": "b"}))

    r = Runner(q, exe)
    task = asyncio.create_task(r.run_forever())
    for _ in range(50):
        if exe.run_job.await_count >= 2:
            break
        await asyncio.sleep(0.01)
    await r.stop()
    await task
    assert exe.run_job.await_count == 2
