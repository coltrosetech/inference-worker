import asyncio

import pytest

from worker.core.errors import AppError, ErrorCode
from worker.pipeline.queue import Job, JobQueue


def _job(jid: str = "a") -> Job:
    return Job(job_id=jid, preset="edit", raw_request={"job_id": jid})


@pytest.mark.asyncio
async def test_put_and_size():
    q = JobQueue(max_depth=4)
    assert q.depth() == 0
    await q.put(_job("1"))
    assert q.depth() == 1


@pytest.mark.asyncio
async def test_full_queue_raises_429():
    q = JobQueue(max_depth=2)
    await q.put(_job("1"))
    await q.put(_job("2"))
    with pytest.raises(AppError) as exc:
        await q.put(_job("3"))
    assert exc.value.code is ErrorCode.INVALID_PARAMETERS
    assert "full" in exc.value.message.lower()
    assert q.depth() == 2


@pytest.mark.asyncio
async def test_idempotency_duplicate_rejected():
    q = JobQueue(max_depth=4)
    await q.put(_job("1"))
    with pytest.raises(AppError) as exc:
        await q.put(_job("1"))
    assert "duplicate" in exc.value.message.lower()


@pytest.mark.asyncio
async def test_get_blocks_until_put():
    q = JobQueue(max_depth=4)

    async def producer():
        await asyncio.sleep(0.01)
        await q.put(_job("1"))

    async def consumer():
        return await q.get()

    _, job = await asyncio.gather(producer(), consumer())
    assert job.job_id == "1"


@pytest.mark.asyncio
async def test_remove_queued_job():
    q = JobQueue(max_depth=4)
    await q.put(_job("1"))
    await q.put(_job("2"))
    removed = q.remove_if_queued("1")
    assert removed is True
    assert q.depth() == 1


@pytest.mark.asyncio
async def test_remove_unknown_returns_false():
    q = JobQueue(max_depth=4)
    await q.put(_job("1"))
    assert q.remove_if_queued("nope") is False
