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


def _pjob(jid: str, preset: str) -> Job:
    return Job(job_id=jid, preset=preset, raw_request={"job_id": jid})


async def _drain(q: JobQueue) -> list[str]:
    out = []
    while q.depth():
        out.append((await q.get()).job_id)
    return out


@pytest.mark.asyncio
async def test_affinity_batches_same_preset():
    # Interleaved arrival, but same-preset jobs should come out grouped.
    q = JobQueue(max_depth=8, affinity_run_limit=4)
    await q.put(_pjob("flf_a", "wan_flf2v"))
    await q.put(_pjob("i2v_a", "wan_i2v"))
    await q.put(_pjob("flf_b", "wan_flf2v"))
    await q.put(_pjob("i2v_b", "wan_i2v"))
    # flf2v runs first (FIFO head), pulls the other flf2v with it, then the i2v pair.
    assert await _drain(q) == ["flf_a", "flf_b", "i2v_a", "i2v_b"]


@pytest.mark.asyncio
async def test_affinity_respects_run_limit_no_starvation():
    # With a run limit of 2, a waiting different preset must get a turn.
    q = JobQueue(max_depth=8, affinity_run_limit=2)
    for jid in ["a1", "a2", "a3", "a4"]:
        await q.put(_pjob(jid, "A"))
    await q.put(_pjob("b1", "B"))  # arrives after four A's but must not starve
    # 2 A's, then yield to FIFO head (b1), then remaining A's.
    assert await _drain(q) == ["a1", "a2", "b1", "a3", "a4"]


@pytest.mark.asyncio
async def test_limit_one_is_pure_fifo():
    q = JobQueue(max_depth=8, affinity_run_limit=1)
    await q.put(_pjob("a1", "A"))
    await q.put(_pjob("b1", "B"))
    await q.put(_pjob("a2", "A"))
    assert await _drain(q) == ["a1", "b1", "a2"]
