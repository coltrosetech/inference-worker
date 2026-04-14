import asyncio
import time

import pytest

from worker.pipeline.scheduler import Pools


@pytest.mark.asyncio
async def test_gpu_semaphore_serializes():
    p = Pools(io=4, cpu=2, gpu=1)
    order: list[str] = []

    async def work(tag: str) -> None:
        async with p.gpu:
            order.append(f"{tag}-in")
            await asyncio.sleep(0.02)
            order.append(f"{tag}-out")

    await asyncio.gather(work("a"), work("b"), work("c"))
    pairs = [(order[i], order[i+1]) for i in range(0, len(order), 2)]
    for inp, out in pairs:
        assert inp.endswith("-in")
        assert out.endswith("-out")
        assert inp[0] == out[0]


@pytest.mark.asyncio
async def test_io_semaphore_allows_concurrency():
    p = Pools(io=4, cpu=2, gpu=1)
    timestamps: list[float] = []

    async def work() -> None:
        async with p.io:
            timestamps.append(time.monotonic())
            await asyncio.sleep(0.05)

    t0 = time.monotonic()
    await asyncio.gather(*[work() for _ in range(4)])
    assert all(t - t0 < 0.02 for t in timestamps)


@pytest.mark.asyncio
async def test_run_cpu_offloads_blocking_fn():
    p = Pools(io=4, cpu=2, gpu=1)

    def blocking_add(a, b):
        time.sleep(0.01)
        return a + b

    r = await p.run_cpu(blocking_add, 2, 3)
    assert r == 5
