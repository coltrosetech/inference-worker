from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class Pools:
    """Three concurrency pools for the inference pipeline.

    - io: async semaphore for HTTP download/upload
    - cpu: ThreadPoolExecutor for blocking preprocessing (Pillow, canny, etc.)
    - gpu: async semaphore with size 1 to serialize GPU work
    """

    def __init__(self, io: int = 8, cpu: int = 2, gpu: int = 1) -> None:
        self.io = asyncio.Semaphore(io)
        self.cpu_size = cpu
        self.gpu = asyncio.Semaphore(gpu)
        self._cpu_pool = ThreadPoolExecutor(max_workers=cpu, thread_name_prefix="cpu-pool")

    async def run_cpu(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._cpu_pool,
            lambda: fn(*args, **kwargs),
        )

    async def aclose(self) -> None:
        self._cpu_pool.shutdown(wait=False, cancel_futures=True)
