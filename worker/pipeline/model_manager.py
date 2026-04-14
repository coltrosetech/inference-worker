from __future__ import annotations

import asyncio
import time

from worker.core.logging import get_logger
from worker.presets.base import Mode


log = get_logger("worker.pipeline.model_manager")


class ModelManager:
    """Phase 1 coordination-only model manager.

    Tracks which Mode (IMAGE / VIDEO) is currently "active" in ComfyUI.
    ComfyUI itself handles the actual GPU load/unload; we just ensure only one
    `ensure()` runs at a time and record swaps for metrics.
    """

    def __init__(self) -> None:
        self._mode: Mode | None = None
        self._lock = asyncio.Lock()
        self.swap_count = 0
        self.last_swap_sec: float = 0.0

    def current_mode(self) -> Mode | None:
        return self._mode

    async def ensure(self, mode: Mode) -> None:
        async with self._lock:
            if self._mode is None:
                self._mode = mode
                log.info("model_manager.initial_mode", mode=mode.value)
                return
            if self._mode == mode:
                return
            t0 = time.monotonic()
            self._mode = mode
            self.last_swap_sec = time.monotonic() - t0
            self.swap_count += 1
            log.info("model_manager.swap", to=mode.value, swap_count=self.swap_count, duration_sec=self.last_swap_sec)
