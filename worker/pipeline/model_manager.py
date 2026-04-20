from __future__ import annotations

import asyncio
import time

from worker.comfyui.client import ComfyUIClient
from worker.core.logging import get_logger
from worker.presets.base import Mode


log = get_logger("worker.pipeline.model_manager")


class ModelManager:
    """Coordination layer for IMAGE↔VIDEO mode swaps.

    When the active mode changes, issues a POST /free to ComfyUI so the
    previous mode's weights are released before the new preset loads its own.
    Required on 16 GB GPUs where image + video models cannot coexist.
    """

    def __init__(self, comfyui: ComfyUIClient | None = None) -> None:
        self._mode: Mode | None = None
        self._lock = asyncio.Lock()
        self._comfyui = comfyui
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
            old_mode = self._mode
            if self._comfyui is not None:
                try:
                    await self._comfyui.free(unload_models=True, free_memory=True)
                    log.info("model_manager.comfyui_free",
                             from_mode=old_mode.value, to_mode=mode.value)
                except Exception:
                    log.exception("model_manager.free_failed")
            self._mode = mode
            self.last_swap_sec = time.monotonic() - t0
            self.swap_count += 1
            log.info("model_manager.swap",
                     from_mode=old_mode.value, to=mode.value,
                     swap_count=self.swap_count, duration_sec=self.last_swap_sec)
