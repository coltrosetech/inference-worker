from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class WorkerState:
    started_at: float = field(default_factory=time.time)
    ready: bool = False
    current_job_id: str | None = None
    model_loaded: str | None = None
    comfyui_alive: bool = False
    queue_depth: int = 0

    def mark_ready(self) -> None:
        self.ready = True

    def mark_not_ready(self) -> None:
        self.ready = False

    def uptime_sec(self) -> int:
        return int(time.time() - self.started_at)


worker_state = WorkerState()
