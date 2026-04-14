from __future__ import annotations

from dataclasses import dataclass

import httpx

from worker.comfyui.client import ComfyUIClient
from worker.core.config import Settings, get_settings
from worker.core.logging import get_logger
from worker.pipeline.executor import Executor
from worker.pipeline.model_manager import ModelManager
from worker.pipeline.queue import JobQueue
from worker.pipeline.runner import Runner
from worker.pipeline.scheduler import Pools


log = get_logger("worker.app_state")


@dataclass
class AppState:
    settings: Settings
    http: httpx.AsyncClient
    comfyui: ComfyUIClient
    pools: Pools
    model_manager: ModelManager
    queue: JobQueue
    executor: Executor
    runner: Runner

    @classmethod
    def build(cls) -> AppState:
        s = get_settings()
        http = httpx.AsyncClient()
        comfyui = ComfyUIClient(http, s.comfyui_base_url)
        pools = Pools(io=8, cpu=2, gpu=1)
        mm = ModelManager()
        queue = JobQueue(max_depth=s.max_queue_depth)

        comfyui_input_dir = s.comfyui_path / "input"
        comfyui_output_dir = s.comfyui_path / "output"
        comfyui_input_dir.mkdir(parents=True, exist_ok=True)
        comfyui_output_dir.mkdir(parents=True, exist_ok=True)

        executor = Executor(
            http=http,
            comfyui=comfyui,
            pools=pools,
            model_manager=mm,
            workflows_dir=s.workflows_path,
            comfyui_input_dir=comfyui_input_dir,
            comfyui_output_dir=comfyui_output_dir,
            callback_secret=s.callback_hmac_secret,
            worker_id=s.resolved_worker_id(),
        )
        runner = Runner(queue, executor)
        return cls(
            settings=s, http=http, comfyui=comfyui, pools=pools,
            model_manager=mm, queue=queue, executor=executor, runner=runner,
        )

    async def aclose(self) -> None:
        await self.runner.stop()
        await self.pools.aclose()
        await self.http.aclose()
