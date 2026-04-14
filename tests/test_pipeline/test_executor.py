import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from PIL import Image

from worker.comfyui.client import ComfyUIClient, ExecutionResult
from worker.pipeline.executor import Executor, JobContext
from worker.pipeline.model_manager import ModelManager
from worker.pipeline.queue import Job
from worker.pipeline.scheduler import Pools


def _sample_request(tmp_output: Path, input_url: str, callback_url: str, upload_url: str) -> dict:
    return {
        "job_id": "j1",
        "preset": "edit",
        "prompt": "a cat",
        "negative_prompt": "",
        "input_image_url": input_url,
        "parameters": {"steps": 6, "cfg": 1.8, "width": 512, "height": 512, "seed": 42, "strength": 0.7},
        "callback_url": callback_url,
        "upload_url": upload_url,
        "upload_method": "PUT",
        "timeout_sec": 60,
    }


def _make_png(path: Path, size=(512, 512)) -> None:
    Image.new("RGB", size, (128, 128, 255)).save(path, "PNG")


@pytest.mark.asyncio
async def test_executor_happy_path(tmp_path, respx_mock):
    _make_png(tmp_path / "src.png")
    _make_png(tmp_path / "out.png")
    comfyui_input = tmp_path / "cu_input"
    comfyui_output = tmp_path / "cu_output"
    comfyui_input.mkdir()
    comfyui_output.mkdir()

    respx_mock.get("https://storage/in.png").respond(200, content=(tmp_path / "src.png").read_bytes())
    upload_route = respx_mock.put("https://storage/out?sig=xyz").respond(200)
    callback_route = respx_mock.post("https://cb/done").respond(200)

    comfy = MagicMock(spec=ComfyUIClient)
    comfy.base_url = "http://127.0.0.1:8188"
    comfy.convert_workflow = AsyncMock(return_value={"1": {"class_type": "X", "inputs": {}}})
    comfy.submit_prompt = AsyncMock(return_value="pid-1")

    async def fake_wait(prompt_id, client_id, *, timeout_sec):
        dst = comfyui_output / "j1_00001_.png"
        dst.write_bytes((tmp_path / "out.png").read_bytes())
        return ExecutionResult(
            prompt_id=prompt_id,
            outputs={"13": {"images": [{"filename": dst.name, "subfolder": "", "type": "output"}]}},
            success=True,
        )

    comfy.wait_for_completion = AsyncMock(side_effect=fake_wait)
    comfy.interrupt = AsyncMock()

    # Path to project workflows/ dir
    workflows_dir = Path(__file__).parent.parent.parent / "workflows"

    async with httpx.AsyncClient() as http:
        exe = Executor(
            http=http,
            comfyui=comfy,
            pools=Pools(),
            model_manager=ModelManager(),
            workflows_dir=workflows_dir,
            comfyui_input_dir=comfyui_input,
            comfyui_output_dir=comfyui_output,
            callback_secret="s" * 32,
            worker_id="wk-test",
        )
        job = Job(job_id="j1", preset="edit",
                  raw_request=_sample_request(tmp_path, "https://storage/in.png", "https://cb/done", "https://storage/out?sig=xyz"))
        await exe.run_job(job)

    assert upload_route.call_count == 1
    assert callback_route.call_count == 1
    body = json.loads(callback_route.calls.last.request.content)
    assert body["status"] == "success"
    assert body["output_url"] == "https://storage/out"  # upload_url without query string
    assert body["metadata"]["preset"] == "edit"
    assert body["metadata"]["seed"] == 42
    assert "stages_ms" in body
