from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from worker.comfyui.client import ExecutionResult
from worker.pipeline import warmup as wm


class _FakeComfyUIClient:
    def __init__(self, *, success: bool = True, error: str | None = None,
                 convert_raises: Exception | None = None) -> None:
        self.success = success
        self.error = error
        self.convert_raises = convert_raises
        self.submitted_prompts: list[dict] = []
        self.converted_workflows: list[dict] = []

    async def convert_workflow(self, full_workflow: dict) -> dict:
        if self.convert_raises is not None:
            raise self.convert_raises
        self.converted_workflows.append(full_workflow)
        return {"1": {"class_type": "Stub", "inputs": {}}}

    async def submit_prompt(self, api_prompt: dict, client_id: str) -> str:
        self.submitted_prompts.append(api_prompt)
        return "pid-1"

    async def wait_for_completion(self, prompt_id: str, client_id: str, *, timeout_sec: float):
        return ExecutionResult(
            prompt_id=prompt_id,
            outputs={},
            success=self.success,
            error_message=self.error,
        )


@pytest.fixture
def _settings_with_workflows(tmp_path: Path, monkeypatch):
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()
    comfyui_path = tmp_path / "comfyui"
    comfyui_path.mkdir()

    repo_workflows = Path(__file__).parent.parent.parent / "workflows"
    (workflows_dir / "edit.json").write_text((repo_workflows / "edit.json").read_text())
    (workflows_dir / "style.json").write_text((repo_workflows / "style.json").read_text())

    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(comfyui_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path / "cache"))
    monkeypatch.setenv("WORKFLOWS_PATH", str(workflows_dir))

    from worker.core.config import get_settings
    get_settings.cache_clear()
    return get_settings()


def test_ensure_stub_image_creates_png(tmp_path: Path):
    p = tmp_path / "stub.png"
    wm._ensure_stub_image(p)
    assert p.exists()
    assert p.stat().st_size > 0


def test_ensure_stub_image_noop_when_present(tmp_path: Path):
    p = tmp_path / "stub.png"
    p.write_bytes(b"existing")
    wm._ensure_stub_image(p)
    assert p.read_bytes() == b"existing"


async def test_warm_one_edit_success(_settings_with_workflows):
    client = _FakeComfyUIClient(success=True)
    ok = await wm.warm_one("edit", client, _settings_with_workflows)
    assert ok is True
    assert len(client.submitted_prompts) == 1
    assert len(client.converted_workflows) == 0
    prompt = client.submitted_prompts[0]
    assert prompt["input_image"]["inputs"]["image"] == "_warm_edit_input.png"
    assert prompt["save"]["inputs"]["filename_prefix"] == "_warm_edit"


async def test_warm_one_legacy_template_triggers_convert(_settings_with_workflows, monkeypatch):
    """A legacy (non-API) template is converted via /workflow/convert before submit."""
    from worker.comfyui.workflow import WorkflowTemplate
    from worker.presets import PRESETS

    legacy = WorkflowTemplate({
        "nodes": [
            {"id": 1, "type": "LoadImage", "title": "input_image",
             "widgets_values": ["x.png", "image"]},
            {"id": 2, "type": "CLIPTextEncode", "title": "positive_prompt",
             "widgets_values": [""]},
            {"id": 3, "type": "CLIPTextEncode", "title": "negative_prompt",
             "widgets_values": [""]},
            {"id": 4, "type": "KSampler", "title": "sampler",
             "widgets_values": [0, "fixed", 4, 1.0, "euler", "normal", 0.7]},
            {"id": 5, "type": "SaveImage", "title": "save",
             "widgets_values": ["output"]},
        ],
        "links": [],
    })
    monkeypatch.setattr(PRESETS["edit"], "load_template", lambda _: legacy)

    client = _FakeComfyUIClient(success=True)
    ok = await wm.warm_one("edit", client, _settings_with_workflows)
    assert ok is True
    assert len(client.converted_workflows) == 1


async def test_warm_one_returns_false_on_execution_error(_settings_with_workflows):
    client = _FakeComfyUIClient(success=False, error="boom")
    ok = await wm.warm_one("edit", client, _settings_with_workflows)
    assert ok is False


async def test_warm_one_unknown_preset_returns_false(_settings_with_workflows):
    client = _FakeComfyUIClient()
    ok = await wm.warm_one("nope", client, _settings_with_workflows)
    assert ok is False
    assert client.submitted_prompts == []


async def test_warm_one_cleans_up_stub_on_failure(_settings_with_workflows):
    client = _FakeComfyUIClient(success=False, error="x")
    await wm.warm_one("edit", client, _settings_with_workflows)
    cu_input = _settings_with_workflows.comfyui_path / "input"
    assert not (cu_input / "_warm_edit_input.png").exists()


async def test_warm_all_returns_empty_on_comfyui_timeout(_settings_with_workflows, monkeypatch):
    async def _raise(*args, **kwargs):
        raise TimeoutError("nope")
    monkeypatch.setattr(wm, "wait_for_comfyui", _raise)

    async with httpx.AsyncClient() as http:
        client = _FakeComfyUIClient()
        results = await wm.warm_all(http, client, _settings_with_workflows)
    assert results == {}


async def test_warm_all_aggregates_per_preset(_settings_with_workflows, monkeypatch):
    async def _ok(*args, **kwargs):
        return None
    monkeypatch.setattr(wm, "wait_for_comfyui", _ok)

    seen: list[str] = []

    async def _fake_warm_one(name, comfyui, settings, *, timeout_sec=180.0):
        seen.append(name)
        return name == "edit"

    monkeypatch.setattr(wm, "warm_one", _fake_warm_one)

    async with httpx.AsyncClient() as http:
        client = _FakeComfyUIClient()
        results = await wm.warm_all(http, client, _settings_with_workflows,
                                    preset_names=["edit", "style"])
    assert results == {"edit": True, "style": False}
    assert seen == ["edit", "style"]


async def test_warm_all_catches_exception_per_preset(_settings_with_workflows, monkeypatch):
    async def _ok(*args, **kwargs):
        return None
    monkeypatch.setattr(wm, "wait_for_comfyui", _ok)

    async def _boom(name, comfyui, settings, *, timeout_sec=180.0):
        raise RuntimeError("bad")

    monkeypatch.setattr(wm, "warm_one", _boom)

    async with httpx.AsyncClient() as http:
        client = _FakeComfyUIClient()
        results = await wm.warm_all(http, client, _settings_with_workflows,
                                    preset_names=["edit"])
    assert results == {"edit": False}
