#!/usr/bin/env python
"""Run one dummy inference per preset to force kernel compilation + cache warm-up.

Invoked by the worker after ComfyUI becomes reachable and before flipping ready=true.
Safe to run multiple times; idempotent effect on the filesystem.
"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import httpx

from worker.comfyui.client import ComfyUIClient, new_client_id
from worker.comfyui.readiness import wait_for_comfyui
from worker.core.config import get_settings
from worker.core.logging import configure_logging, get_logger
from worker.presets import PRESETS
from worker.presets.base import InputPaths


log = get_logger("worker.warmup")


async def warm_one(preset_name: str, http: httpx.AsyncClient, settings) -> bool:
    preset = PRESETS[preset_name]
    cu_input = settings.comfyui_path / "input"
    cu_input.mkdir(parents=True, exist_ok=True)

    stub_src = Path("tests/fixtures/input_512.png")
    stub_ref = Path("tests/fixtures/ref_512.png")
    if not stub_src.exists():
        log.warning("warmup.fixture_missing", path=str(stub_src))
        return False
    input_name = f"_warm_{preset_name}_input.png"
    ref_name = f"_warm_{preset_name}_ref.png"
    shutil.copyfile(stub_src, cu_input / input_name)
    if preset_name == "style":
        shutil.copyfile(stub_ref, cu_input / ref_name)

    tpl = preset.load_template(settings.workflows_path)
    params = preset.Parameters(prompt="warmup", steps=4, cfg=1.2, width=512, height=512, seed=0)
    paths = InputPaths(
        input_image=input_name,
        reference_image=ref_name if preset_name == "style" else None,
    )
    tpl = preset.inject(tpl, params, paths)

    client = ComfyUIClient(http, settings.comfyui_base_url)
    api_prompt = await client.convert_workflow(tpl.to_dict())
    client_id = new_client_id()
    pid = await client.submit_prompt(api_prompt, client_id=client_id)
    result = await client.wait_for_completion(pid, client_id, timeout_sec=180.0)
    (cu_input / input_name).unlink(missing_ok=True)
    if preset_name == "style":
        (cu_input / ref_name).unlink(missing_ok=True)
    if result.success:
        log.info("warmup.ok", preset=preset_name)
        return True
    log.error("warmup.failed", preset=preset_name, message=result.error_message)
    return False


async def main() -> int:
    s = get_settings()
    configure_logging(level=s.log_level, fmt=s.log_format)
    async with httpx.AsyncClient() as http:
        try:
            await wait_for_comfyui(http, s.comfyui_base_url, timeout_sec=120.0)
        except TimeoutError as e:
            log.error("warmup.comfyui_timeout", message=str(e))
            return 1
        ok = True
        for p in ("edit", "style"):
            if p in PRESETS:
                try:
                    ok = ok and await warm_one(p, http, s)
                except Exception:
                    log.exception("warmup.preset_error", preset=p)
                    ok = False
        return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
