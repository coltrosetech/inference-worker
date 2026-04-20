#!/usr/bin/env python
"""Standalone entry point: run one dummy inference per preset to warm ComfyUI.

Thin wrapper around `worker.pipeline.warmup.warm_all`. Safe to run multiple times.
"""
from __future__ import annotations

import asyncio
import sys

import httpx

from worker.comfyui.client import ComfyUIClient
from worker.core.config import get_settings
from worker.core.logging import configure_logging, get_logger
from worker.pipeline.warmup import warm_all


log = get_logger("worker.warmup")


async def main() -> int:
    s = get_settings()
    configure_logging(level=s.log_level, fmt=s.log_format)
    async with httpx.AsyncClient() as http:
        comfyui = ComfyUIClient(http, s.comfyui_base_url)
        results = await warm_all(http, comfyui, s)
    if results and all(results.values()):
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
