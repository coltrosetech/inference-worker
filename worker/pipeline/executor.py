from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from pydantic import ValidationError

from worker.comfyui.client import ComfyUIClient, new_client_id
from worker.core.callback import send_callback
from worker.core.errors import AppError, ErrorCode
from worker.core.logging import get_logger
from worker.io.downloader import download_to_file
from worker.io.uploader import upload_file
from worker.pipeline.model_manager import ModelManager
from worker.pipeline.queue import Job
from worker.pipeline.scheduler import Pools
from worker.presets import get_preset
from worker.presets.base import InputPaths


log = get_logger("worker.executor")


@dataclass
class JobContext:
    job: Job
    stages_ms: dict[str, int] = field(default_factory=dict)


class Executor:
    """Orchestrates one job through download → preprocess → ComfyUI → upload → callback."""

    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        comfyui: ComfyUIClient,
        pools: Pools,
        model_manager: ModelManager,
        workflows_dir: Path,
        comfyui_input_dir: Path,
        comfyui_output_dir: Path,
        callback_secret: str,
        worker_id: str,
    ) -> None:
        self._http = http
        self._comfyui = comfyui
        self._pools = pools
        self._model_manager = model_manager
        self._workflows_dir = workflows_dir
        self._cu_input = comfyui_input_dir
        self._cu_output = comfyui_output_dir
        self._callback_secret = callback_secret
        self._worker_id = worker_id
        self._active_prompt_ids: dict[str, str] = {}

    async def run_job(self, job: Job) -> None:
        """Top-level per-job coroutine. Never raises — always emits a callback."""
        ctx = JobContext(job=job)
        req = job.raw_request
        started = time.monotonic()

        try:
            preset = get_preset(req["preset"])
            try:
                params = preset.Parameters(
                    prompt=req.get("prompt", ""),
                    negative_prompt=req.get("negative_prompt", ""),
                    **{k: v for k, v in req.get("parameters", {}).items() if v is not None},
                )
            except ValidationError as ve:
                raise AppError(ErrorCode.INVALID_PARAMETERS, str(ve), retryable=False) from ve
            await self._model_manager.ensure(preset.mode)

            # Stage: download
            with _stage(ctx, "download"):
                input_path = self._cu_input / f"{job.job_id}_input.png"
                ref_path: Path | None = None
                mask_path: Path | None = None
                async with self._pools.io:
                    await download_to_file(self._http, req["input_image_url"], input_path)
                if req.get("reference_image_url"):
                    ref_path = self._cu_input / f"{job.job_id}_ref.png"
                    async with self._pools.io:
                        await download_to_file(self._http, req["reference_image_url"], ref_path)
                if req.get("mask_image_url"):
                    mask_path = self._cu_input / f"{job.job_id}_mask.png"
                    async with self._pools.io:
                        await download_to_file(self._http, req["mask_image_url"], mask_path)

            # Stage: preprocess (thread-pool offload)
            with _stage(ctx, "preprocess"):
                await self._pools.run_cpu(_resize_max_side, input_path, 2048)
                if ref_path:
                    await self._pools.run_cpu(_resize_max_side, ref_path, 2048)
                if mask_path:
                    await self._pools.run_cpu(_resize_max_side, mask_path, 2048)

            input_paths = InputPaths(
                input_image=input_path.name,
                reference_image=ref_path.name if ref_path else None,
                mask_image=mask_path.name if mask_path else None,
            )

            # Stage: inference
            with _stage(ctx, "inference"):
                async with self._pools.gpu:
                    result_path = await self._run_inference(
                        preset, params, input_paths,
                        job_id=job.job_id,
                        timeout_sec=float(req.get("timeout_sec") or 300),
                    )

            # Stage: upload
            with _stage(ctx, "upload"):
                async with self._pools.io:
                    size = await upload_file(
                        self._http,
                        req["upload_url"],
                        result_path,
                        method=req.get("upload_method", "PUT"),
                        content_type=preset.output_content_type,
                    )

            duration_ms = int((time.monotonic() - started) * 1000)
            await send_callback(
                self._http,
                url=req["callback_url"],
                secret=self._callback_secret,
                payload={
                    "job_id": job.job_id,
                    "worker_id": self._worker_id,
                    "status": "success",
                    "output_url": req["upload_url"].split("?", 1)[0],
                    "output_kind": preset.output_content_type,
                    "output_bytes": size,
                    "duration_ms": duration_ms,
                    "stages_ms": ctx.stages_ms,
                    "metadata": {
                        "preset": preset.name,
                        "seed": _extract_seed(params),
                        "model": "juggernautXL_v9_lightning",
                        "steps": getattr(params, "steps", None),
                    },
                },
            )
            log.info("job.completed", job_id=job.job_id, preset=preset.name,
                     duration_ms=duration_ms, stages_ms=ctx.stages_ms)

            _cleanup_paths(input_path, ref_path, mask_path, result_path)

        except AppError as ae:
            duration_ms = int((time.monotonic() - started) * 1000)
            log.error("job.failed", job_id=job.job_id, code=ae.code.value, retryable=ae.retryable, message=ae.message)
            await send_callback(
                self._http,
                url=req["callback_url"],
                secret=self._callback_secret,
                payload={
                    "job_id": job.job_id,
                    "worker_id": self._worker_id,
                    "status": "failed",
                    "duration_ms": duration_ms,
                    "stages_ms": ctx.stages_ms,
                    "error": ae.to_dict(),
                    "metadata": {"preset": req.get("preset")},
                },
            )

        except Exception as e:  # noqa: BLE001 — last-resort
            duration_ms = int((time.monotonic() - started) * 1000)
            log.exception("job.internal_error", job_id=job.job_id)
            await send_callback(
                self._http,
                url=req["callback_url"],
                secret=self._callback_secret,
                payload={
                    "job_id": job.job_id,
                    "worker_id": self._worker_id,
                    "status": "failed",
                    "duration_ms": duration_ms,
                    "stages_ms": ctx.stages_ms,
                    "error": {"code": ErrorCode.INTERNAL_ERROR.value, "message": str(e), "retryable": True},
                    "metadata": {"preset": req.get("preset")},
                },
            )

    async def cancel(self, job_id: str) -> None:
        pid = self._active_prompt_ids.get(job_id)
        if pid:
            await self._comfyui.interrupt()

    async def _run_inference(
        self,
        preset,
        params,
        input_paths: InputPaths,
        *,
        job_id: str,
        timeout_sec: float,
    ) -> Path:
        tpl = preset.load_template(self._workflows_dir)
        tpl = preset.inject(tpl, params, input_paths)
        if tpl.is_api_format():
            try:
                tpl.set_input("save", "filename_prefix", job_id)
            except KeyError:
                pass
            api_prompt = tpl.to_dict()
        else:
            try:
                tpl.set_widget("save", 0, job_id)
            except KeyError:
                pass
            api_prompt = await self._comfyui.convert_workflow(tpl.to_dict())
        client_id = new_client_id()
        prompt_id = await self._comfyui.submit_prompt(api_prompt, client_id=client_id)
        self._active_prompt_ids[job_id] = prompt_id
        try:
            result = await self._comfyui.wait_for_completion(prompt_id, client_id, timeout_sec=timeout_sec)
        except TimeoutError as e:
            await self._comfyui.interrupt()
            raise AppError(ErrorCode.JOB_TIMEOUT, str(e), retryable=True) from e
        finally:
            self._active_prompt_ids.pop(job_id, None)

        if not result.success:
            raise AppError(ErrorCode.INFERENCE_FAILED, result.error_message or "inference failed", retryable=True)

        return _resolve_output_path(result.outputs, self._cu_output, preset.output_extension)


@contextmanager
def _stage(ctx: JobContext, name: str):
    t0 = time.monotonic()
    try:
        yield
    finally:
        ctx.stages_ms[name] = int((time.monotonic() - t0) * 1000)


def _resize_max_side(path: Path, max_side: int) -> None:
    img = Image.open(path)
    img.load()
    if max(img.size) <= max_side:
        return
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    img.save(path)


def _resolve_output_path(outputs: dict, output_dir: Path, expected_ext: str) -> Path:
    """Pick the first output file from ComfyUI's outputs dict."""
    for _, node_out in outputs.items():
        for bucket in ("images", "gifs", "videos"):
            for item in node_out.get(bucket, []):
                name = item.get("filename", "")
                subfolder = item.get("subfolder", "")
                path = output_dir / subfolder / name if subfolder else output_dir / name
                if path.suffix.lstrip(".").lower() == expected_ext.lower():
                    return path
    raise AppError(ErrorCode.INFERENCE_FAILED, "no matching output file in ComfyUI history", retryable=True)


def _extract_seed(params: Any) -> int | None:
    return getattr(params, "seed", None)


def _cleanup_paths(*paths: Path | None) -> None:
    for p in paths:
        if p:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
