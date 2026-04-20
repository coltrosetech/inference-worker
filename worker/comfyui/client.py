from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass

import httpx
import websockets

from worker.core.errors import AppError, ErrorCode


@dataclass
class ExecutionResult:
    prompt_id: str
    outputs: dict
    success: bool
    error_message: str | None = None


class ComfyUIClient:
    """Thin async client over ComfyUI's HTTP + WebSocket API."""

    def __init__(self, http: httpx.AsyncClient, base_url: str) -> None:
        self._http = http
        self._base = base_url.rstrip("/")

    @property
    def base_url(self) -> str:
        return self._base

    async def convert_workflow(self, full_workflow: dict) -> dict:
        """POST /workflow/convert — full workflow → API format."""
        try:
            r = await self._http.post(
                f"{self._base}/workflow/convert",
                json={"workflow": full_workflow},
                timeout=30.0,
            )
            if r.status_code >= 500:
                raise AppError(ErrorCode.COMFYUI_UNAVAILABLE, f"convert HTTP {r.status_code}", retryable=True)
            if r.status_code >= 400:
                raise AppError(ErrorCode.INFERENCE_FAILED, f"convert HTTP {r.status_code}: {r.text}", retryable=False)
            data = r.json()
            if "api_prompt" in data:
                return data["api_prompt"]
            return data
        except AppError:
            raise
        except httpx.HTTPError as e:
            raise AppError(ErrorCode.COMFYUI_UNAVAILABLE, f"convert network error: {e}", retryable=True) from e

    async def submit_prompt(self, api_prompt: dict, client_id: str) -> str:
        """POST /prompt — returns prompt_id."""
        try:
            r = await self._http.post(
                f"{self._base}/prompt",
                json={"prompt": api_prompt, "client_id": client_id},
                timeout=30.0,
            )
            if r.status_code >= 500:
                raise AppError(ErrorCode.COMFYUI_UNAVAILABLE, f"prompt HTTP {r.status_code}", retryable=True)
            if r.status_code >= 400:
                raise AppError(ErrorCode.INFERENCE_FAILED, f"prompt HTTP {r.status_code}: {r.text}", retryable=False)
            return r.json()["prompt_id"]
        except AppError:
            raise
        except httpx.HTTPError as e:
            raise AppError(ErrorCode.COMFYUI_UNAVAILABLE, f"prompt network error: {e}", retryable=True) from e

    async def interrupt(self) -> None:
        """POST /interrupt — fire-and-forget (best effort)."""
        try:
            await self._http.post(f"{self._base}/interrupt", timeout=5.0)
        except httpx.HTTPError:
            pass

    async def free(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        """POST /free — drop loaded models to free VRAM."""
        await self._http.post(
            f"{self._base}/free",
            json={"unload_models": unload_models, "free_memory": free_memory},
            timeout=30.0,
        )

    async def get_history_outputs(self, prompt_id: str) -> dict:
        """GET /history/{prompt_id} — returns the `outputs` dict of that prompt."""
        r = await self._http.get(f"{self._base}/history/{prompt_id}", timeout=15.0)
        r.raise_for_status()
        hist = r.json()
        if prompt_id not in hist:
            raise KeyError(f"prompt_id {prompt_id!r} not in history")
        return hist[prompt_id].get("outputs", {})

    async def get_history_entry(self, prompt_id: str) -> dict | None:
        """GET /history/{prompt_id} — returns the full entry, or None if not yet present."""
        try:
            r = await self._http.get(f"{self._base}/history/{prompt_id}", timeout=15.0)
            r.raise_for_status()
            return r.json().get(prompt_id)
        except httpx.HTTPError:
            return None

    async def wait_for_completion(
        self,
        prompt_id: str,
        client_id: str,
        *,
        timeout_sec: float,
    ) -> ExecutionResult:
        """Open a WebSocket, wait until the given prompt_id finishes or errors.

        Returns with success=True when ComfyUI emits `executing: {node: null, prompt_id}`,
        success=False on `execution_error`, raises TimeoutError past deadline.
        """
        ws_url = self._base.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec
        async with websockets.connect(ws_url, max_size=2**23, open_timeout=10) as ws:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for prompt {prompt_id}")
                entry = await self.get_history_entry(prompt_id)
                if entry is not None:
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        msg = "execution_error"
                        for m in status.get("messages", []):
                            if isinstance(m, (list, tuple)) and len(m) >= 2 and m[0] == "execution_error":
                                payload = m[1] if isinstance(m[1], dict) else {}
                                msg = payload.get("exception_message") or payload.get("exception_type") or msg
                                break
                        return ExecutionResult(
                            prompt_id=prompt_id, outputs={},
                            success=False, error_message=msg,
                        )
                    if status.get("completed") is True or entry.get("outputs"):
                        return ExecutionResult(
                            prompt_id=prompt_id,
                            outputs=entry.get("outputs", {}),
                            success=True,
                        )
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 2.0))
                except asyncio.TimeoutError:
                    continue
                if isinstance(raw, bytes):
                    continue  # preview images: ignore
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                mtype = msg.get("type")
                data = msg.get("data", {})
                if data.get("prompt_id") != prompt_id:
                    continue
                if mtype == "executing" and data.get("node") is None:
                    outputs = await self.get_history_outputs(prompt_id)
                    return ExecutionResult(prompt_id=prompt_id, outputs=outputs, success=True)
                if mtype == "execution_error":
                    return ExecutionResult(
                        prompt_id=prompt_id,
                        outputs={},
                        success=False,
                        error_message=data.get("exception_message") or data.get("exception_type") or "unknown",
                    )


def new_client_id() -> str:
    return f"worker-{uuid.uuid4().hex}"
