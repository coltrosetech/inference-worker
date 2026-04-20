"""Spawn the ComfyUI venv to produce a clothing-only mask PNG.

The worker's own venv intentionally stays minimal (no torch/transformers),
so we shell out to `/venv/main/bin/python scripts/segment_clothing.py` and
wait for it to write the PNG. Output is a binary L-mode PNG where only the
requested clothing classes are 255; everything else (background, face,
hair, skin, hats, etc.) is 0 — safe to feed straight into an inpaint graph
without destroying regions the user wants preserved.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from worker.core.errors import AppError, ErrorCode
from worker.core.logging import get_logger


log = get_logger("worker.segmentation")

_DEFAULT_PY = os.environ.get("COMFYUI_PYTHON", "/venv/main/bin/python")
_DEFAULT_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "segment_clothing.py"
_DEFAULT_MODEL_DIR = os.environ.get(
    "SEGFORMER_CLOTHES_DIR", "/workspace/ComfyUI/models/segformer_b2_clothes"
)


async def segment_clothing_to_mask(
    input_path: Path,
    output_path: Path,
    categories: list[str],
    *,
    timeout_sec: float = 60.0,
) -> Path:
    """Run the segmentation CLI as a subprocess and return `output_path`.

    Raises AppError on non-zero exit or timeout.
    """
    if not categories:
        raise AppError(ErrorCode.INVALID_PARAMETERS, "auto_mask_categories empty")

    script = Path(os.environ.get("SEGFORMER_SCRIPT", str(_DEFAULT_SCRIPT)))
    if not script.exists():
        raise AppError(ErrorCode.INTERNAL_ERROR, f"segment script missing: {script}")

    cmd = [
        _DEFAULT_PY,
        str(script),
        "--input", str(input_path),
        "--output", str(output_path),
        "--categories", ",".join(categories),
        "--model-dir", _DEFAULT_MODEL_DIR,
    ]
    log.info("segment.start", input=str(input_path), categories=categories)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError as e:
        proc.kill()
        raise AppError(ErrorCode.JOB_TIMEOUT, "segmentation subprocess timed out", retryable=True) from e

    if proc.returncode != 0:
        err = (stderr_b or b"").decode(errors="replace").strip()
        log.error("segment.failed", returncode=proc.returncode, stderr=err[:2000])
        raise AppError(
            ErrorCode.INFERENCE_FAILED,
            f"segmentation failed (rc={proc.returncode}): {err[-400:]}",
            retryable=True,
        )

    if not output_path.exists():
        raise AppError(ErrorCode.INTERNAL_ERROR, "segmentation subprocess did not produce a mask")

    log.info("segment.ok", output=str(output_path), bytes=output_path.stat().st_size)
    return output_path
