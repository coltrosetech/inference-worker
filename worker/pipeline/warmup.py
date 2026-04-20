from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from worker.comfyui.client import ComfyUIClient, new_client_id
from worker.comfyui.readiness import wait_for_comfyui
from worker.core.config import Settings
from worker.core.logging import get_logger
from worker.presets import PRESETS
from worker.presets.base import InputPaths


log = get_logger("worker.warmup")


STUB_PNG_SIZE = (512, 512)


def _ensure_stub_image(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", STUB_PNG_SIZE, color=(128, 128, 128)).save(path, "PNG")


def _ensure_stub_mask(path: Path) -> None:
    """Grayscale mask: white 256x256 square centered on black background."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", STUB_PNG_SIZE, color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = STUB_PNG_SIZE[0] // 2, STUB_PNG_SIZE[1] // 2
    draw.rectangle((cx - 128, cy - 128, cx + 128, cy + 128), fill=255)
    img.save(path, "PNG")


async def warm_one(
    preset_name: str,
    comfyui: ComfyUIClient,
    settings: Settings,
    *,
    timeout_sec: float | None = None,
) -> bool:
    if preset_name not in PRESETS:
        log.warning("warmup.unknown_preset", preset=preset_name)
        return False

    preset = PRESETS[preset_name]
    cu_input = settings.comfyui_path / "input"
    cu_input.mkdir(parents=True, exist_ok=True)

    input_name = f"_warm_{preset_name}_input.png"
    ref_name = f"_warm_{preset_name}_ref.png"
    mask_name = f"_warm_{preset_name}_mask.png"
    input_file = cu_input / input_name
    ref_file = cu_input / ref_name
    mask_file = cu_input / mask_name
    _ensure_stub_image(input_file)
    if preset.needs_reference_image:
        _ensure_stub_image(ref_file)
    if preset.needs_mask_image:
        _ensure_stub_mask(mask_file)

    effective_timeout = timeout_sec if timeout_sec is not None else preset.warmup_timeout_sec

    try:
        tpl = preset.load_template(settings.workflows_path)
        params = preset.Parameters(**preset.warmup_params())
        paths = InputPaths(
            input_image=input_name,
            reference_image=ref_name if preset.needs_reference_image else None,
            mask_image=mask_name if preset.needs_mask_image else None,
        )
        tpl = preset.inject(tpl, params, paths)

        if tpl.is_api_format():
            try:
                tpl.set_input("save", "filename_prefix", f"_warm_{preset_name}")
            except KeyError:
                pass
            api_prompt = tpl.to_dict()
        else:
            try:
                tpl.set_widget("save", 0, f"_warm_{preset_name}")
            except KeyError:
                pass
            api_prompt = await comfyui.convert_workflow(tpl.to_dict())

        client_id = new_client_id()
        pid = await comfyui.submit_prompt(api_prompt, client_id=client_id)
        result = await comfyui.wait_for_completion(pid, client_id, timeout_sec=effective_timeout)
        if result.success:
            log.info("warmup.ok", preset=preset_name)
            return True
        log.error("warmup.failed", preset=preset_name, message=result.error_message)
        return False
    finally:
        input_file.unlink(missing_ok=True)
        if preset.needs_reference_image:
            ref_file.unlink(missing_ok=True)
        if preset.needs_mask_image:
            mask_file.unlink(missing_ok=True)


def _resolve_preset_names(preset_names: Iterable[str] | None, settings: Settings) -> list[str]:
    if preset_names is not None:
        return list(preset_names)
    configured = (settings.warmup_presets or "").strip()
    if configured:
        return [n.strip() for n in configured.split(",") if n.strip()]
    return list(PRESETS.keys())


async def warm_all(
    http: httpx.AsyncClient,
    comfyui: ComfyUIClient,
    settings: Settings,
    *,
    preset_names: Iterable[str] | None = None,
    comfyui_wait_timeout_sec: float = 120.0,
) -> dict[str, bool]:
    try:
        await wait_for_comfyui(http, settings.comfyui_base_url, timeout_sec=comfyui_wait_timeout_sec)
    except TimeoutError as e:
        log.error("warmup.comfyui_timeout", message=str(e))
        return {}

    names = _resolve_preset_names(preset_names, settings)
    results: dict[str, bool] = {}
    for name in names:
        try:
            results[name] = await warm_one(name, comfyui, settings)
        except Exception:
            log.exception("warmup.preset_error", preset=name)
            results[name] = False
    return results
