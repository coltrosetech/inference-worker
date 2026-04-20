"""Run edit vs edit_premium on a set of (image, prompt) pairs and save outputs.

Bypasses the worker HTTP layer — talks to ComfyUI directly using the worker's
preset classes for parameter injection. Results saved to samples/outputs/.
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


log = get_logger("visual_compare")


SAMPLES_DIR = Path("/workspace/works/samples")
INPUTS_DIR = SAMPLES_DIR / "inputs"
OUTPUTS_DIR = SAMPLES_DIR / "outputs"


CASES = [
    {
        "image": "portrait.jpg",
        "prompt": "the same person wearing round black sunglasses and smiling warmly",
        "label": "portrait_sunglasses",
    },
    {
        "image": "landscape.jpg",
        "prompt": "transform the scene into a snowy winter landscape, snow covering the mountains and the road, cold atmosphere",
        "label": "landscape_winter",
    },
    {
        "image": "object.jpg",
        "prompt": "the same composition but the wooden cutting board is replaced by an ornate golden metal plate with royal engravings",
        "label": "object_gold_plate",
    },
]


PRESET_CONFIGS = {
    "edit": {
        "params": {"steps": 6, "cfg": 1.8, "strength": 0.75, "preservation": 0.6, "seed": 42},
        "resolution": (1024, 1024),
    },
    "edit_premium": {
        "params": {"steps": 20, "cfg": 1.0, "guidance": 2.5, "seed": 42},
        "resolution": (1024, 1024),
    },
}


async def run_case(
    comfyui: ComfyUIClient,
    settings,
    preset_name: str,
    image_filename: str,
    prompt: str,
    label: str,
) -> Path | None:
    preset = PRESETS[preset_name]
    cu_input = settings.comfyui_path / "input"
    cu_input.mkdir(parents=True, exist_ok=True)

    staged_name = f"cmp_{label}_{preset_name}.png"
    staged = cu_input / staged_name
    # Copy the JPG as-is; ComfyUI LoadImage accepts both jpg/png.
    # The staged name ends .png but LoadImage sniffs format.
    shutil.copyfile(INPUTS_DIR / image_filename, staged)

    config = PRESET_CONFIGS[preset_name]
    params_kwargs = dict(config["params"])
    params_kwargs["prompt"] = prompt
    params_kwargs["width"] = config["resolution"][0]
    params_kwargs["height"] = config["resolution"][1]
    params = preset.Parameters(**params_kwargs)

    tpl = preset.load_template(settings.workflows_path)
    paths = InputPaths(input_image=staged_name)
    tpl = preset.inject(tpl, params, paths)

    # Route the save to a unique prefix so we can find the output file.
    prefix = f"cmp_{label}_{preset_name}"
    try:
        tpl.set_input("save", "filename_prefix", prefix)
    except KeyError:
        pass

    if tpl.is_api_format():
        api_prompt = tpl.to_dict()
    else:
        api_prompt = await comfyui.convert_workflow(tpl.to_dict())

    log.info("case.start", preset=preset_name, image=image_filename, label=label)
    client_id = new_client_id()
    pid = await comfyui.submit_prompt(api_prompt, client_id=client_id)
    result = await comfyui.wait_for_completion(pid, client_id, timeout_sec=900.0)

    staged.unlink(missing_ok=True)

    if not result.success:
        log.error("case.failed", preset=preset_name, label=label,
                  message=result.error_message)
        return None

    # Resolve output file from the history's outputs dict.
    out_dir = settings.comfyui_path / "output"
    result_path: Path | None = None
    for _, node_out in result.outputs.items():
        for bucket in ("images", "gifs", "videos"):
            for item in node_out.get(bucket, []):
                name = item.get("filename", "")
                subfolder = item.get("subfolder", "")
                path = out_dir / subfolder / name if subfolder else out_dir / name
                if path.exists() and name.startswith(prefix):
                    result_path = path
                    break
            if result_path:
                break
        if result_path:
            break

    if result_path is None:
        log.error("case.no_output", preset=preset_name, label=label)
        return None

    final = OUTPUTS_DIR / f"{label}_{preset_name}.png"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result_path, final)
    log.info("case.done", preset=preset_name, label=label, path=str(final))
    return final


async def main() -> int:
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as http:
        await wait_for_comfyui(http, settings.comfyui_base_url, timeout_sec=120.0)
        comfyui = ComfyUIClient(http, settings.comfyui_base_url)

        for case in CASES:
            for preset_name in ("edit", "edit_premium"):
                try:
                    await run_case(
                        comfyui, settings, preset_name,
                        case["image"], case["prompt"], case["label"],
                    )
                except Exception:
                    log.exception("case.error", preset=preset_name,
                                  label=case["label"])

    print("\n=== RESULTS ===")
    for case in CASES:
        print(f"[{case['label']}]  prompt: {case['prompt']}")
        for p in ("edit", "edit_premium"):
            f = OUTPUTS_DIR / f"{case['label']}_{p}.png"
            status = f"{f.stat().st_size // 1024} KB" if f.exists() else "MISSING"
            print(f"  {p:15} -> {f.name}  ({status})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
