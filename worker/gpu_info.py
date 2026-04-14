from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GpuInfo:
    name: str = ""
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    util_ratio: float = 0.0


def _try_nvml(gpu_device: int) -> GpuInfo | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        return GpuInfo(
            name=name,
            vram_used_gb=mem.used / (1024**3),
            vram_total_gb=mem.total / (1024**3),
            util_ratio=util.gpu / 100.0,
        )
    except Exception:
        return None


def safe_gpu_info(*, gpu_device: int) -> GpuInfo:
    info = _try_nvml(gpu_device)
    return info or GpuInfo()
