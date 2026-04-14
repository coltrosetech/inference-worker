from worker.gpu_info import GpuInfo, safe_gpu_info


def test_safe_gpu_info_returns_defaults_when_nvml_unavailable(monkeypatch):
    monkeypatch.setattr("worker.gpu_info._try_nvml", lambda _gpu_device: None)
    info = safe_gpu_info(gpu_device=0)
    assert isinstance(info, GpuInfo)
    assert info.name == ""
    assert info.vram_total_gb == 0.0
