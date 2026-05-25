from __future__ import annotations

import socket
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Identity
    worker_id: str = "wk-auto"
    worker_api_key: str = Field(..., min_length=16)
    callback_hmac_secret: str = Field(..., min_length=16)

    # Paths
    comfyui_path: Path
    models_path: Path
    cache_path: Path
    workflows_path: Path

    # Runtime
    worker_port: int = 8000
    comfyui_internal_port: int = 8188
    comfyui_host: str = "127.0.0.1"
    max_queue_depth: int = 8
    job_timeout_sec_default: int = 300
    gpu_device: int = 0
    # Preset-affinity scheduling: run up to N consecutive jobs of the same preset
    # before yielding to the FIFO head, so the GPU isn't reloading a different
    # model set every job (an i2v↔flf2v swap costs ~85s). <=1 disables (pure FIFO).
    affinity_run_limit: int = 4

    # External
    hf_token: str = ""
    cloudflare_tunnel_token: str = ""
    allowed_backend_ips: str = ""

    # Warmup
    warmup_presets: str = ""  # comma-separated preset names; empty = all registered

    # Observability
    log_level: str = "info"
    log_format: Literal["json", "text"] = "json"
    metrics_port: int = 9090
    sentry_dsn: str = ""

    @field_validator("worker_api_key", "callback_hmac_secret")
    @classmethod
    def _non_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("secret cannot be blank")
        return v

    def resolved_worker_id(self) -> str:
        if self.worker_id in ("wk-auto", "", "auto"):
            return f"wk-{socket.gethostname()}"
        return self.worker_id

    @property
    def comfyui_base_url(self) -> str:
        return f"http://{self.comfyui_host}:{self.comfyui_internal_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
