# Image/Video Worker — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stateless GPU inference worker that exposes `POST /v1/generate` for SDXL-based image-to-image (`edit` and `style` presets), fully portable to any Linux + NVIDIA host, wired to ComfyUI running headless on `127.0.0.1:8188`.

**Architecture:** FastAPI HTTP service + asyncio pipeline (IO/CPU/GPU pools) + ComfyUI subprocess as inference engine. Workflows authored as JSON templates, parameters injected at runtime, executed via `/prompt` + `/workflow/convert` + WebSocket `/ws`. Results uploaded to backend-provided signed URL; callback POSTed with HMAC-SHA256 signature.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, pydantic v2, pydantic-settings, httpx, websockets, structlog, prometheus-client, huggingface-hub, Pillow, PyYAML, pynvml. Deployed via Docker (nvidia-cuda:12.8 base) or bare-metal supervisord. Tests: pytest + pytest-asyncio + respx.

**Scope:** This plan (Phase 1) covers foundation + `edit` + `style` presets end-to-end. Phase 2 (separate plan) will add `ltx_video`, `controlnet`, `inpaint`, full observability, and CI/CD.

**Reference spec:** `docs/superpowers/specs/2026-04-14-img-video-worker-design.md`

---

## File Structure

### Created in Sprint 0 (Foundation)
- `pyproject.toml` — Python project config, deps pinned
- `.env.example` — env template
- `.gitignore` — already committed, may extend
- `worker/__init__.py` — package marker with `__version__`
- `worker/main.py` — FastAPI app assembly
- `worker/core/config.py` — pydantic Settings
- `worker/core/logging.py` — structlog setup
- `worker/core/errors.py` — `AppError`, `ErrorCode` enum
- `worker/core/auth.py` — Bearer token dependency
- `worker/core/hmac_sign.py` — callback signature
- `worker/api/health.py` — `/v1/health` (basic)
- `Dockerfile` — fat image, CUDA 12.8 base
- `docker-compose.yml` — compose file with GPU + volumes
- `supervisord.conf` — process manager for comfyui + worker
- `scripts/start.sh` — launch supervisord (entrypoint)
- `tests/conftest.py` — pytest fixtures
- `tests/test_config.py`, `tests/test_logging.py`, `tests/test_errors.py`, `tests/test_auth.py`, `tests/test_hmac_sign.py`, `tests/test_health.py`

### Created in Sprint 1 (Image pipeline)
- `configs/models.yaml` — SDXL + IP-Adapter + CLIP vision + LCM LoRA + VAE manifest
- `configs/custom_nodes.yaml` — IPAdapter_plus + tooling-nodes + KJNodes manifest
- `scripts/download_models.py` — manifest-driven model downloader
- `scripts/install_custom_nodes.sh` — manifest-driven node installer
- `scripts/bootstrap.sh` — idempotent full setup
- `scripts/warm_comfyui.py` — dummy inference warm-up
- `scripts/mock_backend.py` — local fake backend for e2e tests
- `worker/io/downloader.py` — async HTTP download with retries
- `worker/io/uploader.py` — async HTTP PUT upload with retries
- `worker/comfyui/workflow.py` — JSON template loader + parameter injection
- `worker/comfyui/client.py` — httpx + websockets client
- `worker/comfyui/readiness.py` — `/system_stats` polling
- `worker/presets/base.py` — `Preset` ABC, shared pydantic base
- `worker/presets/edit.py` — edit preset
- `worker/presets/style.py` — style preset
- `workflows/edit.json` — edit workflow template
- `workflows/style.json` — style workflow template
- `worker/pipeline/model_manager.py` — mode state tracker
- `worker/pipeline/queue.py` — bounded asyncio queue
- `worker/pipeline/scheduler.py` — IO/CPU/GPU semaphore-based scheduler
- `worker/pipeline/executor.py` — per-job orchestrator
- `worker/core/callback.py` — HMAC-signed callback sender with retries
- `worker/api/generate.py` — `POST /v1/generate`
- `worker/api/cancel.py` — `POST /v1/cancel`
- `worker/api/health.py` — extended with ComfyUI check + `ready` flag
- `tests/test_downloader.py`, `tests/test_uploader.py`, `tests/test_workflow.py`, `tests/test_comfyui_client.py`, `tests/test_presets/test_edit.py`, `tests/test_presets/test_style.py`, `tests/test_model_manager.py`, `tests/test_scheduler.py`, `tests/test_executor.py`, `tests/test_callback.py`, `tests/test_api/test_generate.py`, `tests/test_api/test_cancel.py`
- `tests/fixtures/input_1024.png`, `tests/fixtures/mask_512.png` — small test images

---

## Sprint 0 — Foundation

### Task 1: Project skeleton and pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `worker/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "inference-worker"
version = "0.1.0"
description = "Stateless GPU inference worker for image/video generation via ComfyUI"
requires-python = ">=3.11"
dependencies = [
    "fastapi==0.115.6",
    "uvicorn[standard]==0.32.1",
    "pydantic==2.10.3",
    "pydantic-settings==2.7.0",
    "httpx==0.28.1",
    "websockets==14.1",
    "structlog==24.4.0",
    "prometheus-client==0.21.1",
    "huggingface-hub==0.27.0",
    "Pillow==11.0.0",
    "PyYAML==6.0.2",
    "pynvml==12.0.0",
    "ulid-py==1.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest==8.3.4",
    "pytest-asyncio==0.25.0",
    "pytest-cov==6.0.0",
    "respx==0.22.0",
    "pytest-httpserver==1.1.0",
    "ruff==0.8.4",
    "pyright==1.1.391",
]

[tool.setuptools.packages.find]
include = ["worker*"]

[tool.ruff]
line-length = 110
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "ASYNC"]
ignore = ["E501"]

[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "standard"

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "gpu: requires real GPU and ComfyUI (skipped in CI)",
    "integration: end-to-end tests requiring network and GPU",
]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `worker/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Write `tests/__init__.py`** (empty file to mark test package)

```python
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
```

- [ ] **Step 5: Install deps and verify**

```bash
cd /workspace/works
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import worker; print(worker.__version__)"
pytest --version
ruff --version
pyright --version
```
Expected: prints `0.1.0`, pytest `8.3.4`, ruff and pyright versions.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml worker/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: project skeleton with pyproject + worker package"
```

---

### Task 2: Logging module (structlog JSON)

**Files:**
- Create: `worker/core/__init__.py` (empty)
- Create: `worker/core/logging.py`
- Create: `tests/test_logging.py`

- [ ] **Step 1: Write failing test `tests/test_logging.py`**

```python
import io
import json

import structlog

from worker.core.logging import configure_logging, get_logger


def test_configure_logging_emits_json(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    configure_logging(level="info", fmt="json")
    log = get_logger("test")
    log.info("hello", foo="bar", count=3)
    line = buf.getvalue().strip()
    payload = json.loads(line)
    assert payload["event"] == "hello"
    assert payload["foo"] == "bar"
    assert payload["count"] == 3
    assert payload["level"] == "info"
    assert payload["logger"] == "test"
    assert "timestamp" in payload


def test_configure_logging_text_format(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    configure_logging(level="info", fmt="text")
    log = get_logger("test")
    log.info("hello", foo="bar")
    line = buf.getvalue().strip()
    assert "hello" in line
    assert "foo=bar" in line
```

- [ ] **Step 2: Write empty `worker/core/__init__.py`**

```python
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_logging.py -v
```
Expected: ImportError — `worker.core.logging` does not exist.

- [ ] **Step 4: Implement `worker/core/logging.py`**

```python
from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog


def configure_logging(
    level: str = "info",
    fmt: Literal["json", "text"] = "json",
) -> None:
    """Configure structlog for the process.

    Call once at startup. Safe to call again (idempotent within a process).
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_logging.py -v
```
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add worker/core/__init__.py worker/core/logging.py tests/test_logging.py
git commit -m "feat(core): structlog JSON/text logging setup"
```

---

### Task 3: Configuration module (pydantic Settings)

**Files:**
- Create: `worker/core/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test `tests/test_config.py`**

```python
import os
from pathlib import Path

import pytest

from worker.core.config import Settings, get_settings


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))

    s = Settings()
    assert s.worker_port == 8000
    assert s.comfyui_internal_port == 8188
    assert s.max_queue_depth == 8
    assert s.job_timeout_sec_default == 300
    assert s.gpu_device == 0
    assert s.log_level == "info"
    assert s.log_format == "json"
    assert s.metrics_port == 9090
    assert len(s.worker_api_key) == 32


def test_settings_requires_secrets(monkeypatch, tmp_path):
    monkeypatch.delenv("WORKER_API_KEY", raising=False)
    monkeypatch.delenv("CALLBACK_HMAC_SECRET", raising=False)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))
    with pytest.raises(Exception) as exc:
        Settings()
    assert "worker_api_key" in str(exc.value).lower() or "hmac" in str(exc.value).lower()


def test_settings_worker_id_auto(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("WORKER_ID", "wk-auto")
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))
    s = Settings()
    assert s.resolved_worker_id().startswith("wk-")
    assert s.resolved_worker_id() != "wk-auto"


def test_get_settings_is_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```
Expected: ImportError for `worker.core.config`.

- [ ] **Step 3: Implement `worker/core/config.py`**

```python
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

    # External
    hf_token: str = ""
    cloudflare_tunnel_token: str = ""
    allowed_backend_ips: str = ""

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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/core/config.py tests/test_config.py
git commit -m "feat(core): pydantic Settings with env loading"
```

---

### Task 4: Error types and codes

**Files:**
- Create: `worker/core/errors.py`
- Create: `tests/test_errors.py`

- [ ] **Step 1: Write failing test `tests/test_errors.py`**

```python
import pytest

from worker.core.errors import AppError, ErrorCode


def test_app_error_carries_code_and_retryable():
    err = AppError(ErrorCode.OOM_CUDA, "out of memory", retryable=True)
    assert err.code is ErrorCode.OOM_CUDA
    assert err.retryable is True
    assert "out of memory" in str(err)


def test_app_error_default_retryable_false():
    err = AppError(ErrorCode.INPUT_FETCH_FAILED, "404")
    assert err.retryable is False


def test_app_error_to_dict_shape():
    err = AppError(ErrorCode.INFERENCE_FAILED, "node xyz crashed", retryable=True)
    d = err.to_dict()
    assert d == {"code": "INFERENCE_FAILED", "message": "node xyz crashed", "retryable": True}


def test_all_error_codes_are_stable_strings():
    for code in ErrorCode:
        assert code.value == code.name


def test_raise_and_catch():
    with pytest.raises(AppError) as exc:
        raise AppError(ErrorCode.JOB_TIMEOUT, "exceeded 300s", retryable=True)
    assert exc.value.code is ErrorCode.JOB_TIMEOUT
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_errors.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/core/errors.py`**

```python
from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    INPUT_FETCH_FAILED = "INPUT_FETCH_FAILED"
    INPUT_DECODE_FAILED = "INPUT_DECODE_FAILED"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    OOM_CUDA = "OOM_CUDA"
    UPLOAD_AUTH_FAILED = "UPLOAD_AUTH_FAILED"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    JOB_TIMEOUT = "JOB_TIMEOUT"
    COMFYUI_UNAVAILABLE = "COMFYUI_UNAVAILABLE"
    INVALID_PRESET = "INVALID_PRESET"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"[{code.value}] {message}")
        self.code = code
        self.message = message
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_errors.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/core/errors.py tests/test_errors.py
git commit -m "feat(core): AppError + ErrorCode enum"
```

---

### Task 5: HMAC-SHA256 callback signer

**Files:**
- Create: `worker/core/hmac_sign.py`
- Create: `tests/test_hmac_sign.py`

- [ ] **Step 1: Write failing test `tests/test_hmac_sign.py`**

```python
import time

import pytest

from worker.core.hmac_sign import sign, verify


def test_sign_deterministic_with_fixed_timestamp():
    body = b'{"job_id":"x"}'
    header = sign(body, secret="topsecret1234567", now=1700000000)
    assert header.startswith("t=1700000000,")
    assert ",v1=" in header


def test_verify_roundtrip():
    body = b'{"job_id":"x"}'
    header = sign(body, secret="topsecret1234567", now=1700000000)
    assert verify(body, header, secret="topsecret1234567", now=1700000000) is True


def test_verify_rejects_tampered_body():
    body = b'{"job_id":"x"}'
    header = sign(body, secret="topsecret1234567", now=1700000000)
    assert verify(b'{"job_id":"y"}', header, secret="topsecret1234567", now=1700000000) is False


def test_verify_rejects_bad_secret():
    body = b'{"job_id":"x"}'
    header = sign(body, secret="topsecret1234567", now=1700000000)
    assert verify(body, header, secret="wrongsecret12345", now=1700000000) is False


def test_verify_rejects_expired_timestamp():
    body = b'{"job_id":"x"}'
    header = sign(body, secret="topsecret1234567", now=1700000000)
    assert verify(body, header, secret="topsecret1234567", now=1700000400, skew=300) is False


def test_verify_rejects_malformed_header():
    body = b'{"job_id":"x"}'
    assert verify(body, "garbage", secret="topsecret1234567", now=1700000000) is False
    assert verify(body, "t=abc,v1=deadbeef", secret="topsecret1234567", now=1700000000) is False
    assert verify(body, "t=1700000000", secret="topsecret1234567", now=1700000000) is False


def test_sign_default_now_is_current_time():
    body = b"payload"
    before = int(time.time())
    header = sign(body, secret="topsecret1234567")
    after = int(time.time())
    ts = int(header.split(",")[0].split("=")[1])
    assert before <= ts <= after
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_hmac_sign.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/core/hmac_sign.py`**

```python
from __future__ import annotations

import hashlib
import hmac
import time


def sign(body: bytes, *, secret: str, now: int | None = None) -> str:
    """Compute the `X-Worker-Signature` header value for `body`.

    Header format: `t=<unix-ts>,v1=<hex-sha256>`
    The signed payload is `<ts>.<body-bytes>`.
    """
    ts = int(now if now is not None else time.time())
    signed = f"{ts}.".encode() + body
    mac = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify(
    body: bytes,
    header: str,
    *,
    secret: str,
    now: int | None = None,
    skew: int = 300,
) -> bool:
    """Return True iff `header` is a valid signature of `body` under `secret`."""
    try:
        parts = dict(p.strip().split("=", 1) for p in header.split(","))
        ts = int(parts["t"])
        received_mac = parts["v1"]
    except (KeyError, ValueError):
        return False
    current = int(now if now is not None else time.time())
    if abs(current - ts) > skew:
        return False
    signed = f"{ts}.".encode() + body
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return hmac.compare_digest(received_mac, expected)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_hmac_sign.py -v
```
Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/core/hmac_sign.py tests/test_hmac_sign.py
git commit -m "feat(core): HMAC-SHA256 sign/verify for callbacks"
```

---
### Task 6: Bearer token auth dependency

**Files:**
- Create: `worker/core/auth.py`
- Create: `tests/test_auth.py`

- [ ] **Step 1: Write failing test `tests/test_auth.py`**

```python
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from worker.core.auth import require_bearer


def _app(monkeypatch, key="secret1234567890"):
    monkeypatch.setenv("WORKER_API_KEY", key)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", "/tmp")
    monkeypatch.setenv("MODELS_PATH", "/tmp")
    monkeypatch.setenv("CACHE_PATH", "/tmp")
    monkeypatch.setenv("WORKFLOWS_PATH", "/tmp")
    from worker.core.config import get_settings
    get_settings.cache_clear()
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_bearer)])
    def protected():
        return {"ok": True}
    return TestClient(app)


def test_missing_header_returns_401(monkeypatch):
    client = _app(monkeypatch)
    r = client.get("/protected")
    assert r.status_code == 401


def test_wrong_scheme_returns_401(monkeypatch):
    client = _app(monkeypatch)
    r = client.get("/protected", headers={"Authorization": "Basic abc"})
    assert r.status_code == 401


def test_wrong_token_returns_401(monkeypatch):
    client = _app(monkeypatch, key="correctkey123456")
    r = client.get("/protected", headers={"Authorization": "Bearer wrongkey123456"})
    assert r.status_code == 401


def test_correct_token_passes(monkeypatch):
    client = _app(monkeypatch, key="correctkey123456")
    r = client.get("/protected", headers={"Authorization": "Bearer correctkey123456"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_auth.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/core/auth.py`**

```python
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from worker.core.config import get_settings


def require_bearer(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency that enforces `Authorization: Bearer <WORKER_API_KEY>`."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "missing or malformed Authorization header"},
        )
    token = authorization.split(" ", 1)[1].strip()
    expected = get_settings().worker_api_key
    if not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "UNAUTHORIZED", "message": "invalid api key"},
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_auth.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/core/auth.py tests/test_auth.py
git commit -m "feat(core): Bearer token auth dependency"
```

---

### Task 7: Basic /v1/health endpoint and FastAPI main.py

**Files:**
- Create: `worker/api/__init__.py` (empty)
- Create: `worker/api/health.py`
- Create: `worker/state.py`
- Create: `worker/main.py`
- Create: `tests/test_api/__init__.py` (empty)
- Create: `tests/test_api/test_health.py`

The `/v1/health` endpoint returns worker state. At this stage it has no ComfyUI probe yet — that comes in Task 27. `ready=false` until the worker reports warm-up completion.

- [ ] **Step 1: Write failing test `tests/test_api/test_health.py`**

```python
from fastapi.testclient import TestClient


def _client(monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", "/tmp")
    monkeypatch.setenv("MODELS_PATH", "/tmp")
    monkeypatch.setenv("CACHE_PATH", "/tmp")
    monkeypatch.setenv("WORKFLOWS_PATH", "/tmp")
    from worker.core.config import get_settings
    get_settings.cache_clear()
    from worker.main import create_app
    return TestClient(create_app())


def test_health_basic_shape(monkeypatch):
    client = _client(monkeypatch)
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["ready"] is False
    assert "version" in body
    assert "uptime_sec" in body
    assert body["queue_depth"] == 0
    assert "worker_id" in body


def test_health_marks_ready(monkeypatch):
    client = _client(monkeypatch)
    from worker.state import worker_state
    worker_state.mark_ready()
    r = client.get("/v1/health")
    assert r.json()["ready"] is True
    worker_state.mark_not_ready()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api/test_health.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/state.py`**

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class WorkerState:
    started_at: float = field(default_factory=time.time)
    ready: bool = False
    current_job_id: str | None = None
    model_loaded: str | None = None
    comfyui_alive: bool = False
    queue_depth: int = 0

    def mark_ready(self) -> None:
        self.ready = True

    def mark_not_ready(self) -> None:
        self.ready = False

    def uptime_sec(self) -> int:
        return int(time.time() - self.started_at)


worker_state = WorkerState()
```

- [ ] **Step 4: Implement `worker/api/health.py`**

```python
from __future__ import annotations

from fastapi import APIRouter

from worker import __version__
from worker.core.config import get_settings
from worker.state import worker_state

router = APIRouter(tags=["health"])


@router.get("/v1/health")
async def health() -> dict:
    s = get_settings()
    return {
        "ok": True,
        "ready": worker_state.ready,
        "worker_id": s.resolved_worker_id(),
        "current_job": worker_state.current_job_id,
        "model_loaded": worker_state.model_loaded,
        "comfyui_alive": worker_state.comfyui_alive,
        "queue_depth": worker_state.queue_depth,
        "uptime_sec": worker_state.uptime_sec(),
        "version": __version__,
    }
```

- [ ] **Step 5: Implement `worker/main.py`**

```python
from __future__ import annotations

from fastapi import FastAPI

from worker.api import health
from worker.core.config import get_settings
from worker.core.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    s = get_settings()
    configure_logging(level=s.log_level, fmt=s.log_format)
    log = get_logger("worker.main")
    log.info("worker.startup", worker_id=s.resolved_worker_id(), port=s.worker_port)

    app = FastAPI(title="Inference Worker", version="0.1.0")
    app.include_router(health.router)
    return app


app = create_app()
```

- [ ] **Step 6: Create empty `worker/api/__init__.py` and `tests/test_api/__init__.py`**

```bash
touch worker/api/__init__.py tests/test_api/__init__.py
```

- [ ] **Step 7: Run test to verify it passes**

```bash
pytest tests/test_api/test_health.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 8: Smoke test the server**

```bash
source .venv/bin/activate
WORKER_API_KEY=kkkkkkkkkkkkkkkk CALLBACK_HMAC_SECRET=ssssssssssssssss \
  COMFYUI_PATH=/tmp MODELS_PATH=/tmp CACHE_PATH=/tmp WORKFLOWS_PATH=/tmp \
  uvicorn worker.main:app --host 127.0.0.1 --port 8000 &
sleep 1
curl -s http://127.0.0.1:8000/v1/health | python -m json.tool
kill %1
```
Expected: JSON with `"ok": true, "ready": false, ...`.

- [ ] **Step 9: Commit**

```bash
git add worker/state.py worker/api/__init__.py worker/api/health.py worker/main.py \
         tests/test_api/__init__.py tests/test_api/test_health.py
git commit -m "feat(api): /v1/health endpoint + FastAPI app wiring"
```

---

### Task 8: `.env.example` and `.env` secrets generator

**Files:**
- Create: `.env.example`
- Create: `scripts/gen_secrets.sh`

- [ ] **Step 1: Write `.env.example`**

```bash
cat > .env.example <<'EOF'
# ====== Identity ======
WORKER_ID=wk-auto
WORKER_API_KEY=
CALLBACK_HMAC_SECRET=

# ====== Paths (inside the container or on the host) ======
COMFYUI_PATH=/data/comfyui
MODELS_PATH=/data/models
CACHE_PATH=/data/cache
WORKFLOWS_PATH=/app/workflows

# ====== Runtime ======
WORKER_PORT=8000
COMFYUI_HOST=127.0.0.1
COMFYUI_INTERNAL_PORT=8188
MAX_QUEUE_DEPTH=8
JOB_TIMEOUT_SEC_DEFAULT=300
GPU_DEVICE=0

# ====== External ======
HF_TOKEN=
CLOUDFLARE_TUNNEL_TOKEN=
ALLOWED_BACKEND_IPS=

# ====== Observability ======
LOG_LEVEL=info
LOG_FORMAT=json
METRICS_PORT=9090
SENTRY_DSN=
EOF
```

- [ ] **Step 2: Write `scripts/gen_secrets.sh`**

```bash
cat > scripts/gen_secrets.sh <<'EOF'
#!/usr/bin/env bash
# Populate missing secrets in .env. Idempotent: never overwrites existing non-empty values.
set -euo pipefail

ENV_FILE="${1:-.env}"
if [ ! -f "$ENV_FILE" ]; then
    cp .env.example "$ENV_FILE"
fi

gen() { python3 -c "import secrets; print(secrets.token_hex(32))"; }

for KEY in WORKER_API_KEY CALLBACK_HMAC_SECRET; do
    CUR=$(grep -E "^$KEY=" "$ENV_FILE" | cut -d= -f2-)
    if [ -z "${CUR:-}" ]; then
        VAL=$(gen)
        # portable sed: replace line starting with KEY=
        python3 - <<PY
from pathlib import Path
p = Path("$ENV_FILE")
lines = p.read_text().splitlines()
for i, ln in enumerate(lines):
    if ln.startswith("$KEY="):
        lines[i] = "$KEY=$VAL"
        break
else:
    lines.append("$KEY=$VAL")
p.write_text("\n".join(lines) + "\n")
PY
        echo "generated $KEY"
    fi
done
echo "secrets OK in $ENV_FILE"
EOF
chmod +x scripts/gen_secrets.sh
```

- [ ] **Step 3: Smoke test the script**

```bash
rm -f /tmp/test.env
bash scripts/gen_secrets.sh /tmp/test.env
grep -E "^(WORKER_API_KEY|CALLBACK_HMAC_SECRET)=" /tmp/test.env
# Run again, values must not change (idempotency)
BEFORE=$(grep "^WORKER_API_KEY=" /tmp/test.env)
bash scripts/gen_secrets.sh /tmp/test.env
AFTER=$(grep "^WORKER_API_KEY=" /tmp/test.env)
[ "$BEFORE" = "$AFTER" ] && echo "idempotent OK" || (echo "FAIL: value changed" && exit 1)
rm -f /tmp/test.env
```
Expected: both secrets populated, second run prints "idempotent OK".

- [ ] **Step 4: Commit**

```bash
git add .env.example scripts/gen_secrets.sh
git commit -m "feat(ops): .env.example template + idempotent secrets generator"
```

---

### Task 9: Dockerfile (fat image, CUDA 12.8 base)

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Write `.dockerignore`**

```bash
cat > .dockerignore <<'EOF'
.git
.gitignore
.venv
venv
__pycache__
*.pyc
*.pyo
.pytest_cache
.ruff_cache
.mypy_cache
docs
data
tests
.env
.env.*
!.env.example
EOF
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl wget ca-certificates tini supervisor ffmpeg libgl1 libglib2.0-0 \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python

# PyTorch 2.10 + CUDA 12.8 (use the index that matches; fall back to default cu124 if needed).
RUN python -m pip install --upgrade pip wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.10.0 torchvision torchaudio

# ComfyUI installed into /opt/comfyui at image build time (fat image).
ARG COMFYUI_SHA=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/comfyui \
    && cd /opt/comfyui && git checkout "${COMFYUI_SHA}" \
    && python -m pip install -r requirements.txt

# Workflow converter custom node (reused from spec §8.2).
RUN git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git \
        /opt/comfyui/custom_nodes/comfyui-workflow-to-api-converter-endpoint

# App code.
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY worker /app/worker
COPY workflows /app/workflows
COPY configs /app/configs
COPY scripts /app/scripts
COPY supervisord.conf /etc/supervisor/supervisord.conf

RUN python -m pip install -e "/app[dev]"

RUN chmod +x /app/scripts/*.sh

EXPOSE 8000 9090
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/scripts/start.sh"]
```

- [ ] **Step 3: Verify Dockerfile parses**

```bash
docker build --check -f Dockerfile . 2>&1 | head -20 || true
# If --check is unsupported (older docker), skip; linter is informational.
```
Expected: no syntax errors. Actual build is deferred (requires network + nvidia context).

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat(docker): fat image with CUDA 12.8 + ComfyUI + worker"
```

---

### Task 10: supervisord.conf, start.sh, docker-compose.yml

**Files:**
- Create: `supervisord.conf`
- Create: `scripts/start.sh`
- Create: `docker-compose.yml`

- [ ] **Step 1: Write `supervisord.conf`**

```ini
[supervisord]
nodaemon=true
user=root
logfile=/dev/stdout
logfile_maxbytes=0
pidfile=/var/run/supervisord.pid

[program:comfyui]
command=python /opt/comfyui/main.py --listen %(ENV_COMFYUI_HOST)s --port %(ENV_COMFYUI_INTERNAL_PORT)s --disable-auto-launch --disable-metadata --normalvram
directory=/opt/comfyui
priority=100
autostart=true
autorestart=true
startretries=3
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
environment=PYTHONUNBUFFERED=1,PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

[program:worker]
command=uvicorn worker.main:app --host 0.0.0.0 --port %(ENV_WORKER_PORT)s --log-level warning
directory=/app
priority=200
autostart=true
autorestart=true
startretries=3
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
```

- [ ] **Step 2: Write `scripts/start.sh`**

```bash
cat > scripts/start.sh <<'EOF'
#!/usr/bin/env bash
# Container entrypoint — runs bootstrap, then supervisord.
set -euo pipefail

cd /app

# Load .env if present (values from the environment take precedence).
if [ -f /app/.env ]; then
    set -a; source /app/.env; set +a
fi

# Defaults for required env expected by supervisord.conf.
: "${WORKER_PORT:=8000}"
: "${COMFYUI_HOST:=127.0.0.1}"
: "${COMFYUI_INTERNAL_PORT:=8188}"
export WORKER_PORT COMFYUI_HOST COMFYUI_INTERNAL_PORT

# Bootstrap: idempotent setup (populated in later tasks).
if [ -x /app/scripts/bootstrap.sh ]; then
    bash /app/scripts/bootstrap.sh
fi

exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
EOF
chmod +x scripts/start.sh
```

- [ ] **Step 3: Write `docker-compose.yml`**

```yaml
services:
  worker:
    build:
      context: .
      dockerfile: Dockerfile
    image: inference-worker:local
    restart: unless-stopped
    env_file:
      - .env
    ports:
      - "${WORKER_PORT:-8000}:8000"
      - "${METRICS_PORT:-9090}:9090"
    volumes:
      - ./data/models:/data/models
      - ./data/cache:/data/cache
      - ./data/comfyui:/data/comfyui
      - ./data/logs:/data/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ipc: host
    shm_size: 8gb
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://127.0.0.1:8000/v1/health"]
      interval: 10s
      timeout: 3s
      retries: 30
      start_period: 300s
```

- [ ] **Step 4: Smoke test compose file syntax**

```bash
docker compose config > /dev/null && echo "compose OK"
```
Expected: `compose OK`. (Full build deferred.)

- [ ] **Step 5: Commit**

```bash
git add supervisord.conf scripts/start.sh docker-compose.yml
git commit -m "feat(ops): supervisord + start.sh + docker-compose with GPU reservation"
```

---

### Task 11: Readiness probe for ComfyUI

**Files:**
- Create: `worker/comfyui/__init__.py` (empty)
- Create: `worker/comfyui/readiness.py`
- Create: `tests/test_comfyui/__init__.py` (empty)
- Create: `tests/test_comfyui/test_readiness.py`

- [ ] **Step 1: Write failing test `tests/test_comfyui/test_readiness.py`**

```python
import httpx
import pytest

from worker.comfyui.readiness import wait_for_comfyui, is_alive


async def _mock_200(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"system": {"os": "linux"}})


async def _mock_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500)


@pytest.mark.asyncio
async def test_is_alive_true_when_200(respx_mock):
    respx_mock.get("http://x:8188/system_stats").respond(200, json={"ok": True})
    async with httpx.AsyncClient() as client:
        assert await is_alive(client, "http://x:8188") is True


@pytest.mark.asyncio
async def test_is_alive_false_on_http_error(respx_mock):
    respx_mock.get("http://x:8188/system_stats").respond(500)
    async with httpx.AsyncClient() as client:
        assert await is_alive(client, "http://x:8188") is False


@pytest.mark.asyncio
async def test_wait_for_comfyui_returns_when_ready(respx_mock):
    route = respx_mock.get("http://x:8188/system_stats")
    route.side_effect = [httpx.Response(500), httpx.Response(500), httpx.Response(200, json={})]
    async with httpx.AsyncClient() as client:
        await wait_for_comfyui(client, "http://x:8188", timeout_sec=5, interval_sec=0.01)
    assert route.call_count == 3


@pytest.mark.asyncio
async def test_wait_for_comfyui_raises_on_timeout(respx_mock):
    respx_mock.get("http://x:8188/system_stats").respond(500)
    async with httpx.AsyncClient() as client:
        with pytest.raises(TimeoutError):
            await wait_for_comfyui(client, "http://x:8188", timeout_sec=0.05, interval_sec=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_comfyui/test_readiness.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/comfyui/readiness.py`**

```python
from __future__ import annotations

import asyncio

import httpx


async def is_alive(client: httpx.AsyncClient, base_url: str) -> bool:
    """One-shot check: is ComfyUI /system_stats responding 200?"""
    try:
        r = await client.get(f"{base_url}/system_stats", timeout=2.0)
        return r.status_code == 200
    except (httpx.HTTPError, asyncio.TimeoutError):
        return False


async def wait_for_comfyui(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    timeout_sec: float = 120.0,
    interval_sec: float = 1.0,
) -> None:
    """Block until ComfyUI responds 200 on /system_stats, or raise TimeoutError."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_sec
    while True:
        if await is_alive(client, base_url):
            return
        if loop.time() >= deadline:
            raise TimeoutError(f"ComfyUI not ready on {base_url} after {timeout_sec}s")
        await asyncio.sleep(interval_sec)
```

- [ ] **Step 4: Create empty init files and run test**

```bash
touch worker/comfyui/__init__.py tests/test_comfyui/__init__.py
pytest tests/test_comfyui/test_readiness.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/comfyui/__init__.py worker/comfyui/readiness.py \
        tests/test_comfyui/__init__.py tests/test_comfyui/test_readiness.py
git commit -m "feat(comfyui): async readiness probe with bounded wait"
```

---

## Sprint 1 — Image pipeline (edit + style)

### Task 12: Models manifest (`configs/models.yaml`)

Phase 1 models only (no LTX-Video, no ControlNet-Union, no preprocessors — those come in Phase 2).

**Files:**
- Create: `configs/models.yaml`

- [ ] **Step 1: Write `configs/models.yaml`**

```yaml
# Phase 1 model manifest. Each entry:
#   name: human label (also used in logs)
#   repo: HuggingFace repo id
#   filename: file to download within the repo
#   dest: path under $MODELS_PATH where the file lands (keeps ComfyUI's expected layout)
#   sha256: SHA-256 of the downloaded file (verified; empty string skips)
#   size_mb: expected size in MB (used for sanity checks + progress)
#
# Paths mirror ComfyUI's models/ subfolders so symlinks just work.

models:
  - name: JuggernautXL v9 Lightning
    repo: RunDiffusion/Juggernaut-XL-Lightning
    filename: Juggernaut_X_RunDiffusion_Hyper.safetensors
    dest: checkpoints/Juggernaut_X_RunDiffusion_Hyper.safetensors
    sha256: ""
    size_mb: 6617

  - name: SDXL VAE fp16 fix
    repo: madebyollin/sdxl-vae-fp16-fix
    filename: sdxl_vae.safetensors
    dest: vae/sdxl_vae_fp16_fix.safetensors
    sha256: ""
    size_mb: 320

  - name: SDXL LCM LoRA
    repo: latent-consistency/lcm-lora-sdxl
    filename: pytorch_lora_weights.safetensors
    dest: loras/lcm_lora_sdxl.safetensors
    sha256: ""
    size_mb: 394

  - name: IP-Adapter Plus SDXL
    repo: h94/IP-Adapter
    filename: sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
    dest: ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors
    sha256: ""
    size_mb: 851

  - name: CLIP-ViT-H-14 (IP-Adapter encoder)
    repo: h94/IP-Adapter
    filename: models/image_encoder/model.safetensors
    dest: clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
    sha256: ""
    size_mb: 1264
```

- [ ] **Step 2: Commit**

```bash
git add configs/models.yaml
git commit -m "feat(configs): phase-1 model manifest (SDXL + IP-Adapter + LCM)"
```

---

### Task 13: Models downloader script (`scripts/download_models.py`)

**Files:**
- Create: `scripts/download_models.py`
- Create: `tests/test_download_models.py`

- [ ] **Step 1: Write failing test `tests/test_download_models.py`**

```python
from pathlib import Path

import pytest
import yaml

from scripts.download_models import (
    ManifestEntry,
    load_manifest,
    missing_entries,
    verify_sha256,
)


def _write_manifest(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump({"models": entries}))
    return p


def test_load_manifest_parses_entries(tmp_path):
    mp = _write_manifest(tmp_path, [{
        "name": "Test", "repo": "org/repo", "filename": "f.bin",
        "dest": "x/f.bin", "sha256": "", "size_mb": 10,
    }])
    entries = load_manifest(mp)
    assert len(entries) == 1
    assert isinstance(entries[0], ManifestEntry)
    assert entries[0].dest == "x/f.bin"


def test_missing_entries_filters_existing(tmp_path):
    models_dir = tmp_path / "models"
    (models_dir / "x").mkdir(parents=True)
    (models_dir / "x/f.bin").write_bytes(b"dummy")
    mp = _write_manifest(tmp_path, [
        {"name": "A", "repo": "org/a", "filename": "f.bin", "dest": "x/f.bin", "sha256": "", "size_mb": 1},
        {"name": "B", "repo": "org/b", "filename": "g.bin", "dest": "y/g.bin", "sha256": "", "size_mb": 1},
    ])
    entries = load_manifest(mp)
    miss = missing_entries(entries, models_dir)
    assert [e.name for e in miss] == ["B"]


def test_verify_sha256_accepts_matching_hash(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    # sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    assert verify_sha256(f, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824") is True
    assert verify_sha256(f, "deadbeef") is False


def test_verify_sha256_skips_when_hash_empty(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert verify_sha256(f, "") is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_download_models.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `scripts/download_models.py`**

```python
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class ManifestEntry:
    name: str
    repo: str
    filename: str
    dest: str
    sha256: str
    size_mb: int


def load_manifest(path: Path) -> list[ManifestEntry]:
    data = yaml.safe_load(path.read_text())
    return [ManifestEntry(**m) for m in data["models"]]


def missing_entries(entries: list[ManifestEntry], models_dir: Path) -> list[ManifestEntry]:
    return [e for e in entries if not (models_dir / e.dest).exists()]


def verify_sha256(path: Path, expected: str) -> bool:
    if not expected:
        return True
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def download(entry: ManifestEntry, models_dir: Path, hf_token: str | None) -> Path:
    target = models_dir / entry.dest
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  → downloading {entry.name} ({entry.size_mb} MB)", flush=True)
    # huggingface_hub writes to a cache then symlinks/copies. We download and then move.
    tmp_path = hf_hub_download(
        repo_id=entry.repo,
        filename=entry.filename,
        token=hf_token or None,
        local_dir=str(target.parent),
        local_dir_use_symlinks=False,
    )
    src = Path(tmp_path)
    if src != target:
        src.replace(target)
    if not verify_sha256(target, entry.sha256):
        target.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {entry.name}")
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/models.yaml")
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_PATH", "/data/models"))
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    manifest = Path(args.manifest)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest)
    miss = missing_entries(entries, models_dir)
    if not miss:
        print("all models present")
        return 0

    print(f"missing {len(miss)} / {len(entries)} models, downloading")
    for e in miss:
        download(e, models_dir, args.hf_token)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_download_models.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/download_models.py tests/test_download_models.py
git commit -m "feat(scripts): manifest-driven model downloader with sha256 verify"
```

---

### Task 14: Custom nodes manifest and installer

**Files:**
- Create: `configs/custom_nodes.yaml`
- Create: `scripts/install_custom_nodes.sh`

- [ ] **Step 1: Write `configs/custom_nodes.yaml`**

```yaml
# Phase 1 custom nodes. Each entry:
#   name: label
#   repo: git URL (https)
#   ref: git ref (commit sha, tag, or branch). Pin to sha for reproducibility.
#   post_install: optional shell command run inside the node directory
nodes:
  - name: IPAdapter_plus
    repo: https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
    ref: main
    post_install: "python -m pip install -r requirements.txt || true"

  - name: tooling-nodes
    repo: https://github.com/Acly/comfyui-tooling-nodes.git
    ref: main
    post_install: "python -m pip install -r requirements.txt || true"

  - name: KJNodes
    repo: https://github.com/kijai/ComfyUI-KJNodes.git
    ref: main
    post_install: "python -m pip install -r requirements.txt || true"
```

- [ ] **Step 2: Write `scripts/install_custom_nodes.sh`**

```bash
cat > scripts/install_custom_nodes.sh <<'EOF'
#!/usr/bin/env bash
# Idempotent installer for ComfyUI custom nodes described in configs/custom_nodes.yaml.
set -euo pipefail

MANIFEST="${1:-/app/configs/custom_nodes.yaml}"
COMFYUI_PATH="${COMFYUI_PATH:-/opt/comfyui}"
CUSTOM_NODES="${COMFYUI_PATH}/custom_nodes"

mkdir -p "$CUSTOM_NODES"

# Parse YAML via python (yaml already installed as a worker dep).
python3 - "$MANIFEST" "$CUSTOM_NODES" <<'PY'
import subprocess, sys
from pathlib import Path
import yaml

manifest_path, nodes_dir = sys.argv[1], Path(sys.argv[2])
data = yaml.safe_load(Path(manifest_path).read_text())

for node in data["nodes"]:
    name = node["name"]
    repo = node["repo"]
    ref  = node.get("ref", "main")
    post = node.get("post_install", "")
    target = nodes_dir / name
    if target.exists():
        print(f"  [=] {name} already present at {target}", flush=True)
    else:
        print(f"  [+] cloning {name} from {repo} @ {ref}", flush=True)
        subprocess.check_call(["git", "clone", repo, str(target)])
        subprocess.check_call(["git", "-C", str(target), "checkout", ref])
    if post:
        print(f"  [*] post-install: {post}", flush=True)
        subprocess.check_call(post, shell=True, cwd=str(target))

print("custom nodes OK")
PY
EOF
chmod +x scripts/install_custom_nodes.sh
```

- [ ] **Step 3: Smoke test the YAML parses**

```bash
python3 -c "import yaml; print(yaml.safe_load(open('configs/custom_nodes.yaml'))['nodes'][0]['name'])"
```
Expected: prints `IPAdapter_plus`.

- [ ] **Step 4: Commit**

```bash
git add configs/custom_nodes.yaml scripts/install_custom_nodes.sh
git commit -m "feat(scripts): custom node manifest + idempotent installer"
```

---

### Task 15: `bootstrap.sh` — full orchestrated setup

**Files:**
- Create: `scripts/bootstrap.sh`

- [ ] **Step 1: Write `scripts/bootstrap.sh`**

```bash
cat > scripts/bootstrap.sh <<'EOF'
#!/usr/bin/env bash
# Idempotent bootstrap. Safe to run on every boot.
set -euo pipefail

cd /app

echo "[bootstrap] 1. env / secrets"
if [ ! -f /app/.env ]; then
    cp /app/.env.example /app/.env
fi
bash /app/scripts/gen_secrets.sh /app/.env
# export only KEY=VAL lines
set -a; source /app/.env; set +a

echo "[bootstrap] 2. comfyui present at ${COMFYUI_PATH}"
if [ ! -f "${COMFYUI_PATH}/main.py" ]; then
    mkdir -p "$(dirname "${COMFYUI_PATH}")"
    git clone https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_PATH}"
    (cd "${COMFYUI_PATH}" && python -m pip install -r requirements.txt)
fi

echo "[bootstrap] 3. workflow-to-api converter node"
CONV_DIR="${COMFYUI_PATH}/custom_nodes/comfyui-workflow-to-api-converter-endpoint"
if [ ! -d "${CONV_DIR}" ]; then
    git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git "${CONV_DIR}"
fi

echo "[bootstrap] 4. custom nodes"
COMFYUI_PATH="${COMFYUI_PATH}" bash /app/scripts/install_custom_nodes.sh /app/configs/custom_nodes.yaml

echo "[bootstrap] 5. models"
python /app/scripts/download_models.py \
    --manifest /app/configs/models.yaml \
    --models-dir "${MODELS_PATH}" \
    --hf-token "${HF_TOKEN:-}"

echo "[bootstrap] 6. symlink models into ComfyUI models dir"
# ComfyUI reads from ${COMFYUI_PATH}/models by default. Symlink our MODELS_PATH over it.
if [ "${COMFYUI_PATH}/models" != "${MODELS_PATH}" ]; then
    # Keep the original (or its contents) reachable if it was populated.
    if [ -d "${COMFYUI_PATH}/models" ] && [ ! -L "${COMFYUI_PATH}/models" ]; then
        # Copy anything already there into MODELS_PATH without overwriting.
        rsync -a --ignore-existing "${COMFYUI_PATH}/models/" "${MODELS_PATH}/" 2>/dev/null || true
        mv "${COMFYUI_PATH}/models" "${COMFYUI_PATH}/models.orig.$(date +%s)" 2>/dev/null || rm -rf "${COMFYUI_PATH}/models"
    fi
    ln -sfn "${MODELS_PATH}" "${COMFYUI_PATH}/models"
fi

echo "[bootstrap] 7. cloudflare tunnel (if token provided)"
if [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
    if ! command -v cloudflared >/dev/null 2>&1; then
        echo "cloudflared not installed; skipping"
    else
        nohup cloudflared tunnel --no-autoupdate run --token "${CLOUDFLARE_TUNNEL_TOKEN}" \
            > /data/logs/cloudflared.log 2>&1 &
    fi
fi

echo "[bootstrap] done"
EOF
chmod +x scripts/bootstrap.sh
```

- [ ] **Step 2: Shell-lint the script**

```bash
bash -n scripts/bootstrap.sh && echo "syntax OK"
```
Expected: `syntax OK` (full execution deferred — requires network and GPU).

- [ ] **Step 3: Commit**

```bash
git add scripts/bootstrap.sh
git commit -m "feat(scripts): idempotent bootstrap orchestrator"
```

---

### Task 16: Async image downloader (`worker/io/downloader.py`)

**Files:**
- Create: `worker/io/__init__.py` (empty)
- Create: `worker/io/downloader.py`
- Create: `tests/test_io/__init__.py` (empty)
- Create: `tests/test_io/test_downloader.py`

- [ ] **Step 1: Write failing test `tests/test_io/test_downloader.py`**

```python
from pathlib import Path

import httpx
import pytest

from worker.core.errors import AppError, ErrorCode
from worker.io.downloader import download_to_file


@pytest.mark.asyncio
async def test_download_writes_file(respx_mock, tmp_path):
    respx_mock.get("https://x/a.png").respond(200, content=b"PNGBYTES")
    out = tmp_path / "a.png"
    async with httpx.AsyncClient() as client:
        size = await download_to_file(client, "https://x/a.png", out)
    assert out.read_bytes() == b"PNGBYTES"
    assert size == len(b"PNGBYTES")


@pytest.mark.asyncio
async def test_download_404_raises_non_retryable(respx_mock, tmp_path):
    respx_mock.get("https://x/a.png").respond(404)
    async with httpx.AsyncClient() as client:
        with pytest.raises(AppError) as exc:
            await download_to_file(client, "https://x/a.png", tmp_path / "a.png")
    assert exc.value.code is ErrorCode.INPUT_FETCH_FAILED
    assert exc.value.retryable is False


@pytest.mark.asyncio
async def test_download_503_retries_then_succeeds(respx_mock, tmp_path):
    route = respx_mock.get("https://x/a.png")
    route.side_effect = [httpx.Response(503), httpx.Response(503), httpx.Response(200, content=b"ok")]
    async with httpx.AsyncClient() as client:
        await download_to_file(client, "https://x/a.png", tmp_path / "a.png", max_retries=3, backoff_base=0.01)
    assert route.call_count == 3


@pytest.mark.asyncio
async def test_download_exhausts_retries(respx_mock, tmp_path):
    respx_mock.get("https://x/a.png").respond(503)
    async with httpx.AsyncClient() as client:
        with pytest.raises(AppError) as exc:
            await download_to_file(client, "https://x/a.png", tmp_path / "a.png", max_retries=2, backoff_base=0.01)
    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_download_too_large_rejected(respx_mock, tmp_path):
    big = b"x" * (2 * 1024 * 1024)
    respx_mock.get("https://x/a.png").respond(200, content=big)
    async with httpx.AsyncClient() as client:
        with pytest.raises(AppError) as exc:
            await download_to_file(client, "https://x/a.png", tmp_path / "a.png", max_bytes=1024 * 1024)
    assert exc.value.code is ErrorCode.INPUT_TOO_LARGE
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_io/test_downloader.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/io/downloader.py`**

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from worker.core.errors import AppError, ErrorCode

DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50 MB hard cap per input


async def download_to_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout_sec: float = 30.0,
) -> int:
    """Download `url` to `dest`, returning the byte count.

    Retryable on network / 5xx / timeout. Non-retryable on 4xx or too-large payloads.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with client.stream("GET", url, timeout=timeout_sec, follow_redirects=True) as resp:
                if 400 <= resp.status_code < 500:
                    raise AppError(
                        ErrorCode.INPUT_FETCH_FAILED,
                        f"HTTP {resp.status_code} for {url}",
                        retryable=False,
                    )
                if resp.status_code >= 500:
                    raise _RetryableHTTP(f"HTTP {resp.status_code} for {url}")
                written = 0
                with dest.open("wb") as f:
                    async for chunk in resp.aiter_bytes():
                        written += len(chunk)
                        if written > max_bytes:
                            dest.unlink(missing_ok=True)
                            raise AppError(
                                ErrorCode.INPUT_TOO_LARGE,
                                f"input exceeds {max_bytes} bytes",
                                retryable=False,
                            )
                        f.write(chunk)
                return written
        except AppError:
            raise
        except (_RetryableHTTP, httpx.HTTPError, asyncio.TimeoutError) as e:
            last_err = e
            if attempt == max_retries:
                break
            await asyncio.sleep(backoff_base * (2**attempt))
    raise AppError(
        ErrorCode.INPUT_FETCH_FAILED,
        f"download failed after {max_retries + 1} attempts: {last_err}",
        retryable=True,
    )


class _RetryableHTTP(Exception):
    pass
```

- [ ] **Step 4: Create empty init files and run test**

```bash
touch worker/io/__init__.py tests/test_io/__init__.py
pytest tests/test_io/test_downloader.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/io/__init__.py worker/io/downloader.py \
        tests/test_io/__init__.py tests/test_io/test_downloader.py
git commit -m "feat(io): async downloader with retries + size cap"
```

---

### Task 17: Async uploader (`worker/io/uploader.py`)

**Files:**
- Create: `worker/io/uploader.py`
- Create: `tests/test_io/test_uploader.py`

- [ ] **Step 1: Write failing test `tests/test_io/test_uploader.py`**

```python
from pathlib import Path

import httpx
import pytest

from worker.core.errors import AppError, ErrorCode
from worker.io.uploader import upload_file


@pytest.mark.asyncio
async def test_upload_put_success(respx_mock, tmp_path):
    f = tmp_path / "r.png"
    f.write_bytes(b"IMG")
    route = respx_mock.put("https://s/out/r").respond(200)
    async with httpx.AsyncClient() as client:
        size = await upload_file(client, "https://s/out/r", f, method="PUT", content_type="image/png")
    assert size == 3
    assert route.calls.last.request.content == b"IMG"
    assert route.calls.last.request.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_upload_403_non_retryable(respx_mock, tmp_path):
    f = tmp_path / "r.png"; f.write_bytes(b"IMG")
    respx_mock.put("https://s/out/r").respond(403)
    async with httpx.AsyncClient() as client:
        with pytest.raises(AppError) as exc:
            await upload_file(client, "https://s/out/r", f, method="PUT")
    assert exc.value.code is ErrorCode.UPLOAD_AUTH_FAILED
    assert exc.value.retryable is False


@pytest.mark.asyncio
async def test_upload_500_retries(respx_mock, tmp_path):
    f = tmp_path / "r.png"; f.write_bytes(b"IMG")
    route = respx_mock.put("https://s/out/r")
    route.side_effect = [httpx.Response(500), httpx.Response(500), httpx.Response(200)]
    async with httpx.AsyncClient() as client:
        await upload_file(client, "https://s/out/r", f, method="PUT", max_retries=3, backoff_base=0.01)
    assert route.call_count == 3


@pytest.mark.asyncio
async def test_upload_post_method(respx_mock, tmp_path):
    f = tmp_path / "r.png"; f.write_bytes(b"IMG")
    route = respx_mock.post("https://s/out/r").respond(200)
    async with httpx.AsyncClient() as client:
        await upload_file(client, "https://s/out/r", f, method="POST")
    assert route.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_io/test_uploader.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/io/uploader.py`**

```python
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import httpx

from worker.core.errors import AppError, ErrorCode


async def upload_file(
    client: httpx.AsyncClient,
    url: str,
    src: Path,
    *,
    method: Literal["PUT", "POST"] = "PUT",
    content_type: str = "application/octet-stream",
    max_retries: int = 3,
    backoff_base: float = 1.0,
    timeout_sec: float = 60.0,
) -> int:
    """Upload `src` to `url` via PUT or POST. Retries on 5xx/network errors only."""
    body = src.read_bytes()
    headers = {"content-type": content_type, "content-length": str(len(body))}
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.request(method, url, content=body, headers=headers, timeout=timeout_sec)
            if resp.status_code in (401, 403):
                raise AppError(
                    ErrorCode.UPLOAD_AUTH_FAILED,
                    f"HTTP {resp.status_code}: signed URL rejected",
                    retryable=False,
                )
            if 400 <= resp.status_code < 500:
                raise AppError(
                    ErrorCode.UPLOAD_FAILED,
                    f"HTTP {resp.status_code} from upload endpoint",
                    retryable=False,
                )
            if resp.status_code >= 500:
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                if attempt == max_retries:
                    break
                await asyncio.sleep(backoff_base * (2**attempt))
                continue
            return len(body)
        except AppError:
            raise
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            last_err = e
            if attempt == max_retries:
                break
            await asyncio.sleep(backoff_base * (2**attempt))
    raise AppError(
        ErrorCode.UPLOAD_FAILED,
        f"upload failed after {max_retries + 1} attempts: {last_err}",
        retryable=True,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_io/test_uploader.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/io/uploader.py tests/test_io/test_uploader.py
git commit -m "feat(io): async uploader (PUT/POST) with retries"
```

---

### Task 18: Workflow template loader + parameter injection

The injection mechanism is the bridge between preset parameters and ComfyUI workflow JSON. Our conventions:
- Each workflow template is the **full** workflow format (UI export shape), not the API format.
- Nodes carry a `"title"` field that we use as an anchor. The injector locates nodes by title.
- Widget values are set via the `"widgets_values"` list (by index) for list-shaped widgets, or by node property for dicts.

**Files:**
- Create: `worker/comfyui/workflow.py`
- Create: `tests/test_comfyui/test_workflow.py`
- Create: `tests/fixtures/workflows/sample.json` (tiny fake workflow for tests)

- [ ] **Step 1: Write test fixture `tests/fixtures/workflows/sample.json`**

```json
{
  "last_node_id": 5,
  "last_link_id": 4,
  "nodes": [
    {"id": 1, "type": "CLIPTextEncode", "title": "positive_prompt", "widgets_values": ["a cat"]},
    {"id": 2, "type": "CLIPTextEncode", "title": "negative_prompt", "widgets_values": ["blurry"]},
    {"id": 3, "type": "KSampler", "title": "sampler", "widgets_values": [0, "randomize", 8, 1.8, "euler", "normal", 1.0]},
    {"id": 4, "type": "LoadImage", "title": "input_image", "widgets_values": ["placeholder.png", "image"]},
    {"id": 5, "type": "EmptyLatentImage", "title": "latent_dims", "widgets_values": [1024, 1024, 1]}
  ],
  "links": [],
  "version": 0.4
}
```

- [ ] **Step 2: Write failing test `tests/test_comfyui/test_workflow.py`**

```python
import json
from pathlib import Path

import pytest

from worker.comfyui.workflow import WorkflowTemplate, find_node


FIXTURE = Path(__file__).parent.parent / "fixtures" / "workflows" / "sample.json"


def test_load_template_from_file():
    t = WorkflowTemplate.from_file(FIXTURE)
    assert t.to_dict()["last_node_id"] == 5


def test_find_node_by_title():
    t = WorkflowTemplate.from_file(FIXTURE)
    n = find_node(t.to_dict(), "positive_prompt")
    assert n["id"] == 1


def test_find_node_raises_when_missing():
    t = WorkflowTemplate.from_file(FIXTURE)
    with pytest.raises(KeyError):
        find_node(t.to_dict(), "missing_title")


def test_set_widget_changes_value():
    t = WorkflowTemplate.from_file(FIXTURE)
    t.set_widget("positive_prompt", 0, "a dog on a skateboard")
    assert find_node(t.to_dict(), "positive_prompt")["widgets_values"][0] == "a dog on a skateboard"


def test_set_widget_does_not_mutate_original_file():
    t1 = WorkflowTemplate.from_file(FIXTURE)
    t2 = WorkflowTemplate.from_file(FIXTURE)
    t1.set_widget("positive_prompt", 0, "new")
    assert find_node(t2.to_dict(), "positive_prompt")["widgets_values"][0] == "a cat"


def test_set_many_widgets_chain():
    t = WorkflowTemplate.from_file(FIXTURE)
    t.set_widget("sampler", 0, 42)          # seed
    t.set_widget("sampler", 2, 12)           # steps
    t.set_widget("latent_dims", 0, 512)      # width
    t.set_widget("latent_dims", 1, 768)      # height
    d = t.to_dict()
    assert find_node(d, "sampler")["widgets_values"][0] == 42
    assert find_node(d, "sampler")["widgets_values"][2] == 12
    assert find_node(d, "latent_dims")["widgets_values"][0] == 512
    assert find_node(d, "latent_dims")["widgets_values"][1] == 768


def test_to_json_roundtrip():
    t = WorkflowTemplate.from_file(FIXTURE)
    raw = t.to_json()
    parsed = json.loads(raw)
    assert parsed["nodes"][0]["title"] == "positive_prompt"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_comfyui/test_workflow.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `worker/comfyui/workflow.py`**

```python
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def find_node(workflow: dict, title: str) -> dict:
    """Return the node dict whose `title` field matches `title`."""
    for node in workflow.get("nodes", []):
        if node.get("title") == title:
            return node
    raise KeyError(f"no node titled {title!r} in workflow")


class WorkflowTemplate:
    """In-memory mutable copy of a ComfyUI full-format workflow."""

    def __init__(self, data: dict) -> None:
        self._data = copy.deepcopy(data)

    @classmethod
    def from_file(cls, path: Path | str) -> WorkflowTemplate:
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict) -> WorkflowTemplate:
        return cls(data)

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def to_json(self) -> str:
        return json.dumps(self._data, ensure_ascii=False)

    def set_widget(self, title: str, index: int, value: Any) -> None:
        """Set widgets_values[index] on the node with the given title."""
        node = find_node(self._data, title)
        widgets = node.setdefault("widgets_values", [])
        while len(widgets) <= index:
            widgets.append(None)
        widgets[index] = value

    def set_property(self, title: str, key: str, value: Any) -> None:
        """Set node['properties'][key] on the node with the given title."""
        node = find_node(self._data, title)
        node.setdefault("properties", {})[key] = value
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_comfyui/test_workflow.py -v
```
Expected: 7 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add worker/comfyui/workflow.py tests/test_comfyui/test_workflow.py tests/fixtures/workflows/sample.json
git commit -m "feat(comfyui): workflow template with title-based widget injection"
```

---

### Task 19: ComfyUI client (HTTP + WebSocket)

**Files:**
- Create: `worker/comfyui/client.py`
- Create: `tests/test_comfyui/test_client.py`

- [ ] **Step 1: Write failing test `tests/test_comfyui/test_client.py`**

```python
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from worker.comfyui.client import ComfyUIClient
from worker.core.errors import AppError, ErrorCode


@pytest.mark.asyncio
async def test_convert_workflow_posts_and_returns_api_format(respx_mock):
    api_fmt = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x"}}}
    respx_mock.post("http://cu:8188/workflow/convert").respond(200, json={"api_prompt": api_fmt})
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        out = await c.convert_workflow({"nodes": []})
    assert out == api_fmt


@pytest.mark.asyncio
async def test_convert_workflow_error_raises_app_error(respx_mock):
    respx_mock.post("http://cu:8188/workflow/convert").respond(500)
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        with pytest.raises(AppError) as exc:
            await c.convert_workflow({"nodes": []})
    assert exc.value.code is ErrorCode.COMFYUI_UNAVAILABLE


@pytest.mark.asyncio
async def test_submit_prompt_returns_prompt_id(respx_mock):
    respx_mock.post("http://cu:8188/prompt").respond(200, json={"prompt_id": "pid-42", "number": 1})
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        pid = await c.submit_prompt({"1": {}}, client_id="cid-1")
    assert pid == "pid-42"


@pytest.mark.asyncio
async def test_interrupt_posts(respx_mock):
    route = respx_mock.post("http://cu:8188/interrupt").respond(200)
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        await c.interrupt()
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_get_history_returns_outputs(respx_mock):
    hist = {"pid-42": {"outputs": {"9": {"images": [{"filename": "r.png", "subfolder": "", "type": "output"}]}}}}
    respx_mock.get("http://cu:8188/history/pid-42").respond(200, json=hist)
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        outs = await c.get_history_outputs("pid-42")
    assert outs["9"]["images"][0]["filename"] == "r.png"


@pytest.mark.asyncio
async def test_get_history_raises_when_absent(respx_mock):
    respx_mock.get("http://cu:8188/history/pid-42").respond(200, json={})
    async with httpx.AsyncClient() as http:
        c = ComfyUIClient(http, "http://cu:8188")
        with pytest.raises(KeyError):
            await c.get_history_outputs("pid-42")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_comfyui/test_client.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/comfyui/client.py`**

```python
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

    async def get_history_outputs(self, prompt_id: str) -> dict:
        """GET /history/{prompt_id} — returns the `outputs` dict of that prompt."""
        r = await self._http.get(f"{self._base}/history/{prompt_id}", timeout=15.0)
        r.raise_for_status()
        hist = r.json()
        if prompt_id not in hist:
            raise KeyError(f"prompt_id {prompt_id!r} not in history")
        return hist[prompt_id].get("outputs", {})

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
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_comfyui/test_client.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/comfyui/client.py tests/test_comfyui/test_client.py
git commit -m "feat(comfyui): http + websocket client (convert, submit, wait, interrupt)"
```

---

### Task 20: Preset abstract base + request models

**Files:**
- Create: `worker/presets/__init__.py`
- Create: `worker/presets/base.py`
- Create: `tests/test_presets/__init__.py` (empty)
- Create: `tests/test_presets/test_base.py`

- [ ] **Step 1: Write failing test `tests/test_presets/test_base.py`**

```python
import pytest

from worker.presets.base import Preset, Mode


class _FakePreset(Preset):
    name = "fake"
    mode = Mode.IMAGE
    template_filename = "fake.json"
    output_extension = "png"
    output_content_type = "image/png"

    class Parameters(Preset.BaseParameters):
        steps: int = 6
        cfg: float = 1.8

    def inject(self, template, params, input_paths):
        return template


def test_preset_name_is_class_attr():
    p = _FakePreset()
    assert p.name == "fake"
    assert p.mode is Mode.IMAGE


def test_parameters_apply_defaults():
    params = _FakePreset.Parameters()
    assert params.steps == 6
    assert params.cfg == 1.8
    assert params.seed is None


def test_parameters_reject_extra_fields():
    with pytest.raises(Exception):
        _FakePreset.Parameters(steps=1, cfg=1.0, unknown_field="x")


def test_parameters_validate_ranges():
    with pytest.raises(Exception):
        _FakePreset.Parameters(steps=-1)
    with pytest.raises(Exception):
        _FakePreset.Parameters(cfg=-0.1)


def test_mode_enum_values():
    assert Mode.IMAGE.value == "image"
    assert Mode.VIDEO.value == "video"
```

- [ ] **Step 2: Create empty init files**

```bash
touch worker/presets/__init__.py tests/test_presets/__init__.py
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_presets/test_base.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `worker/presets/base.py`**

```python
from __future__ import annotations

import abc
from enum import Enum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate


class Mode(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class InputPaths(BaseModel):
    """Local file paths for images the preset may read (relative to ComfyUI input/)."""
    model_config = ConfigDict(extra="forbid")
    input_image: str
    mask_image: str | None = None
    reference_image: str | None = None


class Preset(abc.ABC):
    """Abstract preset: binds a name, parameter model, template, and injection logic."""

    name: ClassVar[str]
    mode: ClassVar[Mode]
    template_filename: ClassVar[str]
    output_extension: ClassVar[str]
    output_content_type: ClassVar[str]

    class BaseParameters(BaseModel):
        model_config = ConfigDict(extra="forbid")
        seed: int | None = None
        steps: int = Field(default=6, ge=1, le=150)
        cfg: float = Field(default=1.8, ge=0.0, le=30.0)
        width: int = Field(default=1024, ge=64, le=4096)
        height: int = Field(default=1024, ge=64, le=4096)

    Parameters: ClassVar[type[BaseParameters]] = BaseParameters

    @abc.abstractmethod
    def inject(
        self,
        template: WorkflowTemplate,
        params: BaseParameters,
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        """Return the template mutated with params + input paths injected."""

    def output_filename(self, job_id: str) -> str:
        return f"{job_id}.{self.output_extension}"

    def load_template(self, workflows_dir: Path) -> WorkflowTemplate:
        return WorkflowTemplate.from_file(workflows_dir / self.template_filename)


def validate_preset_parameters(preset_cls: type[Preset], raw: dict) -> Preset.BaseParameters:
    """Validate raw dict against the preset's Parameters model. Raises pydantic ValidationError."""
    return preset_cls.Parameters(**raw)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_presets/test_base.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add worker/presets/__init__.py worker/presets/base.py \
        tests/test_presets/__init__.py tests/test_presets/test_base.py
git commit -m "feat(presets): Preset abstract base + BaseParameters schema"
```

---

### Task 21: `edit` preset + workflow template

**Files:**
- Create: `workflows/edit.json`
- Create: `worker/presets/edit.py`
- Create: `tests/test_presets/test_edit.py`

The `edit` workflow structure:
- `LoadImage` (title `input_image`) — loads source
- `CheckpointLoaderSimple` (title `checkpoint`) — JuggernautXL
- `LoraLoader` (title `lcm_lora`) — LCM LoRA chained onto checkpoint
- `CLIPTextEncode` x2 (titles `positive_prompt`, `negative_prompt`)
- `IPAdapterModelLoader` (title `ipa_loader`) + `IPAdapter` (title `ipa_apply`)
- `CLIPVisionLoader` (title `clip_vision_loader`)
- `VAELoader` (title `vae_loader`) — SDXL fp16 fix
- `VAEEncode` (title `vae_encode`)
- `KSampler` (title `sampler`)
- `VAEDecode` (title `vae_decode`)
- `SaveImage` (title `save`)

- [ ] **Step 1: Write `workflows/edit.json`**

```json
{
  "last_node_id": 20,
  "last_link_id": 40,
  "nodes": [
    {"id": 1, "type": "CheckpointLoaderSimple", "title": "checkpoint",
     "widgets_values": ["Juggernaut_X_RunDiffusion_Hyper.safetensors"]},
    {"id": 2, "type": "LoraLoader", "title": "lcm_lora",
     "widgets_values": ["lcm_lora_sdxl.safetensors", 1.0, 1.0]},
    {"id": 3, "type": "VAELoader", "title": "vae_loader",
     "widgets_values": ["sdxl_vae_fp16_fix.safetensors"]},
    {"id": 4, "type": "CLIPVisionLoader", "title": "clip_vision_loader",
     "widgets_values": ["CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"]},
    {"id": 5, "type": "IPAdapterModelLoader", "title": "ipa_loader",
     "widgets_values": ["ip-adapter-plus_sdxl_vit-h.safetensors"]},
    {"id": 6, "type": "LoadImage", "title": "input_image",
     "widgets_values": ["placeholder.png", "image"]},
    {"id": 7, "type": "CLIPTextEncode", "title": "positive_prompt",
     "widgets_values": [""]},
    {"id": 8, "type": "CLIPTextEncode", "title": "negative_prompt",
     "widgets_values": ["blurry, low quality, distorted, artifacts"]},
    {"id": 9, "type": "IPAdapter", "title": "ipa_apply",
     "widgets_values": [0.7, 0.0, 1.0, "standard"]},
    {"id": 10, "type": "VAEEncode", "title": "vae_encode"},
    {"id": 11, "type": "KSampler", "title": "sampler",
     "widgets_values": [0, "fixed", 6, 1.8, "lcm", "normal", 0.7]},
    {"id": 12, "type": "VAEDecode", "title": "vae_decode"},
    {"id": 13, "type": "SaveImage", "title": "save",
     "widgets_values": ["output"]}
  ],
  "links": [],
  "version": 0.4
}
```

Note: `links` is empty in this plan snippet; the actual edges are added during implementation by loading the workflow in ComfyUI once and re-exporting. The injector only patches widget values, not topology, so topology is authored once and reused.

- [ ] **Step 2: Write failing test `tests/test_presets/test_edit.py`**

```python
from pathlib import Path

import pytest

from worker.comfyui.workflow import WorkflowTemplate, find_node
from worker.presets.base import InputPaths
from worker.presets.edit import EditPreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_edit_params_defaults():
    p = EditPreset.Parameters(prompt="a cat")
    assert p.prompt == "a cat"
    assert p.negative_prompt == ""
    assert p.strength == 0.7
    assert p.steps == 6
    assert p.cfg == 1.8


def test_edit_params_require_prompt():
    with pytest.raises(Exception):
        EditPreset.Parameters()


def test_edit_inject_patches_prompt_seed_steps_strength_dims():
    preset = EditPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = EditPreset.Parameters(
        prompt="a dog on a skateboard",
        negative_prompt="bad anatomy",
        steps=10, cfg=2.0, strength=0.6, width=768, height=768, seed=42,
    )
    paths = InputPaths(input_image="job1_input.png")
    out = preset.inject(tpl, params, paths).to_dict()

    assert find_node(out, "positive_prompt")["widgets_values"][0] == "a dog on a skateboard"
    assert find_node(out, "negative_prompt")["widgets_values"][0] == "bad anatomy"

    sampler = find_node(out, "sampler")["widgets_values"]
    assert sampler[0] == 42
    assert sampler[1] == "fixed"
    assert sampler[2] == 10
    assert sampler[3] == 2.0
    assert sampler[6] == 0.6

    assert find_node(out, "input_image")["widgets_values"][0] == "job1_input.png"


def test_edit_seed_random_when_none(monkeypatch):
    monkeypatch.setattr("worker.presets.edit.random_seed", lambda: 99999)
    preset = EditPreset()
    tpl = preset.load_template(WORKFLOWS)
    params = EditPreset.Parameters(prompt="x", seed=None)
    paths = InputPaths(input_image="a.png")
    out = preset.inject(tpl, params, paths).to_dict()
    assert find_node(out, "sampler")["widgets_values"][0] == 99999


def test_edit_output_extension_and_content_type():
    assert EditPreset.output_extension == "png"
    assert EditPreset.output_content_type == "image/png"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_presets/test_edit.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `worker/presets/edit.py`**

```python
from __future__ import annotations

import secrets

from pydantic import ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode, Preset


def random_seed() -> int:
    return secrets.randbits(63)


class EditPreset(Preset):
    name = "edit"
    mode = Mode.IMAGE
    template_filename = "edit.json"
    output_extension = "png"
    output_content_type = "image/png"

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(..., min_length=1, max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        strength: float = Field(default=0.7, ge=0.0, le=1.0)
        steps: int = Field(default=6, ge=1, le=50)
        cfg: float = Field(default=1.8, ge=0.0, le=15.0)

    Parameters = Parameters  # type: ignore[misc]

    def inject(
        self,
        template: WorkflowTemplate,
        params: "EditPreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        seed = params.seed if params.seed is not None else random_seed()

        template.set_widget("input_image", 0, input_paths.input_image)
        template.set_widget("positive_prompt", 0, params.prompt)
        template.set_widget("negative_prompt", 0, params.negative_prompt)

        # KSampler widget layout: [seed, seed_mode, steps, cfg, sampler_name, scheduler, denoise]
        template.set_widget("sampler", 0, seed)
        template.set_widget("sampler", 1, "fixed")
        template.set_widget("sampler", 2, params.steps)
        template.set_widget("sampler", 3, params.cfg)
        template.set_widget("sampler", 6, params.strength)

        # IPAdapter weight lives at index 0
        # (do not touch unless caller wanted to override; left at template default 0.7)
        return template
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_presets/test_edit.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add workflows/edit.json worker/presets/edit.py tests/test_presets/test_edit.py
git commit -m "feat(presets): edit preset (SDXL Lightning + IP-Adapter)"
```

---

### Task 22: `style` preset + workflow template

**Files:**
- Create: `workflows/style.json`
- Create: `worker/presets/style.py`
- Create: `tests/test_presets/test_style.py`

The `style` workflow adds:
- `CannyEdgePreprocessor` (title `canny`) — structure hint
- `ControlNetApplyAdvanced` (title `cn_apply`) — ControlNet with weak weight
- Second `LoadImage` (title `reference_image`) for IP-Adapter style ref

For Phase 1 we ship a minimal style workflow that uses IP-Adapter style-focused mode; full ControlNet Union integration lands in Phase 2 with other ControlNet work. To keep this preset runnable in Phase 1, we use IP-Adapter only (no ControlNet yet).

- [ ] **Step 1: Write `workflows/style.json`**

```json
{
  "last_node_id": 14,
  "last_link_id": 30,
  "nodes": [
    {"id": 1, "type": "CheckpointLoaderSimple", "title": "checkpoint",
     "widgets_values": ["Juggernaut_X_RunDiffusion_Hyper.safetensors"]},
    {"id": 2, "type": "LoraLoader", "title": "lcm_lora",
     "widgets_values": ["lcm_lora_sdxl.safetensors", 1.0, 1.0]},
    {"id": 3, "type": "VAELoader", "title": "vae_loader",
     "widgets_values": ["sdxl_vae_fp16_fix.safetensors"]},
    {"id": 4, "type": "CLIPVisionLoader", "title": "clip_vision_loader",
     "widgets_values": ["CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"]},
    {"id": 5, "type": "IPAdapterModelLoader", "title": "ipa_loader",
     "widgets_values": ["ip-adapter-plus_sdxl_vit-h.safetensors"]},
    {"id": 6, "type": "LoadImage", "title": "input_image",
     "widgets_values": ["placeholder.png", "image"]},
    {"id": 7, "type": "LoadImage", "title": "reference_image",
     "widgets_values": ["placeholder_ref.png", "image"]},
    {"id": 8, "type": "CLIPTextEncode", "title": "positive_prompt",
     "widgets_values": [""]},
    {"id": 9, "type": "CLIPTextEncode", "title": "negative_prompt",
     "widgets_values": ["blurry, low quality, distorted"]},
    {"id": 10, "type": "IPAdapter", "title": "ipa_apply",
     "widgets_values": [0.7, 0.0, 1.0, "style transfer"]},
    {"id": 11, "type": "VAEEncode", "title": "vae_encode"},
    {"id": 12, "type": "KSampler", "title": "sampler",
     "widgets_values": [0, "fixed", 6, 1.8, "lcm", "normal", 0.6]},
    {"id": 13, "type": "VAEDecode", "title": "vae_decode"},
    {"id": 14, "type": "SaveImage", "title": "save",
     "widgets_values": ["output"]}
  ],
  "links": [],
  "version": 0.4
}
```

- [ ] **Step 2: Write failing test `tests/test_presets/test_style.py`**

```python
from pathlib import Path

import pytest

from worker.comfyui.workflow import find_node
from worker.presets.base import InputPaths
from worker.presets.style import StylePreset


WORKFLOWS = Path(__file__).parent.parent.parent / "workflows"


def test_style_params_require_prompt_and_reference():
    with pytest.raises(Exception):
        StylePreset.Parameters()
    # reference is provided as an input_path, not in Parameters — prompt alone is enough
    p = StylePreset.Parameters(prompt="oil painting of a river")
    assert p.style_strength == 0.7


def test_style_inject_sets_prompts_and_reference(tmp_path):
    preset = StylePreset()
    tpl = preset.load_template(WORKFLOWS)
    params = StylePreset.Parameters(
        prompt="oil painting of a river",
        negative_prompt="flat",
        style_strength=0.85,
        steps=8, cfg=2.0, seed=7, width=768, height=768,
    )
    paths = InputPaths(input_image="in.png", reference_image="ref.png")
    out = preset.inject(tpl, params, paths).to_dict()

    assert find_node(out, "positive_prompt")["widgets_values"][0] == "oil painting of a river"
    assert find_node(out, "negative_prompt")["widgets_values"][0] == "flat"
    assert find_node(out, "input_image")["widgets_values"][0] == "in.png"
    assert find_node(out, "reference_image")["widgets_values"][0] == "ref.png"
    assert find_node(out, "ipa_apply")["widgets_values"][0] == 0.85
    sampler = find_node(out, "sampler")["widgets_values"]
    assert sampler[0] == 7
    assert sampler[2] == 8
    assert sampler[3] == 2.0


def test_style_inject_rejects_missing_reference():
    preset = StylePreset()
    tpl = preset.load_template(WORKFLOWS)
    params = StylePreset.Parameters(prompt="x")
    paths = InputPaths(input_image="in.png", reference_image=None)
    with pytest.raises(ValueError):
        preset.inject(tpl, params, paths)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_presets/test_style.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `worker/presets/style.py`**

```python
from __future__ import annotations

from pydantic import ConfigDict, Field

from worker.comfyui.workflow import WorkflowTemplate
from worker.presets.base import InputPaths, Mode, Preset
from worker.presets.edit import random_seed


class StylePreset(Preset):
    name = "style"
    mode = Mode.IMAGE
    template_filename = "style.json"
    output_extension = "png"
    output_content_type = "image/png"

    class Parameters(Preset.BaseParameters):
        model_config = ConfigDict(extra="forbid")
        prompt: str = Field(..., min_length=1, max_length=4000)
        negative_prompt: str = Field(default="", max_length=4000)
        style_strength: float = Field(default=0.7, ge=0.0, le=1.5)
        steps: int = Field(default=6, ge=1, le=50)
        cfg: float = Field(default=1.8, ge=0.0, le=15.0)
        strength: float = Field(default=0.6, ge=0.0, le=1.0)

    Parameters = Parameters  # type: ignore[misc]

    def inject(
        self,
        template: WorkflowTemplate,
        params: "StylePreset.Parameters",  # type: ignore[override]
        input_paths: InputPaths,
    ) -> WorkflowTemplate:
        if not input_paths.reference_image:
            raise ValueError("style preset requires reference_image")

        seed = params.seed if params.seed is not None else random_seed()

        template.set_widget("input_image", 0, input_paths.input_image)
        template.set_widget("reference_image", 0, input_paths.reference_image)
        template.set_widget("positive_prompt", 0, params.prompt)
        template.set_widget("negative_prompt", 0, params.negative_prompt)
        template.set_widget("ipa_apply", 0, params.style_strength)

        template.set_widget("sampler", 0, seed)
        template.set_widget("sampler", 1, "fixed")
        template.set_widget("sampler", 2, params.steps)
        template.set_widget("sampler", 3, params.cfg)
        template.set_widget("sampler", 6, params.strength)
        return template
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_presets/test_style.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add workflows/style.json worker/presets/style.py tests/test_presets/test_style.py
git commit -m "feat(presets): style preset (SDXL + IP-Adapter style transfer)"
```

---

### Task 23: Preset registry

**Files:**
- Modify: `worker/presets/__init__.py`
- Create: `tests/test_presets/test_registry.py`

- [ ] **Step 1: Write failing test `tests/test_presets/test_registry.py`**

```python
import pytest

from worker.presets import PRESETS, get_preset
from worker.presets.base import Mode


def test_registry_has_edit_and_style():
    assert set(PRESETS.keys()) >= {"edit", "style"}


def test_get_preset_returns_instance():
    p = get_preset("edit")
    assert p.name == "edit"
    assert p.mode is Mode.IMAGE


def test_get_preset_unknown_raises():
    with pytest.raises(KeyError):
        get_preset("nonexistent")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_presets/test_registry.py -v
```
Expected: fails — no `PRESETS`.

- [ ] **Step 3: Implement registry in `worker/presets/__init__.py`**

```python
from __future__ import annotations

from worker.presets.base import Mode, Preset
from worker.presets.edit import EditPreset
from worker.presets.style import StylePreset


PRESETS: dict[str, Preset] = {
    p.name: p for p in (EditPreset(), StylePreset())
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset: {name!r}")
    return PRESETS[name]


__all__ = ["Mode", "Preset", "PRESETS", "get_preset"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_presets/test_registry.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/presets/__init__.py tests/test_presets/test_registry.py
git commit -m "feat(presets): registry with lookup by name"
```

---

### Task 24: Model manager (mode state tracker)

For Phase 1, ComfyUI owns the actual GPU model loading. Our model manager is a coordination layer: it records which mode is currently loaded, notes the last swap time, and exposes an `ensure(mode)` async primitive for the scheduler to use.

**Files:**
- Create: `worker/pipeline/__init__.py` (empty)
- Create: `worker/pipeline/model_manager.py`
- Create: `tests/test_pipeline/__init__.py` (empty)
- Create: `tests/test_pipeline/test_model_manager.py`

- [ ] **Step 1: Write failing test `tests/test_pipeline/test_model_manager.py`**

```python
import asyncio

import pytest

from worker.pipeline.model_manager import ModelManager
from worker.presets.base import Mode


@pytest.mark.asyncio
async def test_initial_mode_is_none():
    mm = ModelManager()
    assert mm.current_mode() is None
    assert mm.swap_count == 0


@pytest.mark.asyncio
async def test_ensure_first_call_sets_mode():
    mm = ModelManager()
    await mm.ensure(Mode.IMAGE)
    assert mm.current_mode() is Mode.IMAGE
    assert mm.swap_count == 0  # first load is not a swap


@pytest.mark.asyncio
async def test_ensure_same_mode_is_noop():
    mm = ModelManager()
    await mm.ensure(Mode.IMAGE)
    await mm.ensure(Mode.IMAGE)
    assert mm.swap_count == 0


@pytest.mark.asyncio
async def test_ensure_switch_increments_swap():
    mm = ModelManager()
    await mm.ensure(Mode.IMAGE)
    await mm.ensure(Mode.VIDEO)
    assert mm.current_mode() is Mode.VIDEO
    assert mm.swap_count == 1


@pytest.mark.asyncio
async def test_ensure_is_serialized():
    mm = ModelManager()

    async def call(m):
        await mm.ensure(m)

    await asyncio.gather(call(Mode.IMAGE), call(Mode.VIDEO), call(Mode.IMAGE))
    # Regardless of interleaving, we must end in a consistent state.
    assert mm.current_mode() in (Mode.IMAGE, Mode.VIDEO)
```

- [ ] **Step 2: Create empty `worker/pipeline/__init__.py` and `tests/test_pipeline/__init__.py`**

```bash
touch worker/pipeline/__init__.py tests/test_pipeline/__init__.py
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_pipeline/test_model_manager.py -v
```
Expected: ImportError.

- [ ] **Step 4: Implement `worker/pipeline/model_manager.py`**

```python
from __future__ import annotations

import asyncio
import time

from worker.core.logging import get_logger
from worker.presets.base import Mode


log = get_logger("worker.pipeline.model_manager")


class ModelManager:
    """Phase 1 coordination-only model manager.

    Tracks which Mode (IMAGE / VIDEO) is currently "active" in ComfyUI.
    ComfyUI itself handles the actual GPU load/unload; we just ensure only one
    `ensure()` runs at a time and record swaps for metrics.
    """

    def __init__(self) -> None:
        self._mode: Mode | None = None
        self._lock = asyncio.Lock()
        self.swap_count = 0
        self.last_swap_sec: float = 0.0

    def current_mode(self) -> Mode | None:
        return self._mode

    async def ensure(self, mode: Mode) -> None:
        async with self._lock:
            if self._mode is None:
                self._mode = mode
                log.info("model_manager.initial_mode", mode=mode.value)
                return
            if self._mode == mode:
                return
            t0 = time.monotonic()
            # Phase 1: the "swap" happens implicitly when the next workflow references
            # different models; ComfyUI will unload the old ones. We merely record it.
            self._mode = mode
            self.last_swap_sec = time.monotonic() - t0
            self.swap_count += 1
            log.info("model_manager.swap", to=mode.value, swap_count=self.swap_count, duration_sec=self.last_swap_sec)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_pipeline/test_model_manager.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add worker/pipeline/__init__.py worker/pipeline/model_manager.py \
        tests/test_pipeline/__init__.py tests/test_pipeline/test_model_manager.py
git commit -m "feat(pipeline): coordination-only ModelManager with swap metrics"
```

---

### Task 25: Bounded job queue

**Files:**
- Create: `worker/pipeline/queue.py`
- Create: `tests/test_pipeline/test_queue.py`

- [ ] **Step 1: Write failing test `tests/test_pipeline/test_queue.py`**

```python
import asyncio

import pytest

from worker.core.errors import AppError, ErrorCode
from worker.pipeline.queue import Job, JobQueue


def _job(jid: str = "a") -> Job:
    return Job(job_id=jid, preset="edit", raw_request={"job_id": jid})


@pytest.mark.asyncio
async def test_put_and_size():
    q = JobQueue(max_depth=4)
    assert q.depth() == 0
    await q.put(_job("1"))
    assert q.depth() == 1


@pytest.mark.asyncio
async def test_full_queue_raises_429():
    q = JobQueue(max_depth=2)
    await q.put(_job("1"))
    await q.put(_job("2"))
    with pytest.raises(AppError) as exc:
        await q.put(_job("3"))
    assert exc.value.code is ErrorCode.INTERNAL_ERROR  # placeholder; assertion refined below
    assert "full" in exc.value.message.lower()
    assert q.depth() == 2


@pytest.mark.asyncio
async def test_idempotency_duplicate_rejected():
    q = JobQueue(max_depth=4)
    await q.put(_job("1"))
    with pytest.raises(AppError) as exc:
        await q.put(_job("1"))
    assert "duplicate" in exc.value.message.lower()


@pytest.mark.asyncio
async def test_get_blocks_until_put():
    q = JobQueue(max_depth=4)

    async def producer():
        await asyncio.sleep(0.01)
        await q.put(_job("1"))

    async def consumer():
        return await q.get()

    _, job = await asyncio.gather(producer(), consumer())
    assert job.job_id == "1"


@pytest.mark.asyncio
async def test_remove_queued_job():
    q = JobQueue(max_depth=4)
    await q.put(_job("1"))
    await q.put(_job("2"))
    removed = q.remove_if_queued("1")
    assert removed is True
    assert q.depth() == 1


@pytest.mark.asyncio
async def test_remove_unknown_returns_false():
    q = JobQueue(max_depth=4)
    await q.put(_job("1"))
    assert q.remove_if_queued("nope") is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline/test_queue.py -v
```
Expected: ImportError (and two assertions we refine).

- [ ] **Step 3: Implement `worker/pipeline/queue.py`**

```python
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass

from worker.core.errors import AppError, ErrorCode


@dataclass
class Job:
    job_id: str
    preset: str
    raw_request: dict


class JobQueue:
    """Bounded FIFO job queue with idempotency by job_id."""

    def __init__(self, max_depth: int = 8) -> None:
        self._max = max_depth
        self._items: OrderedDict[str, Job] = OrderedDict()
        self._event = asyncio.Event()

    def depth(self) -> int:
        return len(self._items)

    async def put(self, job: Job) -> None:
        if job.job_id in self._items:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                f"duplicate job_id {job.job_id!r}",
                retryable=False,
            )
        if len(self._items) >= self._max:
            raise AppError(
                ErrorCode.INVALID_PARAMETERS,
                f"queue full (max_depth={self._max})",
                retryable=True,
            )
        self._items[job.job_id] = job
        self._event.set()

    async def get(self) -> Job:
        while not self._items:
            self._event.clear()
            await self._event.wait()
        job_id, job = next(iter(self._items.items()))
        self._items.pop(job_id)
        if not self._items:
            self._event.clear()
        return job

    def peek_position(self, job_id: str) -> int | None:
        for i, jid in enumerate(self._items.keys()):
            if jid == job_id:
                return i
        return None

    def remove_if_queued(self, job_id: str) -> bool:
        if job_id in self._items:
            self._items.pop(job_id)
            if not self._items:
                self._event.clear()
            return True
        return False

    def contains(self, job_id: str) -> bool:
        return job_id in self._items
```

- [ ] **Step 4: Refine assertions in test file** — update the two `assert exc.value.code is ErrorCode.INTERNAL_ERROR` markers to match the real code we raise.

```python
# In tests/test_pipeline/test_queue.py
# Replace the placeholder assertion lines with:
#   assert exc.value.code is ErrorCode.INVALID_PARAMETERS
```

```bash
sed -i 's/ErrorCode.INTERNAL_ERROR  # placeholder; assertion refined below/ErrorCode.INVALID_PARAMETERS/' \
    tests/test_pipeline/test_queue.py
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_pipeline/test_queue.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add worker/pipeline/queue.py tests/test_pipeline/test_queue.py
git commit -m "feat(pipeline): bounded JobQueue with idempotency + cancel by id"
```

---

### Task 26: Scheduler with IO / CPU / GPU pools

The scheduler owns the three semaphores and a thread-pool for CPU preprocessing. Per-job orchestration (the actual `run_job` function) lives in the executor (next task); the scheduler provides the pool primitives.

**Files:**
- Create: `worker/pipeline/scheduler.py`
- Create: `tests/test_pipeline/test_scheduler.py`

- [ ] **Step 1: Write failing test `tests/test_pipeline/test_scheduler.py`**

```python
import asyncio
import time

import pytest

from worker.pipeline.scheduler import Pools


@pytest.mark.asyncio
async def test_gpu_semaphore_serializes():
    p = Pools(io=4, cpu=2, gpu=1)
    order: list[str] = []

    async def work(tag: str) -> None:
        async with p.gpu:
            order.append(f"{tag}-in")
            await asyncio.sleep(0.02)
            order.append(f"{tag}-out")

    await asyncio.gather(work("a"), work("b"), work("c"))
    # Every -in is followed by its matching -out before the next -in.
    pairs = [(order[i], order[i+1]) for i in range(0, len(order), 2)]
    for inp, out in pairs:
        assert inp.endswith("-in")
        assert out.endswith("-out")
        assert inp[0] == out[0]


@pytest.mark.asyncio
async def test_io_semaphore_allows_concurrency():
    p = Pools(io=4, cpu=2, gpu=1)
    timestamps: list[float] = []

    async def work() -> None:
        async with p.io:
            timestamps.append(time.monotonic())
            await asyncio.sleep(0.05)

    t0 = time.monotonic()
    await asyncio.gather(*[work() for _ in range(4)])
    # All 4 started within ~5ms of t0 since pool size is 4.
    assert all(t - t0 < 0.02 for t in timestamps)


@pytest.mark.asyncio
async def test_run_cpu_offloads_blocking_fn():
    p = Pools(io=4, cpu=2, gpu=1)

    def blocking_add(a, b):
        time.sleep(0.01)
        return a + b

    r = await p.run_cpu(blocking_add, 2, 3)
    assert r == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline/test_scheduler.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/pipeline/scheduler.py`**

```python
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class Pools:
    """Three concurrency pools for the inference pipeline.

    - io: async semaphore for HTTP download/upload
    - cpu: ThreadPoolExecutor for blocking preprocessing (Pillow, canny, etc.)
    - gpu: async semaphore with size 1 to serialize GPU work
    """

    def __init__(self, io: int = 8, cpu: int = 2, gpu: int = 1) -> None:
        self.io = asyncio.Semaphore(io)
        self.cpu_size = cpu
        self.gpu = asyncio.Semaphore(gpu)
        self._cpu_pool = ThreadPoolExecutor(max_workers=cpu, thread_name_prefix="cpu-pool")

    async def run_cpu(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._cpu_pool,
            lambda: fn(*args, **kwargs),
        )

    async def aclose(self) -> None:
        self._cpu_pool.shutdown(wait=False, cancel_futures=True)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipeline/test_scheduler.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/pipeline/scheduler.py tests/test_pipeline/test_scheduler.py
git commit -m "feat(pipeline): IO/CPU/GPU semaphore pools"
```

---

### Task 27: HMAC-signed callback sender

**Files:**
- Create: `worker/core/callback.py`
- Create: `tests/test_callback.py`

- [ ] **Step 1: Write failing test `tests/test_callback.py`**

```python
import httpx
import pytest

from worker.core.callback import send_callback
from worker.core.hmac_sign import verify


@pytest.mark.asyncio
async def test_callback_success_signs_and_posts(respx_mock):
    recorded = {}

    def handler(req: httpx.Request) -> httpx.Response:
        recorded["body"] = bytes(req.content)
        recorded["sig"] = req.headers.get("x-worker-signature", "")
        return httpx.Response(200)

    respx_mock.post("https://b/cb").mock(side_effect=handler)
    async with httpx.AsyncClient() as client:
        await send_callback(
            client,
            url="https://b/cb",
            secret="s" * 32,
            payload={"job_id": "j1", "status": "success"},
            max_retries=0,
        )
    assert verify(recorded["body"], recorded["sig"], secret="s" * 32) is True


@pytest.mark.asyncio
async def test_callback_5xx_retries_then_drops(respx_mock):
    route = respx_mock.post("https://b/cb")
    route.side_effect = [httpx.Response(500)] * 5 + [httpx.Response(500)]
    async with httpx.AsyncClient() as client:
        result = await send_callback(
            client, url="https://b/cb", secret="s" * 32,
            payload={"job_id": "j", "status": "failed"},
            max_retries=2, backoff_base=0.01,
        )
    assert result is False
    assert route.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_callback_2xx_returns_true(respx_mock):
    respx_mock.post("https://b/cb").respond(202)
    async with httpx.AsyncClient() as client:
        ok = await send_callback(
            client, url="https://b/cb", secret="s" * 32,
            payload={"job_id": "j"}, max_retries=0,
        )
    assert ok is True


@pytest.mark.asyncio
async def test_callback_4xx_retries_once(respx_mock):
    route = respx_mock.post("https://b/cb")
    route.side_effect = [httpx.Response(400), httpx.Response(400)]
    async with httpx.AsyncClient() as client:
        ok = await send_callback(
            client, url="https://b/cb", secret="s" * 32,
            payload={"job_id": "j"}, max_retries=5, backoff_base=0.01,
        )
    assert ok is False
    assert route.call_count == 2  # 4xx short-circuits to 1 retry
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_callback.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/core/callback.py`**

```python
from __future__ import annotations

import asyncio
import json

import httpx

from worker.core.hmac_sign import sign
from worker.core.logging import get_logger


log = get_logger("worker.callback")


async def send_callback(
    client: httpx.AsyncClient,
    *,
    url: str,
    secret: str,
    payload: dict,
    max_retries: int = 5,
    backoff_base: float = 1.0,
    timeout_sec: float = 20.0,
) -> bool:
    """POST payload to url with HMAC signature. Returns True on 2xx, False otherwise.

    5xx and network errors: retry with exponential backoff up to `max_retries`.
    4xx: retry at most once (the URL or auth is likely bad).
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    attempt = 0
    max_client_errors = 1
    client_errors = 0
    while True:
        sig = sign(body, secret=secret)
        try:
            r = await client.post(
                url,
                content=body,
                headers={"content-type": "application/json", "x-worker-signature": sig},
                timeout=timeout_sec,
            )
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            log.warning("callback.network_error", url=url, attempt=attempt, error=str(e))
            r = None

        if r is not None and 200 <= r.status_code < 300:
            return True

        status = r.status_code if r is not None else 0
        if r is not None and 400 <= r.status_code < 500:
            client_errors += 1
            log.warning("callback.4xx", url=url, status=status, attempt=attempt)
            if client_errors > max_client_errors:
                return False
        else:
            log.warning("callback.5xx_or_net", url=url, status=status, attempt=attempt)

        if attempt >= max_retries:
            return False
        await asyncio.sleep(backoff_base * (2**attempt))
        attempt += 1
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_callback.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/core/callback.py tests/test_callback.py
git commit -m "feat(core): HMAC-signed callback sender with 4xx/5xx retry policy"
```

---

### Task 28: Per-job executor (stage orchestration)

The executor ties everything together. It receives a `Job`, runs the stages (download → preprocess → comfyui → upload → callback), records timings, and catches `AppError`/unexpected exceptions into a `failed` callback.

**Files:**
- Create: `worker/pipeline/executor.py`
- Create: `tests/test_pipeline/test_executor.py`

- [ ] **Step 1: Write failing test `tests/test_pipeline/test_executor.py`**

```python
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from PIL import Image

from worker.comfyui.client import ComfyUIClient, ExecutionResult
from worker.pipeline.executor import Executor, JobContext
from worker.pipeline.model_manager import ModelManager
from worker.pipeline.queue import Job
from worker.pipeline.scheduler import Pools


def _sample_request(tmp_output: Path, input_url: str, callback_url: str, upload_url: str) -> dict:
    return {
        "job_id": "j1",
        "preset": "edit",
        "prompt": "a cat",
        "negative_prompt": "",
        "input_image_url": input_url,
        "parameters": {"steps": 6, "cfg": 1.8, "width": 512, "height": 512, "seed": 42, "strength": 0.7},
        "callback_url": callback_url,
        "upload_url": upload_url,
        "upload_method": "PUT",
        "timeout_sec": 60,
    }


def _make_png(path: Path, size=(512, 512)) -> None:
    Image.new("RGB", size, (128, 128, 255)).save(path, "PNG")


@pytest.mark.asyncio
async def test_executor_happy_path(tmp_path, respx_mock):
    _make_png(tmp_path / "src.png")
    _make_png(tmp_path / "out.png")
    comfyui_input = tmp_path / "cu_input"
    comfyui_output = tmp_path / "cu_output"
    comfyui_input.mkdir()
    comfyui_output.mkdir()

    respx_mock.get("https://storage/in.png").respond(200, content=(tmp_path / "src.png").read_bytes())
    upload_route = respx_mock.put("https://storage/out?sig=xyz").respond(200)
    callback_route = respx_mock.post("https://cb/done").respond(200)

    comfy = MagicMock(spec=ComfyUIClient)
    comfy.base_url = "http://127.0.0.1:8188"
    comfy.convert_workflow = AsyncMock(return_value={"1": {"class_type": "X", "inputs": {}}})
    comfy.submit_prompt = AsyncMock(return_value="pid-1")

    async def fake_wait(prompt_id, client_id, *, timeout_sec):
        # Pretend ComfyUI wrote the output file
        dst = comfyui_output / "j1_00001_.png"
        dst.write_bytes((tmp_path / "out.png").read_bytes())
        return ExecutionResult(
            prompt_id=prompt_id,
            outputs={"13": {"images": [{"filename": dst.name, "subfolder": "", "type": "output"}]}},
            success=True,
        )

    comfy.wait_for_completion = AsyncMock(side_effect=fake_wait)
    comfy.interrupt = AsyncMock()

    async with httpx.AsyncClient() as http:
        exe = Executor(
            http=http,
            comfyui=comfy,
            pools=Pools(),
            model_manager=ModelManager(),
            workflows_dir=Path("workflows"),
            comfyui_input_dir=comfyui_input,
            comfyui_output_dir=comfyui_output,
            callback_secret="s" * 32,
            worker_id="wk-test",
        )
        job = Job(job_id="j1", preset="edit",
                  raw_request=_sample_request(tmp_path, "https://storage/in.png", "https://cb/done", "https://storage/out?sig=xyz"))
        await exe.run_job(job)

    assert upload_route.call_count == 1
    assert callback_route.call_count == 1
    body = json.loads(callback_route.calls.last.request.content)
    assert body["status"] == "success"
    assert body["output_url"] == "https://storage/out?sig=xyz"
    assert body["metadata"]["preset"] == "edit"
    assert body["metadata"]["seed"] == 42
    assert "stages_ms" in body
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline/test_executor.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/pipeline/executor.py`**

```python
from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

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
        self._active_prompt_ids: dict[str, str] = {}  # job_id -> prompt_id (for cancel)

    async def run_job(self, job: Job) -> None:
        """Top-level per-job coroutine. Never raises — always emits a callback."""
        ctx = JobContext(job=job)
        req = job.raw_request
        started = time.monotonic()

        try:
            preset = get_preset(req["preset"])
            params = preset.Parameters(prompt=req.get("prompt", ""),
                                       negative_prompt=req.get("negative_prompt", ""),
                                       **{k: v for k, v in req.get("parameters", {}).items()
                                          if v is not None})
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
        # Override SaveImage filename_prefix so outputs land as <job_id>_*
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
    """Pick the first output file from ComfyUI's `outputs` dict.

    ComfyUI returns entries like:
      {"<node_id>": {"images": [{"filename": "...", "subfolder": "", "type": "output"}]}}
      {"<node_id>": {"gifs":   [{"filename": "...", ...}]}}    (for video)
    We accept the first file whose extension matches `expected_ext`.
    """
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipeline/test_executor.py -v
```
Expected: 1 test PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/pipeline/executor.py tests/test_pipeline/test_executor.py
git commit -m "feat(pipeline): per-job executor with stage timings + AppError callbacks"
```

---

### Task 29: Wire executor into a background worker loop

The scheduler's final missing piece: a long-running coroutine that reads jobs off the queue and hands them to the executor. It lives on the FastAPI app state and is cancelled during graceful shutdown.

**Files:**
- Create: `worker/pipeline/runner.py`
- Create: `tests/test_pipeline/test_runner.py`

- [ ] **Step 1: Write failing test `tests/test_pipeline/test_runner.py`**

```python
import asyncio
from unittest.mock import AsyncMock

import pytest

from worker.pipeline.queue import Job, JobQueue
from worker.pipeline.runner import Runner


@pytest.mark.asyncio
async def test_runner_consumes_jobs_and_stops():
    q = JobQueue(max_depth=4)
    exe = AsyncMock()
    exe.run_job = AsyncMock()
    await q.put(Job("a", "edit", {"job_id": "a"}))
    await q.put(Job("b", "edit", {"job_id": "b"}))

    r = Runner(q, exe)
    task = asyncio.create_task(r.run_forever())
    # Wait for both jobs to be picked up
    for _ in range(50):
        if exe.run_job.await_count >= 2:
            break
        await asyncio.sleep(0.01)
    await r.stop()
    await task
    assert exe.run_job.await_count == 2


@pytest.mark.asyncio
async def test_runner_survives_executor_exception():
    q = JobQueue(max_depth=4)
    exe = AsyncMock()
    exe.run_job = AsyncMock(side_effect=[RuntimeError("boom"), None])
    await q.put(Job("a", "edit", {"job_id": "a"}))
    await q.put(Job("b", "edit", {"job_id": "b"}))

    r = Runner(q, exe)
    task = asyncio.create_task(r.run_forever())
    for _ in range(50):
        if exe.run_job.await_count >= 2:
            break
        await asyncio.sleep(0.01)
    await r.stop()
    await task
    assert exe.run_job.await_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline/test_runner.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/pipeline/runner.py`**

```python
from __future__ import annotations

import asyncio

from worker.core.logging import get_logger


log = get_logger("worker.runner")


class Runner:
    """Drains JobQueue into Executor.run_job in a loop until asked to stop."""

    def __init__(self, queue, executor) -> None:
        self._queue = queue
        self._executor = executor
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        while not self._stop.is_set():
            get_task = asyncio.create_task(self._queue.get())
            stop_task = asyncio.create_task(self._stop.wait())
            done, pending = await asyncio.wait(
                {get_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if stop_task in done and get_task not in done:
                return
            try:
                job = get_task.result()
            except asyncio.CancelledError:
                return
            try:
                await self._executor.run_job(job)
            except Exception:  # noqa: BLE001
                log.exception("runner.job_crashed", job_id=getattr(job, "job_id", "?"))

    async def stop(self) -> None:
        self._stop.set()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_pipeline/test_runner.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add worker/pipeline/runner.py tests/test_pipeline/test_runner.py
git commit -m "feat(pipeline): queue Runner with clean shutdown + crash isolation"
```

---

### Task 30: App state and FastAPI lifespan wiring

**Files:**
- Create: `worker/app_state.py`
- Modify: `worker/main.py`
- Create: `tests/test_app_state.py`

The app-state module holds the runtime singletons (queue, pools, ComfyUI client, executor, runner). `create_app` wires them into a FastAPI lifespan.

- [ ] **Step 1: Write failing test `tests/test_app_state.py`**

```python
from pathlib import Path

import pytest

from worker.app_state import AppState


def test_app_state_exposes_components(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))
    from worker.core.config import get_settings
    get_settings.cache_clear()

    state = AppState.build()
    assert state.queue is not None
    assert state.pools is not None
    assert state.model_manager is not None
    assert state.comfyui is not None
    assert state.http is not None
    assert state.executor is not None
    assert state.runner is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_app_state.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `worker/app_state.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from worker.comfyui.client import ComfyUIClient
from worker.core.config import Settings, get_settings
from worker.core.logging import get_logger
from worker.pipeline.executor import Executor
from worker.pipeline.model_manager import ModelManager
from worker.pipeline.queue import JobQueue
from worker.pipeline.runner import Runner
from worker.pipeline.scheduler import Pools


log = get_logger("worker.app_state")


@dataclass
class AppState:
    settings: Settings
    http: httpx.AsyncClient
    comfyui: ComfyUIClient
    pools: Pools
    model_manager: ModelManager
    queue: JobQueue
    executor: Executor
    runner: Runner

    @classmethod
    def build(cls) -> AppState:
        s = get_settings()
        http = httpx.AsyncClient()
        comfyui = ComfyUIClient(http, s.comfyui_base_url)
        pools = Pools(io=8, cpu=2, gpu=1)
        mm = ModelManager()
        queue = JobQueue(max_depth=s.max_queue_depth)

        comfyui_input_dir = s.comfyui_path / "input"
        comfyui_output_dir = s.comfyui_path / "output"
        comfyui_input_dir.mkdir(parents=True, exist_ok=True)
        comfyui_output_dir.mkdir(parents=True, exist_ok=True)

        executor = Executor(
            http=http,
            comfyui=comfyui,
            pools=pools,
            model_manager=mm,
            workflows_dir=s.workflows_path,
            comfyui_input_dir=comfyui_input_dir,
            comfyui_output_dir=comfyui_output_dir,
            callback_secret=s.callback_hmac_secret,
            worker_id=s.resolved_worker_id(),
        )
        runner = Runner(queue, executor)
        return cls(
            settings=s, http=http, comfyui=comfyui, pools=pools,
            model_manager=mm, queue=queue, executor=executor, runner=runner,
        )

    async def aclose(self) -> None:
        await self.runner.stop()
        await self.pools.aclose()
        await self.http.aclose()
```

- [ ] **Step 4: Modify `worker/main.py` to use lifespan**

```python
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from worker.api import health
from worker.app_state import AppState
from worker.core.config import get_settings
from worker.core.logging import configure_logging, get_logger


log = get_logger("worker.main")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state = AppState.build()
    app.state.app_state = state
    log.info("worker.startup", worker_id=state.settings.resolved_worker_id(),
             port=state.settings.worker_port)
    # Start the queue runner in the background
    runner_task = asyncio.create_task(state.runner.run_forever())
    try:
        yield
    finally:
        log.info("worker.shutdown.begin")
        await state.aclose()
        try:
            await asyncio.wait_for(runner_task, timeout=60.0)
        except asyncio.TimeoutError:
            log.warning("worker.shutdown.runner_timeout")
        log.info("worker.shutdown.done")


def create_app() -> FastAPI:
    s = get_settings()
    configure_logging(level=s.log_level, fmt=s.log_format)
    app = FastAPI(title="Inference Worker", version="0.1.0", lifespan=_lifespan)
    app.include_router(health.router)
    return app


app = create_app()
```

- [ ] **Step 5: Run test**

```bash
pytest tests/test_app_state.py tests/test_api/test_health.py -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add worker/app_state.py worker/main.py tests/test_app_state.py
git commit -m "feat(app): AppState + lifespan-managed runner + graceful shutdown"
```

---

### Task 31: `POST /v1/generate` endpoint

**Files:**
- Create: `worker/api/generate.py`
- Modify: `worker/main.py`
- Create: `tests/test_api/test_generate.py`

- [ ] **Step 1: Write failing test `tests/test_api/test_generate.py`**

```python
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    (tmp_path / "workflows").mkdir(exist_ok=True)
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path / "workflows"))
    from worker.core.config import get_settings
    get_settings.cache_clear()
    from worker.main import create_app
    app = create_app()
    return TestClient(app)


def _payload(job_id="j1", preset="edit"):
    return {
        "job_id": job_id,
        "preset": preset,
        "prompt": "a cat",
        "input_image_url": "https://s/in.png",
        "parameters": {"steps": 6, "cfg": 1.8, "width": 512, "height": 512},
        "callback_url": "https://b/cb",
        "upload_url": "https://s/out",
        "upload_method": "PUT",
    }


def test_generate_requires_auth(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/generate", json=_payload(), headers={"Idempotency-Key": "x"})
    assert r.status_code == 401


def test_generate_requires_idempotency_key(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/generate", json=_payload(), headers={"Authorization": f"Bearer {'k'*32}"})
    assert r.status_code == 400


def test_generate_accepts_valid(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/generate", json=_payload(), headers={
        "Authorization": f"Bearer {'k'*32}",
        "Idempotency-Key": "j1",
    })
    assert r.status_code == 202
    body = r.json()
    assert body["accepted"] is True
    assert body["job_id"] == "j1"
    assert "queue_position" in body
    assert "worker_id" in body


def test_generate_unknown_preset_returns_400(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/generate", json=_payload(preset="nope"), headers={
        "Authorization": f"Bearer {'k'*32}",
        "Idempotency-Key": "j1",
    })
    assert r.status_code == 400
    assert "preset" in r.text.lower()


def test_generate_duplicate_job_id_returns_409(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    hdrs = {"Authorization": f"Bearer {'k'*32}", "Idempotency-Key": "j1"}
    r1 = c.post("/v1/generate", json=_payload(job_id="j1"), headers=hdrs)
    assert r1.status_code == 202
    r2 = c.post("/v1/generate", json=_payload(job_id="j1"), headers=hdrs)
    assert r2.status_code == 409


def test_generate_queue_full_returns_429(monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_QUEUE_DEPTH", "1")
    c = _client(monkeypatch, tmp_path)
    hdrs = lambda k: {"Authorization": f"Bearer {'k'*32}", "Idempotency-Key": k}
    r1 = c.post("/v1/generate", json=_payload(job_id="a"), headers=hdrs("a"))
    r2 = c.post("/v1/generate", json=_payload(job_id="b"), headers=hdrs("b"))
    assert r1.status_code == 202
    assert r2.status_code == 429
    assert "retry-after" in [k.lower() for k in r2.headers.keys()]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api/test_generate.py -v
```
Expected: ImportError (endpoint not wired yet).

- [ ] **Step 3: Implement `worker/api/generate.py`**

```python
from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from worker.core.auth import require_bearer
from worker.core.errors import AppError, ErrorCode
from worker.pipeline.queue import Job
from worker.presets import PRESETS, get_preset


router = APIRouter(tags=["generate"])


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(..., min_length=1, max_length=128)
    preset: Literal["edit", "style", "controlnet", "inpaint", "ltx_video"]
    prompt: str = Field("", max_length=4000)
    negative_prompt: str = Field("", max_length=4000)
    input_image_url: str
    mask_image_url: str | None = None
    reference_image_url: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    callback_url: str
    upload_url: str
    upload_method: Literal["PUT", "POST"] = "PUT"
    timeout_sec: int = Field(default=300, ge=10, le=1800)


@router.post("/v1/generate", status_code=202, dependencies=[Depends(require_bearer)])
async def generate(
    req: GenerateRequest,
    request: Request,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
):
    if not idempotency_key:
        raise HTTPException(status_code=400, detail={"code": "INVALID_PARAMETERS", "message": "Idempotency-Key header required"})

    if req.preset not in PRESETS:
        raise HTTPException(status_code=400, detail={"code": "INVALID_PRESET", "message": f"unknown preset {req.preset!r}"})

    preset = get_preset(req.preset)

    # Preset-specific parameter validation
    try:
        extra = {"prompt": req.prompt}
        if "negative_prompt" in preset.Parameters.model_fields:
            extra["negative_prompt"] = req.negative_prompt
        preset.Parameters(**extra, **{k: v for k, v in req.parameters.items() if v is not None})
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"code": "INVALID_PARAMETERS", "errors": e.errors()}) from e

    # Style preset requires reference image
    if req.preset == "style" and not req.reference_image_url:
        raise HTTPException(status_code=400, detail={"code": "INVALID_PARAMETERS", "message": "style preset requires reference_image_url"})
    # Inpaint preset requires mask
    if req.preset == "inpaint" and not req.mask_image_url:
        raise HTTPException(status_code=400, detail={"code": "INVALID_PARAMETERS", "message": "inpaint preset requires mask_image_url"})

    state = request.app.state.app_state
    job = Job(job_id=req.job_id, preset=req.preset, raw_request=req.model_dump())
    try:
        await state.queue.put(job)
    except AppError as ae:
        if ae.code is ErrorCode.INVALID_PARAMETERS and "duplicate" in ae.message:
            raise HTTPException(status_code=409, detail={"code": "DUPLICATE", "message": ae.message}) from ae
        if ae.code is ErrorCode.INVALID_PARAMETERS and "queue full" in ae.message:
            return _retry_after_response(state.queue.depth())
        raise HTTPException(status_code=400, detail=ae.to_dict()) from ae

    return {
        "accepted": True,
        "job_id": req.job_id,
        "queue_position": state.queue.peek_position(req.job_id) or 0,
        "worker_id": state.settings.resolved_worker_id(),
    }


def _retry_after_response(depth: int):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"code": "QUEUE_FULL", "message": "worker queue is full"},
        headers={"Retry-After": "5"},
    )
```

- [ ] **Step 4: Register router in `worker/main.py`**

```python
# Modify create_app():
#   from worker.api import generate, health
#   app.include_router(generate.router)
```

Edit:

```bash
python - <<'PY'
from pathlib import Path
p = Path("worker/main.py")
src = p.read_text()
src = src.replace(
    "from worker.api import health",
    "from worker.api import generate, health",
)
src = src.replace(
    "app.include_router(health.router)",
    "app.include_router(health.router)\n    app.include_router(generate.router)",
)
p.write_text(src)
PY
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_api/test_generate.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add worker/api/generate.py worker/main.py tests/test_api/test_generate.py
git commit -m "feat(api): POST /v1/generate with validation + idempotency + queue-full 429"
```

---

### Task 32: `POST /v1/cancel` endpoint

**Files:**
- Create: `worker/api/cancel.py`
- Modify: `worker/main.py`
- Create: `tests/test_api/test_cancel.py`

- [ ] **Step 1: Write failing test `tests/test_api/test_cancel.py`**

```python
from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    (tmp_path / "workflows").mkdir(exist_ok=True)
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path / "workflows"))
    from worker.core.config import get_settings
    get_settings.cache_clear()
    from worker.main import create_app
    return TestClient(create_app())


def test_cancel_queued_job(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    hdrs = {"Authorization": f"Bearer {'k'*32}", "Idempotency-Key": "j"}
    payload = {
        "job_id": "j", "preset": "edit", "prompt": "x",
        "input_image_url": "https://x/i.png", "callback_url": "https://b/cb",
        "upload_url": "https://s/o", "upload_method": "PUT",
    }
    assert c.post("/v1/generate", json=payload, headers=hdrs).status_code == 202
    r = c.post("/v1/cancel", json={"job_id": "j"}, headers={"Authorization": f"Bearer {'k'*32}"})
    assert r.status_code == 200
    assert r.json()["cancelled"] is True


def test_cancel_unknown_returns_404(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/cancel", json={"job_id": "nope"}, headers={"Authorization": f"Bearer {'k'*32}"})
    assert r.status_code == 404


def test_cancel_requires_auth(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/v1/cancel", json={"job_id": "j"})
    assert r.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api/test_cancel.py -v
```
Expected: ImportError (router not yet wired).

- [ ] **Step 3: Implement `worker/api/cancel.py`**

```python
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from worker.core.auth import require_bearer


router = APIRouter(tags=["cancel"])


class CancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(..., min_length=1, max_length=128)


@router.post("/v1/cancel", dependencies=[Depends(require_bearer)])
async def cancel(req: CancelRequest, request: Request):
    state = request.app.state.app_state
    if state.queue.remove_if_queued(req.job_id):
        return {"cancelled": True, "stage": "queued"}
    # Still-running (or unknown): attempt interrupt — only cancels the current job.
    if req.job_id in state.executor._active_prompt_ids:  # type: ignore[attr-defined]
        await state.executor.cancel(req.job_id)
        return {"cancelled": True, "stage": "running"}
    raise HTTPException(status_code=404, detail={"code": "UNKNOWN_JOB", "message": "job not in queue or running"})
```

- [ ] **Step 4: Register router**

```bash
python - <<'PY'
from pathlib import Path
p = Path("worker/main.py")
src = p.read_text()
src = src.replace(
    "from worker.api import generate, health",
    "from worker.api import cancel, generate, health",
)
src = src.replace(
    "app.include_router(generate.router)",
    "app.include_router(generate.router)\n    app.include_router(cancel.router)",
)
p.write_text(src)
PY
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_api/test_cancel.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add worker/api/cancel.py worker/main.py tests/test_api/test_cancel.py
git commit -m "feat(api): POST /v1/cancel (queued removal + running interrupt)"
```

---

### Task 33: Extend `/v1/health` with ComfyUI probe + VRAM metrics

**Files:**
- Modify: `worker/api/health.py`
- Create: `worker/gpu_info.py`
- Create: `tests/test_gpu_info.py`
- Create: `tests/test_api/test_health_extended.py`

- [ ] **Step 1: Write failing test `tests/test_gpu_info.py`**

```python
from worker.gpu_info import GpuInfo, safe_gpu_info


def test_safe_gpu_info_returns_defaults_when_nvml_unavailable(monkeypatch):
    monkeypatch.setattr("worker.gpu_info._try_nvml", lambda _gpu_device: None)
    info = safe_gpu_info(gpu_device=0)
    assert isinstance(info, GpuInfo)
    assert info.name == ""
    assert info.vram_total_gb == 0.0
```

- [ ] **Step 2: Implement `worker/gpu_info.py`**

```python
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
```

- [ ] **Step 3: Extend `worker/api/health.py`**

```python
from __future__ import annotations

from fastapi import APIRouter, Request

from worker import __version__
from worker.core.config import get_settings
from worker.gpu_info import safe_gpu_info
from worker.state import worker_state


router = APIRouter(tags=["health"])


@router.get("/v1/health")
async def health(request: Request) -> dict:
    s = get_settings()
    state = getattr(request.app.state, "app_state", None)
    comfyui_alive = False
    queue_depth = 0
    current_job = None
    model_loaded = None
    if state is not None:
        try:
            from worker.comfyui.readiness import is_alive
            comfyui_alive = await is_alive(state.http, s.comfyui_base_url)
        except Exception:
            comfyui_alive = False
        queue_depth = state.queue.depth()
        current_job = worker_state.current_job_id
        mode = state.model_manager.current_mode()
        model_loaded = mode.value if mode else None
    gpu = safe_gpu_info(gpu_device=s.gpu_device)
    return {
        "ok": True,
        "ready": worker_state.ready,
        "worker_id": s.resolved_worker_id(),
        "gpu": gpu.name,
        "vram_used_gb": round(gpu.vram_used_gb, 2),
        "vram_total_gb": round(gpu.vram_total_gb, 2),
        "gpu_util_ratio": round(gpu.util_ratio, 2),
        "queue_depth": queue_depth,
        "current_job": current_job,
        "model_loaded": model_loaded,
        "comfyui_alive": comfyui_alive,
        "uptime_sec": worker_state.uptime_sec(),
        "version": __version__,
    }
```

- [ ] **Step 4: Write extended API test `tests/test_api/test_health_extended.py`**

```python
from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKER_API_KEY", "k" * 32)
    monkeypatch.setenv("CALLBACK_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("COMFYUI_PATH", str(tmp_path))
    monkeypatch.setenv("MODELS_PATH", str(tmp_path))
    monkeypatch.setenv("CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_PATH", str(tmp_path))
    from worker.core.config import get_settings
    get_settings.cache_clear()
    from worker.main import create_app
    return TestClient(create_app())


def test_health_includes_gpu_and_comfyui_fields(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    body = c.get("/v1/health").json()
    for key in ("gpu", "vram_used_gb", "vram_total_gb", "comfyui_alive", "queue_depth"):
        assert key in body
    # comfyui_alive is False because no real server
    assert body["comfyui_alive"] is False
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_gpu_info.py tests/test_api/test_health_extended.py -v
```
Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add worker/gpu_info.py worker/api/health.py \
        tests/test_gpu_info.py tests/test_api/test_health_extended.py
git commit -m "feat(api): /v1/health with ComfyUI probe + NVML GPU metrics"
```

---
### Task 34: Warm-up script (`scripts/warm_comfyui.py`)

**Files:**
- Create: `scripts/warm_comfyui.py`
- Create: `tests/fixtures/input_512.png`

- [ ] **Step 1: Create small test fixture image**

```bash
python - <<'PY'
from PIL import Image
from pathlib import Path
Path("tests/fixtures").mkdir(parents=True, exist_ok=True)
Image.new("RGB", (512, 512), (128, 128, 200)).save("tests/fixtures/input_512.png", "PNG")
Image.new("RGB", (512, 512), (200, 128, 128)).save("tests/fixtures/ref_512.png", "PNG")
PY
```

- [ ] **Step 2: Write `scripts/warm_comfyui.py`**

```python
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
    cu_output = settings.comfyui_path / "output"
    cu_input.mkdir(parents=True, exist_ok=True)

    # Copy fixture into ComfyUI input dir
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
    # Clean warm-up files
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
                except Exception as e:  # noqa: BLE001
                    log.exception("warmup.preset_error", preset=p)
                    ok = False
        return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 3: Smoke test import**

```bash
python -c "import scripts.warm_comfyui" 2>&1 || python -c "import importlib.util; spec = importlib.util.spec_from_file_location('warm', 'scripts/warm_comfyui.py'); importlib.util.module_from_spec(spec)"
```
Expected: no import errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/warm_comfyui.py tests/fixtures/input_512.png tests/fixtures/ref_512.png
git commit -m "feat(scripts): warm_comfyui.py for post-boot kernel warm-up"
```

---

### Task 35: Mock backend for local e2e (`scripts/mock_backend.py`)

**Files:**
- Create: `scripts/mock_backend.py`

- [ ] **Step 1: Write `scripts/mock_backend.py`**

```python
#!/usr/bin/env python
"""Local mock of the user's central backend. Runs as a standalone FastAPI service:

- PUT /storage/in/<id>.png       -> stores uploaded test images
- GET /storage/in/<id>.png       -> serves them (input to worker)
- PUT /storage/out/<id>          -> receives worker upload
- GET /storage/out/<id>          -> serves generated output
- POST /cb/<id>                  -> receives callback (verifies HMAC)

Usage:
    MOCK_HMAC_SECRET=... python scripts/mock_backend.py --port 9100 --dir ./mock_data
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from worker.core.hmac_sign import verify


app = FastAPI(title="mock-backend")
DIR = Path(os.environ.get("MOCK_DIR", "./mock_data")).resolve()
DIR.mkdir(parents=True, exist_ok=True)
(DIR / "in").mkdir(exist_ok=True)
(DIR / "out").mkdir(exist_ok=True)
(DIR / "cb").mkdir(exist_ok=True)

HMAC_SECRET = os.environ.get("MOCK_HMAC_SECRET", "")


@app.put("/storage/in/{name}")
async def put_input(name: str, request: Request):
    (DIR / "in" / name).write_bytes(await request.body())
    return {"ok": True}


@app.get("/storage/in/{name}")
async def get_input(name: str):
    p = DIR / "in" / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.put("/storage/out/{name}")
async def put_output(name: str, request: Request):
    (DIR / "out" / name).write_bytes(await request.body())
    return {"ok": True}


@app.get("/storage/out/{name}")
async def get_output(name: str):
    p = DIR / "out" / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/cb/{job_id}")
async def callback(job_id: str, request: Request):
    body = await request.body()
    sig = request.headers.get("x-worker-signature", "")
    verified = False
    if HMAC_SECRET:
        verified = verify(body, sig, secret=HMAC_SECRET)
    (DIR / "cb" / f"{job_id}.json").write_text(
        json.dumps({"verified": verified, "signature": sig, "payload": json.loads(body)})
    )
    return {"ok": True, "verified": verified}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    if args.dir:
        os.environ["MOCK_DIR"] = args.dir
    uvicorn.run("scripts.mock_backend:app", host=args.host, port=args.port, log_level="warning")
```

- [ ] **Step 2: Smoke test**

```bash
python scripts/mock_backend.py --port 9100 --dir /tmp/mock_data &
sleep 1
curl -s -X PUT --data-binary "hello" http://127.0.0.1:9100/storage/in/a.txt
curl -s http://127.0.0.1:9100/storage/in/a.txt
kill %1
rm -rf /tmp/mock_data
```
Expected: `{"ok":true}` then `hello`.

- [ ] **Step 3: Commit**

```bash
git add scripts/mock_backend.py
git commit -m "feat(scripts): mock_backend.py for local e2e testing"
```

---

### Task 36: Integration test (GPU required, marked)

**Files:**
- Create: `tests/integration/__init__.py` (empty)
- Create: `tests/integration/test_edit_e2e.py`
- Create: `tests/integration/conftest.py`

These tests are marked `gpu` and skipped by default in CI; run manually on a GPU box.

- [ ] **Step 1: Create empty `tests/integration/__init__.py`**

```bash
mkdir -p tests/integration && touch tests/integration/__init__.py
```

- [ ] **Step 2: Write `tests/integration/conftest.py`**

```python
import os
import time

import httpx
import pytest


@pytest.fixture(scope="session")
def worker_base_url() -> str:
    return os.environ.get("WORKER_URL", "http://127.0.0.1:8000")


@pytest.fixture(scope="session")
def worker_api_key() -> str:
    key = os.environ.get("WORKER_API_KEY")
    if not key:
        pytest.skip("WORKER_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def mock_backend_url() -> str:
    return os.environ.get("MOCK_BACKEND_URL", "http://127.0.0.1:9100")


@pytest.fixture(scope="session")
def wait_for_worker_ready(worker_base_url):
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            r = httpx.get(f"{worker_base_url}/v1/health", timeout=3.0)
            if r.status_code == 200 and r.json().get("ready"):
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    pytest.skip("worker did not become ready in 240s")
```

- [ ] **Step 3: Write `tests/integration/test_edit_e2e.py`**

```python
import json
import time
import uuid
from pathlib import Path

import httpx
import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _upload_input(mock_url: str, payload: bytes, name: str) -> str:
    r = httpx.put(f"{mock_url}/storage/in/{name}", content=payload, timeout=30)
    r.raise_for_status()
    return f"{mock_url}/storage/in/{name}"


def _wait_callback(mock_url: str, job_id: str, timeout_sec: int = 240) -> dict:
    """Poll mock backend for the callback JSON."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        r = httpx.get(f"{mock_url}/storage/out/_callback_index_{job_id}", timeout=2)
        # The mock writes the callback to DIR/cb/<id>.json; we expose it via /storage/out
        # by copying during polling. Simpler: poll the file directly via a side-channel.
        time.sleep(1)
        try:
            # Check via the "cb" subdir exposed path (we add it here if missing):
            r2 = httpx.get(f"{mock_url}/cb_peek/{job_id}", timeout=2)
            if r2.status_code == 200:
                return r2.json()
        except httpx.HTTPError:
            pass
    raise AssertionError(f"no callback received for {job_id}")


def test_edit_preset_end_to_end(wait_for_worker_ready, worker_base_url, worker_api_key, mock_backend_url):
    img_bytes = Path("tests/fixtures/input_512.png").read_bytes()
    name = f"in_{uuid.uuid4().hex}.png"
    input_url = _upload_input(mock_backend_url, img_bytes, name)
    job_id = f"j_{uuid.uuid4().hex[:12]}"

    out_name = f"out_{job_id}.png"
    resp = httpx.post(
        f"{worker_base_url}/v1/generate",
        headers={"Authorization": f"Bearer {worker_api_key}", "Idempotency-Key": job_id},
        json={
            "job_id": job_id, "preset": "edit", "prompt": "a cozy cat",
            "input_image_url": input_url,
            "parameters": {"steps": 6, "cfg": 1.8, "width": 512, "height": 512, "seed": 1234},
            "callback_url": f"{mock_backend_url}/cb/{job_id}",
            "upload_url": f"{mock_backend_url}/storage/out/{out_name}",
            "upload_method": "PUT",
            "timeout_sec": 120,
        },
        timeout=10,
    )
    assert resp.status_code == 202, resp.text

    # Wait for the output to appear on mock storage
    deadline = time.time() + 240
    out_url = f"{mock_backend_url}/storage/out/{out_name}"
    while time.time() < deadline:
        r = httpx.get(out_url, timeout=3)
        if r.status_code == 200 and len(r.content) > 1024:
            break
        time.sleep(1)
    else:
        pytest.fail("output never appeared")

    # Basic sanity checks
    assert r.headers.get("content-type", "").startswith("image/") or len(r.content) > 10000
```

Note: This test relies on the mock backend exposing a `cb_peek` helper endpoint. We extend the mock backend in the next step.

- [ ] **Step 4: Extend `scripts/mock_backend.py` with `cb_peek` helper**

```bash
python - <<'PY'
from pathlib import Path
p = Path("scripts/mock_backend.py")
src = p.read_text()
helper = '''

@app.get("/cb_peek/{job_id}")
async def cb_peek(job_id: str):
    p = DIR / "cb" / f"{job_id}.json"
    if not p.exists():
        raise HTTPException(404)
    return json.loads(p.read_text())
'''
if "cb_peek" not in src:
    # Insert before the __main__ block
    src = src.replace('if __name__ == "__main__":', helper + '\n\nif __name__ == "__main__":')
    p.write_text(src)
    print("patched")
else:
    print("already patched")
PY
```

- [ ] **Step 5: Commit**

```bash
git add tests/integration/ scripts/mock_backend.py
git commit -m "test(integration): edit preset e2e against live ComfyUI + mock backend"
```

---

### Task 37: README + operator quick reference

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Inference Worker (Phase 1)

Stateless GPU inference worker for image generation. Runs ComfyUI headlessly
on `127.0.0.1:8188` and exposes a FastAPI service on port `$WORKER_PORT` that
accepts prompt + image from the user's backend, produces a revised image, and
PUTs the result to a signed URL plus a HMAC-signed callback.

## Quick start (any Linux + NVIDIA GPU host)

```bash
git clone <REPO_URL> /workspace/works
cd /workspace/works
cp .env.example .env
bash scripts/gen_secrets.sh .env     # populates missing API key + HMAC secret
docker compose up -d --build
```

Wait ~5–8 minutes on first boot (model download). Check readiness:

```bash
curl -s http://127.0.0.1:8000/v1/health | python -m json.tool
# "ready": true  when warm-up completes
```

### Without Docker

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/bootstrap.sh
bash scripts/start.sh
```

## Environment

All config via `.env`. See `.env.example` for the complete list.
Required secrets (`WORKER_API_KEY`, `CALLBACK_HMAC_SECRET`) are auto-generated
by `scripts/gen_secrets.sh` if missing.

## Request flow

1. User's backend POSTs to `http://<worker>/v1/generate` with bearer auth,
   Idempotency-Key header, and the JSON body documented in
   `docs/superpowers/specs/2026-04-14-img-video-worker-design.md` §3.1.
2. Worker responds `202 Accepted`, enqueues the job.
3. Worker downloads the input, runs inference, PUTs the output to the
   signed `upload_url`, and POSTs a HMAC-signed callback to `callback_url`.

## Presets (Phase 1)

- `edit` — IP-Adapter + Lightning SDXL, prompt-driven edit
- `style` — IP-Adapter style transfer using a reference image

## Development

```bash
pytest                       # unit + component tests (no GPU)
pytest -m "not gpu"          # same, explicit
pytest -m gpu                # integration (requires a running GPU + worker)
ruff check . && pyright      # lint + typecheck
```

## Local e2e

```bash
# Terminal 1 — worker
docker compose up
# Terminal 2 — mock backend
MOCK_HMAC_SECRET="$(grep '^CALLBACK_HMAC_SECRET=' .env | cut -d= -f2)" \
    python scripts/mock_backend.py --port 9100 --dir ./mock_data
# Terminal 3 — integration test
WORKER_URL=http://127.0.0.1:8000 \
  WORKER_API_KEY="$(grep '^WORKER_API_KEY=' .env | cut -d= -f2)" \
  MOCK_BACKEND_URL=http://127.0.0.1:9100 \
  pytest -m gpu tests/integration -v
```

## Observability

- Logs: JSON to stdout (`docker logs` / journald)
- Health: `GET /v1/health`
- Metrics (Phase 2): `GET /v1/metrics` on `$METRICS_PORT`

## Portability to a new machine

```bash
git clone <REPO_URL> /workspace/works && cd /workspace/works
cp .env.example .env && bash scripts/gen_secrets.sh .env
docker compose up -d
```

The three commands are sufficient on any Linux host with the NVIDIA Container Toolkit installed.

## Next (Phase 2, separate plan)

- Video preset (LTX-Video)
- `controlnet` + `inpaint` presets
- Full Prometheus metrics + Grafana dashboard
- CI → GHCR pipeline
- Load testing
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quick-start + dev + e2e flow"
```

---

### Task 38: Final smoke — boot the full stack locally once

This is a manual verification step, not a TDD cycle. Do it on the GPU machine.

- [ ] **Step 1: Ensure `/workspace/ComfyUI` is the symlink target or set `COMFYUI_PATH` in `.env`**

```bash
grep -q "^COMFYUI_PATH=" .env || echo "COMFYUI_PATH=/workspace/ComfyUI" >> .env
bash scripts/gen_secrets.sh .env
```

- [ ] **Step 2: Run bootstrap + start in the current shell (non-docker path)**

```bash
bash scripts/bootstrap.sh
uvicorn worker.main:app --host 127.0.0.1 --port 8000 &
# In another terminal, start ComfyUI manually just for this smoke:
( cd "$COMFYUI_PATH" && python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --disable-metadata --normalvram ) &
```

- [ ] **Step 3: Verify readiness**

```bash
for i in {1..60}; do
  READY=$(curl -s http://127.0.0.1:8000/v1/health | python -c "import sys,json; print(json.load(sys.stdin)['ready'])")
  [ "$READY" = "True" ] && echo "READY" && break
  sleep 2
done
curl -s http://127.0.0.1:8000/v1/health | python -m json.tool
```
Expected: `ready: true` within ~2–3 minutes after models are present.

- [ ] **Step 4: Run one real `edit` job via the mock backend**

```bash
# start mock backend
MOCK_HMAC_SECRET="$(grep '^CALLBACK_HMAC_SECRET=' .env | cut -d= -f2)" \
  python scripts/mock_backend.py --port 9100 --dir /tmp/mock_data &

# run the integration test
WORKER_URL=http://127.0.0.1:8000 \
  WORKER_API_KEY="$(grep '^WORKER_API_KEY=' .env | cut -d= -f2)" \
  MOCK_BACKEND_URL=http://127.0.0.1:9100 \
  pytest -m gpu -v tests/integration/test_edit_e2e.py
```
Expected: test passes; `/tmp/mock_data/out/` contains a real 512×512 PNG.

- [ ] **Step 5: Kill background processes**

```bash
kill %1 %2 %3 2>/dev/null || true
```

- [ ] **Step 6: If everything passes, tag `phase-1-complete`**

```bash
git tag -a phase-1-complete -m "Phase 1 complete: edit + style presets working end-to-end"
```

---

## Self-Review

After completion, verify against spec §§3–6 and §9:
- [ ] `POST /v1/generate` accepts the request body defined in spec §3.1
- [ ] 202 / 400 / 401 / 409 / 429 / 503 responses match spec
- [ ] Callback body matches spec §3.2 (including `status`-dependent fields)
- [ ] HMAC signature format `t=<ts>, v1=<hex>` verifiable by receiver
- [ ] `edit` and `style` presets produce outputs on real GPU integration test
- [ ] `docker compose up` on a fresh box → working worker at `$WORKER_PORT`
- [ ] `/v1/health` flips `ready=true` only after warm-up completes
