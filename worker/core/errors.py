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
