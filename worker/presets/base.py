from __future__ import annotations

import abc
from enum import Enum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from worker.comfyui.workflow import WorkflowTemplate


class Mode(str, Enum):
    IMAGE = "image"
    IMAGE_PREMIUM = "image_premium"
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
    needs_reference_image: ClassVar[bool] = False
    needs_mask_image: ClassVar[bool] = False
    warmup_timeout_sec: ClassVar[float] = 180.0

    @classmethod
    def warmup_params(cls) -> dict:
        """Kwargs for `Parameters(...)` during warm-up. Subclasses may override."""
        return dict(prompt="warmup", steps=4, cfg=1.2, width=512, height=512, seed=0)

    class BaseParameters(BaseModel):
        model_config = ConfigDict(extra="forbid")
        seed: int | None = None
        steps: int = Field(default=6, ge=1, le=150)
        cfg: float = Field(default=1.8, ge=0.0, le=30.0)
        width: int = Field(default=1024, ge=64, le=4096)
        height: int = Field(default=1024, ge=64, le=4096)

        @field_validator("steps", mode="before")
        @classmethod
        def validate_steps(cls, v):
            if isinstance(v, int):
                if not (1 <= v <= 150):
                    raise ValueError(f"steps must be between 1 and 150, got {v}")
            return v

        @field_validator("cfg", mode="before")
        @classmethod
        def validate_cfg(cls, v):
            if isinstance(v, (int, float)):
                if not (0.0 <= v <= 30.0):
                    raise ValueError(f"cfg must be between 0.0 and 30.0, got {v}")
            return v

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
    """Validate raw dict against the preset's Parameters model."""
    return preset_cls.Parameters(**raw)
