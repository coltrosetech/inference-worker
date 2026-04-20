from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JobRecord:
    job_id: str
    out_name: str
    preset: str
    status: str = "queued"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    duration_ms: int | None = None
    stages_ms: dict[str, int] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    output_url: str | None = None
    output_kind: str | None = None


class Storage:
    """In-memory job registry + on-disk file buckets for uploads/outputs."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        (data_dir / "u").mkdir(parents=True, exist_ok=True)
        (data_dir / "o").mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    # --- files ---
    def upload_path(self, name: str) -> Path:
        return self.data_dir / "u" / name

    def output_path(self, name: str) -> Path:
        return self.data_dir / "o" / name

    # --- jobs ---
    def register(self, rec: JobRecord) -> None:
        with self._lock:
            self._jobs[rec.job_id] = rec

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            for k, v in fields.items():
                setattr(rec, k, v)

    def list(self, limit: int = 20) -> list[JobRecord]:
        with self._lock:
            items = sorted(self._jobs.values(), key=lambda r: r.started_at, reverse=True)
            return items[:limit]
