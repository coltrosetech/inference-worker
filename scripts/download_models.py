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
    print(f"  -> downloading {entry.name} ({entry.size_mb} MB)", flush=True)
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
