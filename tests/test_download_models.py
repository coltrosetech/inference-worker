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
