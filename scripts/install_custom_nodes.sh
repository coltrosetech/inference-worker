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
