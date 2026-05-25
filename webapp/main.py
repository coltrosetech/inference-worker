from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import io
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from webapp.storage import JobRecord, Storage
from worker.core.hmac_sign import verify

SDXL_BUCKETS: list[tuple[int, int]] = [
    (1024, 1024), (1152, 896), (896, 1152),
    (1216, 832), (832, 1216), (1344, 768), (768, 1344),
]


def fit_to_sdxl_bucket(img: Image.Image) -> Image.Image:
    target_aspect = img.width / img.height
    bw, bh = min(SDXL_BUCKETS, key=lambda b: abs((b[0] / b[1]) - target_aspect))
    return ImageOps.fit(img, (bw, bh), method=Image.Resampling.LANCZOS)


DATA_DIR = Path(os.environ.get("WEBAPP_DATA_DIR", "/tmp/webapp_data"))
WORKER_URL = os.environ.get("WORKER_URL", "http://127.0.0.1:8000")
INTERNAL_BASE = os.environ.get("WEBAPP_INTERNAL_BASE", "http://127.0.0.1:8001")
WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "")
CALLBACK_SECRET = os.environ.get("CALLBACK_HMAC_SECRET", "")

if not WORKER_API_KEY:
    raise RuntimeError("WORKER_API_KEY must be set in env")
if not CALLBACK_SECRET:
    raise RuntimeError("CALLBACK_HMAC_SECRET must be set in env")


app = FastAPI(title="inference-worker-webapp", version="0.1.0")
storage = Storage(DATA_DIR)


class GenerateBody(BaseModel):
    preset: str
    prompt: str = ""
    negative_prompt: str = ""
    input_image_name: str
    mask_image_name: str | None = None
    reference_image_name: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_sec: int = 300


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), bucket: bool = True):
    ext = Path(file.filename or "file.png").suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, f"unsupported extension: {ext}")
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "file too large (max 25 MB)")

    uid = uuid.uuid4().hex[:12]

    if bucket:
        try:
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img).convert("RGB")
            fitted = fit_to_sdxl_bucket(img)
            buf = io.BytesIO()
            fitted.save(buf, format="PNG", optimize=False)
            data = buf.getvalue()
            ext = ".png"
            bucket_size = (fitted.width, fitted.height)
        except Exception as e:
            raise HTTPException(400, f"image decode failed: {e}") from e
    else:
        bucket_size = None

    name = f"{uid}{ext}"
    p = storage.upload_path(name)
    p.write_bytes(data)
    return {"name": name, "bytes": len(data), "bucket": bucket_size}


@app.get("/u/{name}")
async def serve_upload(name: str):
    p = storage.upload_path(name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/generate")
async def generate(body: GenerateBody):
    if body.preset not in {"tryon", "ltx_video", "wan_flf2v"}:
        raise HTTPException(400, f"unknown preset {body.preset}")

    job_id = f"web_{uuid.uuid4().hex[:10]}"
    ext = "mp4" if body.preset in {"ltx_video", "wan_flf2v"} else "png"
    out_name = f"{job_id}.{ext}"

    def u(name: str | None) -> str | None:
        return f"{INTERNAL_BASE}/u/{name}" if name else None

    payload: dict[str, Any] = {
        "job_id": job_id,
        "preset": body.preset,
        "prompt": body.prompt,
        "negative_prompt": body.negative_prompt,
        "input_image_url": u(body.input_image_name),
        "callback_url": f"{INTERNAL_BASE}/api/callback",
        "upload_url": f"{INTERNAL_BASE}/o/{out_name}",
        "upload_method": "PUT",
        "parameters": body.parameters,
        "timeout_sec": body.timeout_sec,
    }
    if body.mask_image_name:
        payload["mask_image_url"] = u(body.mask_image_name)
    if body.reference_image_name:
        payload["reference_image_url"] = u(body.reference_image_name)

    output_kind = "video/mp4" if body.preset in {"ltx_video", "wan_flf2v"} else "image/png"
    storage.register(JobRecord(
        job_id=job_id, out_name=out_name, preset=body.preset,
        status="submitting", output_kind=output_kind,
    ))

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{WORKER_URL}/v1/generate",
                headers={"Authorization": f"Bearer {WORKER_API_KEY}", "Idempotency-Key": job_id},
                json=payload,
            )
    except httpx.HTTPError as e:
        storage.update(job_id, status="failed",
                       error={"code": "WORKER_UNREACHABLE", "message": str(e)})
        return {"job_id": job_id, "status": "failed"}

    if r.status_code != 202:
        storage.update(job_id, status="failed",
                       error={"code": "WORKER_REJECTED", "message": r.text[:500]})
        return {"job_id": job_id, "status": "failed", "error": storage.get(job_id).error}

    storage.update(job_id, status="running")
    return {"job_id": job_id, "status": "running", "out_name": out_name}


@app.put("/o/{name}")
async def receive_output(name: str, request: Request):
    p = storage.output_path(name)
    data = await request.body()
    p.write_bytes(data)
    return {"ok": True, "bytes": len(data)}


@app.get("/o/{name}")
async def serve_output(name: str):
    p = storage.output_path(name)
    if not p.exists():
        raise HTTPException(404)
    ct = "video/mp4" if name.endswith(".mp4") else "image/png"
    return FileResponse(p, media_type=ct)


@app.post("/api/callback")
async def callback(request: Request,
                   x_worker_signature: str | None = Header(default=None, alias="X-Worker-Signature")):
    body = await request.body()
    if not verify(body, x_worker_signature or "", secret=CALLBACK_SECRET):
        return JSONResponse({"ok": False, "error": "bad signature"}, status_code=401)
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(400, "invalid json")
    jid = data.get("job_id")
    if not jid or not storage.get(jid):
        return {"ok": True, "known": False}
    rec = storage.get(jid)
    status = data.get("status")
    updates: dict[str, Any] = {
        "status": status,
        "duration_ms": data.get("duration_ms"),
        "stages_ms": data.get("stages_ms") or {},
        "error": data.get("error"),
        "finished_at": time.time(),
    }
    if status == "success":
        updates["output_url"] = f"/o/{rec.out_name}"
    storage.update(jid, **updates)
    return {"ok": True}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    rec = storage.get(job_id)
    if not rec:
        raise HTTPException(404)
    return rec.__dict__


@app.get("/api/jobs")
async def list_jobs(limit: int = 20):
    return [r.__dict__ for r in storage.list(limit=limit)]


@app.get("/api/health")
async def health():
    return {"ok": True, "worker_url": WORKER_URL, "jobs_active": len(storage.list(limit=1000))}


@app.get("/api/worker-health")
async def worker_health():
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{WORKER_URL}/v1/health")
        return r.json()
    except httpx.HTTPError as e:
        return JSONResponse(
            {"ok": False, "ready": False, "error": str(e)}, status_code=503,
        )


# --- live sampling progress (reads ComfyUI queue + log tail) ------------------
_COMFY_BASE = f"http://{os.environ.get('COMFYUI_HOST', '127.0.0.1')}:{os.environ.get('COMFYUI_INTERNAL_PORT', '18188')}"
_COMFY_LOG = os.environ.get("COMFYUI_LOG", "/var/log/portal/comfyui.log")
_STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[([^\]]*)\]")


@app.get("/api/progress")
async def progress():
    """Live sampling progress for the running job: ComfyUI queue + tqdm step
    line from its log so the UI can show 'step X/Y' in real time."""
    running = 0
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            q = (await client.get(f"{_COMFY_BASE}/queue")).json()
        running = len(q.get("queue_running", []))
    except Exception:
        pass
    step = total = None
    detail = ""
    try:
        with open(_COMFY_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", "replace")
        for chunk in reversed(re.split(r"[\r\n]", tail)):
            m = _STEP_RE.search(chunk)
            if m:
                step, total, detail = int(m.group(1)), int(m.group(2)), m.group(3).strip()
                break
    except Exception:
        pass
    return {"running": running, "step": step, "total": total, "detail": detail}


# --- docs viewer (renders docs/SAFE_MODE.md in the browser) -------------------
_DOCS_DIR = Path(__file__).parent.parent / "docs"


@app.get("/docs/safe-mode.md")
async def safe_mode_md():
    p = _DOCS_DIR / "SAFE_MODE.md"
    if not p.exists():
        raise HTTPException(404, "SAFE_MODE.md not found")
    return FileResponse(p, media_type="text/markdown; charset=utf-8")


@app.get("/docs/safe-mode", response_class=HTMLResponse)
async def safe_mode_page():
    return """<!doctype html><html lang="tr"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAFE MODE — inference-worker</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  body{margin:0;background:#0e0f12;color:#e6e6e6;font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:860px;margin:0 auto;padding:40px 24px 80px}
  h1,h2,h3{line-height:1.25} h1{border-bottom:1px solid #2a2c31;padding-bottom:.3em}
  h2{margin-top:2em;border-bottom:1px solid #23252a;padding-bottom:.25em}
  code{background:#1b1d22;padding:.15em .4em;border-radius:4px;font-size:.88em}
  pre{background:#15171b;padding:14px;border-radius:8px;overflow:auto}
  pre code{background:none;padding:0}
  a{color:#f5b942} table{border-collapse:collapse;width:100%;margin:1em 0}
  th,td{border:1px solid #2a2c31;padding:8px 10px;text-align:left;font-size:.92em}
  th{background:#1b1d22} blockquote{border-left:3px solid #f5b942;margin:1em 0;padding:.2em 1em;color:#b9b9b9}
  hr{border:none;border-top:1px solid #23252a;margin:2em 0}
</style></head><body><div class="wrap" id="content">yükleniyor…</div>
<script>
fetch('/docs/safe-mode.md').then(r=>r.text()).then(md=>{
  document.getElementById('content').innerHTML = marked.parse(md);
}).catch(e=>{document.getElementById('content').textContent='hata: '+e});
</script></body></html>"""


# --- static frontend (Vite build output) -------------------------------------
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


@app.get("/")
async def index():
    idx = FRONTEND_DIST / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return JSONResponse(
        {"error": "frontend not built",
         "hint": "cd webapp/frontend && npm install && npm run build"},
        status_code=503,
    )


if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
