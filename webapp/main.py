from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from webapp.storage import JobRecord, Storage
from worker.core.hmac_sign import verify


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
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "file.png").suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, f"unsupported extension: {ext}")
    uid = uuid.uuid4().hex[:12]
    name = f"{uid}{ext}"
    p = storage.upload_path(name)
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "file too large (max 25 MB)")
    p.write_bytes(data)
    return {"name": name, "bytes": len(data)}


@app.get("/u/{name}")
async def serve_upload(name: str):
    p = storage.upload_path(name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/generate")
async def generate(body: GenerateBody):
    if body.preset not in {
        "edit", "style", "controlnet",
        "inpaint", "inpaint_sdxl", "inpaint_realvis", "inpaint_premium",
        "edit_premium",
        "ltx_video",
    }:
        raise HTTPException(400, f"unknown preset {body.preset}")

    job_id = f"web_{uuid.uuid4().hex[:10]}"
    ext = "mp4" if body.preset == "ltx_video" else "png"
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

    output_kind = "video/mp4" if body.preset == "ltx_video" else "image/png"
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
