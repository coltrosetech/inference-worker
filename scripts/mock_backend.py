#!/usr/bin/env python
"""Local mock of the user's central backend. Runs as a standalone FastAPI service:

- PUT /storage/in/<id>.png       -> stores uploaded test images
- GET /storage/in/<id>.png       -> serves them (input to worker)
- PUT /storage/out/<id>          -> receives worker upload
- GET /storage/out/<id>          -> serves generated output
- POST /cb/<id>                  -> receives callback (verifies HMAC)
- GET /cb_peek/<id>              -> retrieves the parsed callback (for tests)

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
from fastapi.responses import FileResponse

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


@app.get("/cb_peek/{job_id}")
async def cb_peek(job_id: str):
    p = DIR / "cb" / f"{job_id}.json"
    if not p.exists():
        raise HTTPException(404)
    return json.loads(p.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    if args.dir:
        os.environ["MOCK_DIR"] = args.dir
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
