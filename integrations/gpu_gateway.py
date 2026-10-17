"""Organizer-controlled gateway for candidate remote GPU jobs.

Run behind an HTTPS reverse proxy; API_KEY stays on management host.
This pilot accepts only centrally staged dataset IDs.
"""
from __future__ import annotations
import hmac
import json
import os
import re
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import requests

app = FastAPI(title="Assessment Studio GPU Gateway")
SAFE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MANAGER = os.getenv("CLUSTER_MANAGER_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("CLUSTER_API_KEY", "")
TOKENS_FILE = Path(os.environ.get("GATEWAY_TOKENS_FILE", "./candidate_tokens.json"))
DATASETS = set(os.getenv("GATEWAY_DATASETS", "dates-v16").split(","))

class Submission(BaseModel):
    model: str
    dataset: str
    epochs: int = Field(ge=1, le=200)
    imgsz: int = Field(ge=320, le=640)
    batch: int = Field(ge=1, le=8)

def candidate(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Candidate token required")
    if not TOKENS_FILE.is_file():
        raise HTTPException(503, "Organizer candidate token registry not configured")
    mapping = json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
    token = authorization[7:]
    for expected, identity in mapping.items():
        if hmac.compare_digest(token, expected):
            if not re.fullmatch(r"C[0-9]+", identity):
                raise HTTPException(503, "Invalid candidate mapping")
            return identity
    raise HTTPException(401, "Invalid candidate token")

def forward(method: str, path: str, **kwargs):
    if not API_KEY:
        raise HTTPException(503, "Management credential missing")
    try:
        resp = requests.request(method, MANAGER + path,
            headers={"x-api-key": API_KEY}, timeout=(5, 40), **kwargs)
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, resp.text[:400])
        return resp
    except requests.RequestException:
        raise HTTPException(502, "Cluster management unavailable")

def owned(jobid: str, identity: str):
    if not re.fullmatch(r"[a-f0-9]{32}", jobid):
        raise HTTPException(404, "Unknown job")
    data = forward("GET", "/jobs/" + jobid).json()
    if data.get("student_id") != identity:
        raise HTTPException(404, "Unknown job")
    return data

@app.post("/training/jobs")
def submit(payload: Submission, authorization: str | None = Header(default=None)):
    identity = candidate(authorization)
    if not SAFE.fullmatch(payload.dataset) or payload.dataset not in DATASETS:
        raise HTTPException(422, "Dataset not published for remote training")
    if payload.model not in ("yolo26n.pt", "yolo26s.pt"):
        raise HTTPException(422, "Model not approved")
    if payload.imgsz % 32:
        raise HTTPException(422, "Image size must be divisible by 32")
    return forward("POST", "/jobs", json={"student_id":identity,"mode":"train",**payload.model_dump()}).json()

@app.get("/training/jobs")
def jobs(authorization: str | None = Header(default=None)):
    identity = candidate(authorization)
    rows = forward("GET", "/jobs", params={"limit":200}).json()
    return [j for j in rows if j.get("student_id") == identity]

@app.get("/training/jobs/{jobid}")
def job(jobid: str, authorization: str | None = Header(default=None)):
    return owned(jobid, candidate(authorization))

@app.post("/training/jobs/{jobid}/cancel")
def cancel(jobid: str, authorization: str | None = Header(default=None)):
    owned(jobid, candidate(authorization))
    return forward("POST", "/jobs/" + jobid + "/cancel").json()

@app.get("/training/jobs/{jobid}/artifact")
def artifact(jobid: str, authorization: str | None = Header(default=None)):
    owned(jobid, candidate(authorization))
    # Stream response and close upstream when finished.
    if not API_KEY:
        raise HTTPException(503, "Management credential missing")
    try:
        upstream = requests.get(MANAGER + "/jobs/" + jobid + "/artifact",
            headers={"x-api-key": API_KEY}, stream=True, timeout=(5, 120))
        if upstream.status_code >= 400:
            status = upstream.status_code
            upstream.close()
            raise HTTPException(status, "Artifact unavailable")
    except requests.RequestException:
        raise HTTPException(502, "Artifact fetch failed")
    def chunks():
        try:
            yield from upstream.iter_content(chunk_size=1024 * 1024)
        finally:
            upstream.close()
    return StreamingResponse(chunks(),media_type="application/zip",
        headers={"Content-Disposition":f'attachment; filename="{jobid}.zip"'})
