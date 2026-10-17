"""FastAPI router to install INSIDE the GPU Cluster management API.

Integration step required: manager/app.py must include this router, and its
dataset registry/worker download route must resolve the returned dataset_id.
This module alone does NOT make uploaded datasets trainable.
"""
from __future__ import annotations
import hashlib
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

router = APIRouter()
BASE = Path(os.getenv("CLUSTER_UPLOADED_DATASETS", "./datasets/uploads")).resolve()
MAX_BYTES = 1024 * 1024 * 1024

@router.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...), student_id: str = Form(...),
                   sha256: str = Form(...), x_api_key: str | None = Header(None)):
    expected = os.environ.get("API_KEY", "")
    if not expected or not __import__("hmac").compare_digest(x_api_key or "", expected):
        raise HTTPException(403, "Unauthorized")
    if not re.fullmatch(r"C[0-9]+", student_id):
        raise HTTPException(422, "Invalid student ID")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise HTTPException(422, "Invalid checksum")
    BASE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BASE) as folder:
        temporary = Path(folder)
        archive = temporary / "upload.zip"
        total = 0
        digest = hashlib.sha256()
        with archive.open("wb") as target:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(413, "Dataset ZIP too large")
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest() != sha256:
            raise HTTPException(422, "SHA256 mismatch")
        try:
            with zipfile.ZipFile(archive) as z:
                members = z.infolist()
                expanded = 0
                if len(members) > 250000:
                    raise HTTPException(422, "Too many files")
                for item in members:
                    p = Path(item.filename)
                    if (p.is_absolute() or ".." in p.parts or
                        "\\" in item.filename or
                        (item.external_attr >> 16) & 0o170000 == 0o120000):
                        raise HTTPException(422, "Unsafe archive entry")
                    expanded += item.file_size
                    if expanded > 4 * MAX_BYTES:
                        raise HTTPException(413, "Unpacked dataset too large")
                unpack = temporary / "unpacked"
                unpack.mkdir()
                z.extractall(unpack)
        except zipfile.BadZipFile:
            raise HTTPException(422, "Invalid ZIP file")
        if not (unpack / "manifest.json").is_file() or not (unpack / "data.yaml").is_file():
            raise HTTPException(422, "Missing Dataset Studio manifest or data.yaml")
        # Content-addressable name; copy atomically, never overwrite an existing dataset.
        dataset_id = f"uploaded-{student_id.lower()}-{sha256[:16]}"
        destination = BASE / dataset_id
        if destination.exists():
            return {"dataset_id": dataset_id, "sha256": sha256, "reused": True}
        os.rename(unpack, destination)
        return {"dataset_id": dataset_id, "sha256": sha256, "reused": False}
