"""Direct client for the GPU Cluster manager; trusted operator workstations only."""
from __future__ import annotations
import os
import re
from pathlib import Path
import requests

class RemoteClusterError(RuntimeError):
    pass

class RemoteClusterClient:
    def __init__(self, base_url: str, api_key: str, candidate_id: str):
        if not base_url.startswith(("http://", "https://")):
            raise RemoteClusterError("Enter a cluster manager HTTP(S) URL")
        if not api_key:
            raise RemoteClusterError("Cluster API key is required")
        if not re.fullmatch(r"C[0-9]+", candidate_id):
            raise RemoteClusterError("Invalid candidate identifier")
        self.url = base_url.rstrip("/")
        self.key = api_key
        self.candidate_id = candidate_id

    def request(self, method: str, path: str, **kwargs):
        try:
            resp = requests.request(method, self.url + path,
                headers={"x-api-key": self.key}, timeout=(5, 60), **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            raise RemoteClusterError(str(exc)) from exc

    def submit(self, model: str, dataset: str, epochs: int, imgsz: int, batch: int):
        return self.request("POST", "/jobs", json={
            "student_id": self.candidate_id, "mode": "train",
            "model": model, "dataset": dataset, "epochs": epochs,
            "imgsz": imgsz, "batch": batch}).json()

    def jobs(self):
        rows = self.request("GET", "/jobs").json()
        if isinstance(rows, dict):
            rows = rows.get("jobs", [])
        if not isinstance(rows, list):
            raise RemoteClusterError("Unexpected /jobs response")
        return [job for job in rows if isinstance(job, dict)
                and job.get("student_id") == self.candidate_id]

    def cancel(self, job_id: str):
        self._check_owned(job_id)
        return self.request("POST", f"/jobs/{job_id}/cancel").json()

    def _check_owned(self, job_id: str):
        # UI filtering only: cluster API key still grants administrator privileges.
        job = self.request("GET", f"/jobs/{job_id}").json()
        if job.get("student_id") != self.candidate_id:
            raise RemoteClusterError("Job does not belong to this workspace")

    def download(self, job_id: str, destination: Path):
        self._check_owned(job_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = self.request("GET", f"/jobs/{job_id}/artifact", stream=True)
        temp = destination.with_suffix(".zip.part")
        try:
            with temp.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk: handle.write(chunk)
            os.replace(temp, destination)
        finally:
            response.close()
            temp.unlink(missing_ok=True)
        return destination
