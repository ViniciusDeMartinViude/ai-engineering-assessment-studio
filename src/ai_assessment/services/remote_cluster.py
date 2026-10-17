"""Unprivileged Assessment Studio client for the organizer GPU gateway."""
from __future__ import annotations
import os
from pathlib import Path
import requests

class RemoteClusterError(RuntimeError):
    pass

class RemoteClusterClient:
    def __init__(self, base_url: str, token: str, candidate_id: str):
        if not base_url.startswith(("http://", "https://")):
            raise RemoteClusterError("Specify an HTTP(S) organizer gateway URL")
        if not token:
            raise RemoteClusterError("Candidate gateway token is missing")
        self.url = base_url.rstrip("/")
        self.token = token
        self.candidate_id = candidate_id

    def request(self, method: str, path: str, **kwargs):
        try:
            response = requests.request(method, self.url + path,
                headers={"Authorization": "Bearer " + self.token},
                timeout=(5, 30), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise RemoteClusterError(str(exc)) from exc

    def submit(self, model: str, dataset: str, epochs: int, imgsz: int, batch: int):
        return self.request("POST", "/training/jobs", json={
            "model": model, "dataset": dataset, "epochs": epochs,
            "imgsz": imgsz, "batch": batch}).json()

    def jobs(self):
        return self.request("GET", "/training/jobs").json()

    def cancel(self, job_id: str):
        return self.request("POST", f"/training/jobs/{job_id}/cancel").json()

    def download(self, job_id: str, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = self.request("GET", f"/training/jobs/{job_id}/artifact", stream=True)
        temp = destination.with_suffix(".zip.part")
        try:
            with temp.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            os.replace(temp, destination)
        finally:
            temp.unlink(missing_ok=True)
        return destination
