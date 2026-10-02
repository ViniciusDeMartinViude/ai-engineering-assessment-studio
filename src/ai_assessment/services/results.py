"""Candidate-visible training results and production-model selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.workspace import CandidateWorkspace
from .training import TrainingError, TrainingService, sha256_file
from .vision import validate_model_path


@dataclass(frozen=True)
class CheckpointReference:
    run_id: str
    path: Path
    sha256: str
    source: str
    selected: bool = False
    status: str = "unknown"
    message: str = ""
    started_at: str = ""
    bytes: int | None = None

    @property
    def is_available(self) -> bool:
        return self.status == "available"


class ResultsService:
    def __init__(self, workspace: CandidateWorkspace) -> None:
        self.workspace = workspace
        self.training = TrainingService(workspace)

    def runs(self) -> list[dict[str, Any]]:
        return self.training.list_runs()

    def selected_model(self) -> dict[str, Any] | None:
        path = self.workspace.resolve_inside("results", "selected_model.json")
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise TrainingError(f"Selected model record is unreadable: {error}") from error
        return value if isinstance(value, dict) else None

    def select_model(self, run_id: str, checkpoint_path: Path | None = None) -> Path:
        return self.training.select_production_model(run_id, checkpoint_path)

    def selected_checkpoint(self) -> CheckpointReference | None:
        selection = self.selected_model()
        if selection is None:
            return None
        run_id = str(selection.get("run_id", ""))
        raw_path = selection.get("path")
        sha = str(selection.get("sha256", ""))
        if not run_id or not isinstance(raw_path, str) or not sha:
            raise TrainingError("Selected model record is incomplete; reselect a completed run in Results.")
        record = self.training.load_run(run_id)
        recorded = self._checkpoint_item_for_path(record, raw_path)
        if recorded is None or recorded.get("sha256") != sha:
            raise TrainingError("Selected model record no longer matches the recorded training run. Reselect the run in Results.")
        reference = CheckpointReference(
            run_id=run_id,
            path=self._resolve_candidate_path(raw_path),
            sha256=sha,
            source="results_selection",
            selected=True,
        )
        return self._with_status(reference)

    def completed_best_checkpoints(self) -> list[CheckpointReference]:
        selected = self.selected_model()
        selected_run = selected.get("run_id") if selected else None
        selected_path = selected.get("path") if selected else None
        references: list[CheckpointReference] = []
        for record in self.training.list_runs():
            if record.get("status") != "completed":
                continue
            item = self._best_checkpoint_item(record)
            if item is None:
                continue
            raw_path = item.get("path")
            sha = item.get("sha256")
            if not isinstance(raw_path, str) or not isinstance(sha, str):
                continue
            run_id = str(record.get("run_id", ""))
            try:
                path = self._resolve_candidate_path(raw_path)
            except TrainingError:
                continue
            references.append(
                self._with_status(
                    CheckpointReference(
                        run_id=run_id,
                        path=path,
                        sha256=sha,
                        source="training_run",
                        selected=selected_run == run_id and selected_path == raw_path,
                        started_at=str(record.get("started_at", "")),
                        bytes=int(item["bytes"]) if isinstance(item.get("bytes"), int) else None,
                    )
                )
            )
        return sorted(references, key=lambda reference: reference.started_at, reverse=True)

    def verify_checkpoint(self, reference: CheckpointReference) -> Path:
        path = self._resolve_candidate_path(reference.path)
        self._verify_trained_checkpoint_path(path)
        validate_model_path(path)
        current = sha256_file(path)
        if current.lower() != reference.sha256.lower():
            raise TrainingError(
                "Checkpoint SHA-256 does not match the recorded training artifact. "
                f"Expected {reference.sha256}, found {current}. The file may have been changed."
            )
        return path

    def _with_status(self, reference: CheckpointReference) -> CheckpointReference:
        try:
            self.verify_checkpoint(reference)
        except TrainingError as error:
            message = str(error)
            status = "missing" if "does not exist" in message or "moved or deleted" in message else "changed"
            return CheckpointReference(
                run_id=reference.run_id,
                path=reference.path,
                sha256=reference.sha256,
                source=reference.source,
                selected=reference.selected,
                status=status,
                message=message,
                started_at=reference.started_at,
                bytes=reference.bytes,
            )
        return CheckpointReference(
            run_id=reference.run_id,
            path=reference.path,
            sha256=reference.sha256,
            source=reference.source,
            selected=reference.selected,
            status="available",
            message="SHA-256 verified against the recorded training artifact.",
            started_at=reference.started_at,
            bytes=reference.bytes,
        )

    def _best_checkpoint_item(self, record: dict[str, Any]) -> dict[str, Any] | None:
        run_id = str(record.get("run_id", ""))
        expected = self.workspace.resolve_inside("models", run_id, "weights", "best.pt")
        for item in record.get("checkpoints", []) or []:
            raw = item.get("path") if isinstance(item, dict) else None
            if not isinstance(raw, str) or not raw.replace("\\", "/").lower().endswith("/weights/best.pt"):
                continue
            try:
                path = self._resolve_candidate_path(raw)
            except TrainingError:
                continue
            if path == expected:
                return item
        return None

    def _checkpoint_item_for_path(self, record: dict[str, Any], raw_path: str) -> dict[str, Any] | None:
        for item in record.get("checkpoints", []) or []:
            item_path = item.get("path") if isinstance(item, dict) else None
            if item_path == raw_path:
                return item
        return None

    def _resolve_candidate_path(self, raw: str | Path) -> Path:
        path = Path(raw)
        candidate = path if path.is_absolute() else self.workspace.root / path
        resolved = candidate.expanduser().resolve()
        root = self.workspace.root.resolve()
        if resolved != root and root not in resolved.parents:
            raise TrainingError(f"Checkpoint path escapes the candidate workspace: {resolved}")
        return resolved

    def _verify_trained_checkpoint_path(self, path: Path) -> None:
        resolved = self._resolve_candidate_path(path)
        relative = resolved.relative_to(self.workspace.root).parts
        if len(relative) >= 3 and relative[0] == "models" and relative[1] == "base_weights":
            raise TrainingError("Base weights are not a trained checkpoint. Select a completed training run instead.")
        if resolved.suffix.lower() != ".pt":
            raise TrainingError("Recorded training checkpoints must be .pt files.")
        if not resolved.is_file():
            raise TrainingError(f"Recorded checkpoint was moved or deleted: {resolved}")

