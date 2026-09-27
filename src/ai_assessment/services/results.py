"""Candidate-visible training results and production-model selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.workspace import CandidateWorkspace
from .training import TrainingError, TrainingService


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

