from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .events import EventLog, utc_now_iso


WorkspaceMode = Literal["training", "assessment"]


WORKSPACE_DIRECTORIES = (
    "dataset",
    "src",
    "models",
    "camera",
    "calibration",
    "results",
    "exports",
    "logs",
    "submission",
)


@dataclass(frozen=True)
class SessionMetadata:
    candidate_id: str
    session_id: str
    mode: WorkspaceMode
    created_at: str
    updated_at: str
    schema_version: str = "m1.session.v1"


@dataclass(frozen=True)
class ProjectMetadata:
    workspace_schema: str = "m1.workspace.v1"
    dataset_source: str | None = None
    assessment_material: str = "client-visible-only"
    capabilities: dict[str, bool] = field(
        default_factory=lambda: {
            "events": True,
            "event_acknowledgements": True,
            "py_side_shell": True,
        }
    )


class CandidateWorkspace:
    def __init__(self, root: Path, session: SessionMetadata) -> None:
        self.root = root.resolve()
        self.session = session
        self.events = EventLog(
            self.root / "logs" / "events.jsonl",
            self.root / "logs" / "event_acks.jsonl",
        )

    @property
    def session_path(self) -> Path:
        return self.root / "session.json"

    @property
    def project_path(self) -> Path:
        return self.root / "project.json"

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        candidate_id: str,
        mode: WorkspaceMode = "training",
    ) -> "CandidateWorkspace":
        workspace_root = root.resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)
        for directory in WORKSPACE_DIRECTORIES:
            (workspace_root / directory).mkdir(parents=True, exist_ok=True)

        now = utc_now_iso()
        session_path = workspace_root / "session.json"
        if session_path.exists():
            return cls.open(workspace_root)

        session = SessionMetadata(
            candidate_id=candidate_id,
            session_id=str(uuid.uuid4()),
            mode=mode,
            created_at=now,
            updated_at=now,
        )
        project = ProjectMetadata()
        session_path.write_text(
            json.dumps(asdict(session), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (workspace_root / "project.json").write_text(
            json.dumps(asdict(project), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        workspace = cls(workspace_root, session)
        workspace.events.append(
            event_type="workspace.created",
            candidate_id=session.candidate_id,
            session_id=session.session_id,
            payload={"mode": mode, "workspace_root": str(workspace_root)},
        )
        return workspace

    @classmethod
    def open(cls, root: Path) -> "CandidateWorkspace":
        workspace_root = root.resolve()
        session_path = workspace_root / "session.json"
        if not session_path.exists():
            raise FileNotFoundError(f"No session.json found in {workspace_root}")
        session = SessionMetadata(**json.loads(session_path.read_text(encoding="utf-8")))
        for directory in WORKSPACE_DIRECTORIES:
            (workspace_root / directory).mkdir(parents=True, exist_ok=True)
        if not (workspace_root / "project.json").exists():
            (workspace_root / "project.json").write_text(
                json.dumps(asdict(ProjectMetadata()), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return cls(workspace_root, session)

    def resolve_inside(self, *parts: str | Path) -> Path:
        candidate = self.root.joinpath(*parts).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"Path escapes candidate workspace: {candidate}")
        return candidate

    def record_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        outcome: str = "recorded",
        artifact_hashes: dict[str, str] | None = None,
    ):
        return self.events.append(
            event_type=event_type,
            candidate_id=self.session.candidate_id,
            session_id=self.session.session_id,
            payload=payload,
            outcome=outcome,
            artifact_hashes=artifact_hashes,
        )
