from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "m1.event.v1"


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with second-independent precision."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class EventEnvelope:
    event_type: str
    candidate_id: str
    session_id: str
    sequence: int
    payload: dict[str, Any] = field(default_factory=dict)
    outcome: str = "recorded"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION
    client_timestamp: str = field(default_factory=utc_now_iso)
    server_timestamp: str | None = None
    artifact_hashes: dict[str, str] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventEnvelope":
        return cls(
            event_type=data["event_type"],
            candidate_id=data["candidate_id"],
            session_id=data["session_id"],
            sequence=int(data["sequence"]),
            payload=dict(data.get("payload", {})),
            outcome=data.get("outcome", "recorded"),
            event_id=data["event_id"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            client_timestamp=data["client_timestamp"],
            server_timestamp=data.get("server_timestamp"),
            artifact_hashes=dict(data.get("artifact_hashes", {})),
        )


@dataclass(frozen=True)
class EventAcknowledgement:
    event_id: str
    receipt_id: str
    acknowledged_at: str = field(default_factory=utc_now_iso)
    server_timestamp: str | None = None

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventAcknowledgement":
        return cls(
            event_id=data["event_id"],
            receipt_id=data["receipt_id"],
            acknowledged_at=data["acknowledged_at"],
            server_timestamp=data.get("server_timestamp"),
        )


class EventLog:
    """Append-only local event queue plus idempotent receipt tracking."""

    def __init__(self, event_path: Path, acknowledgement_path: Path) -> None:
        self.event_path = event_path
        self.acknowledgement_path = acknowledgement_path
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.acknowledgement_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_path.touch(exist_ok=True)
        self.acknowledgement_path.touch(exist_ok=True)

    def next_sequence(self) -> int:
        return len(self.read_events()) + 1

    def append(
        self,
        *,
        event_type: str,
        candidate_id: str,
        session_id: str,
        payload: dict[str, Any] | None = None,
        outcome: str = "recorded",
        artifact_hashes: dict[str, str] | None = None,
    ) -> EventEnvelope:
        event = EventEnvelope(
            event_type=event_type,
            candidate_id=candidate_id,
            session_id=session_id,
            sequence=self.next_sequence(),
            payload=payload or {},
            outcome=outcome,
            artifact_hashes=artifact_hashes or {},
        )
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(event.to_json_line() + "\n")
        return event

    def acknowledge(
        self,
        event_id: str,
        *,
        receipt_id: str,
        server_timestamp: str | None = None,
    ) -> EventAcknowledgement:
        existing = self.read_acknowledgements().get(event_id)
        if existing is not None:
            return existing

        event_ids = {event.event_id for event in self.read_events()}
        if event_id not in event_ids:
            raise KeyError(f"Cannot acknowledge unknown event: {event_id}")

        acknowledgement = EventAcknowledgement(
            event_id=event_id,
            receipt_id=receipt_id,
            server_timestamp=server_timestamp,
        )
        with self.acknowledgement_path.open("a", encoding="utf-8") as handle:
            handle.write(acknowledgement.to_json_line() + "\n")
        return acknowledgement

    def read_events(self) -> list[EventEnvelope]:
        return [
            EventEnvelope.from_dict(json.loads(line))
            for line in self.event_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def read_acknowledgements(self) -> dict[str, EventAcknowledgement]:
        acknowledgements: dict[str, EventAcknowledgement] = {}
        for line in self.acknowledgement_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            acknowledgement = EventAcknowledgement.from_dict(json.loads(line))
            acknowledgements.setdefault(acknowledgement.event_id, acknowledgement)
        return acknowledgements

    def pending_events(self) -> list[EventEnvelope]:
        acknowledged = self.read_acknowledgements()
        return [
            event for event in self.read_events() if event.event_id not in acknowledged
        ]
