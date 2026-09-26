from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_assessment.core.workspace import CandidateWorkspace, WORKSPACE_DIRECTORIES


class CandidateWorkspaceTests(unittest.TestCase):
    def test_create_workspace_writes_expected_structure_and_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "C014"

            workspace = CandidateWorkspace.create(root, candidate_id="C014")

            self.assertTrue((root / "session.json").exists())
            self.assertTrue((root / "project.json").exists())
            for directory in WORKSPACE_DIRECTORIES:
                self.assertTrue((root / directory).is_dir(), directory)
            self.assertEqual(workspace.session.candidate_id, "C014")
            self.assertEqual(len(workspace.events.read_events()), 1)

            reopened = CandidateWorkspace.open(root)

            self.assertEqual(reopened.session.session_id, workspace.session.session_id)
            self.assertEqual(reopened.session.candidate_id, "C014")

    def test_event_acknowledgements_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = CandidateWorkspace.create(
                Path(temp_dir) / "C014",
                candidate_id="C014",
            )
            event = workspace.record_event("workspace.reopened", {"source": "test"})

            first_ack = workspace.events.acknowledge(
                event.event_id,
                receipt_id="receipt-001",
                server_timestamp="2026-09-26T08:00:00Z",
            )
            second_ack = workspace.events.acknowledge(
                event.event_id,
                receipt_id="receipt-ignored",
                server_timestamp="2026-09-26T08:05:00Z",
            )

            self.assertEqual(first_ack, second_ack)
            self.assertEqual(first_ack.receipt_id, "receipt-001")
            self.assertEqual(len(workspace.events.read_acknowledgements()), 1)
            pending_ids = {item.event_id for item in workspace.events.pending_events()}
            self.assertNotIn(event.event_id, pending_ids)

    def test_resolve_inside_rejects_workspace_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = CandidateWorkspace.create(Path(temp_dir) / "C014", candidate_id="C014")

            with self.assertRaises(ValueError):
                workspace.resolve_inside("..", "other-candidate", "submission.json")


if __name__ == "__main__":
    unittest.main()
