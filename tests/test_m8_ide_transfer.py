from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication

from ai_assessment.core.workspace import CandidateWorkspace
from ai_assessment.modules.ide import CodeEditor, IDEPage
from ai_assessment.services.ai_ide_transfer import CodeBlock, IDEHandoff, extract_code_blocks, sha256_text


class M8TransferContractTests(unittest.TestCase):
    def test_single_and_multiple_fenced_blocks_keep_boundaries(self) -> None:
        response = "before\n```python\nprint(1)\n```\nmiddle\n```javascript\nconsole.log(2)\n```\nafter"
        blocks = extract_code_blocks(response)
        self.assertEqual([block.block_id for block in blocks], ["block-1", "block-2"])
        self.assertEqual([block.language for block in blocks], ["python", "javascript"])
        self.assertEqual(blocks[0].code, "print(1)")
        self.assertEqual(blocks[1].code, "console.log(2)")

    def test_no_code_block_is_empty_and_untrusted_path_is_not_parsed(self) -> None:
        self.assertEqual(extract_code_blocks("plain answer"), ())
        block = CodeBlock("block-1", "python", "print(1)", sha256_text("print(1)"), "../../escape.py")
        self.assertEqual(block.suggested_path, "../../escape.py")


class M8QtHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = CandidateWorkspace.create(Path(self.temp_dir.name) / "C014", candidate_id="C014")
        self.page = IDEPage(self.workspace)
        self.block = CodeBlock("block-1", "python", "print('generated')\n", sha256_text("print('generated')\n"), "..\\..\\escape.py")
        self.handoff = IDEHandoff("response-1", self.block)

    def tearDown(self) -> None:
        self.page.execution.shutdown()
        self.page.terminal.shutdown()
        self.page.deleteLater()
        self.temp_dir.cleanup()

    def test_new_file_is_unsaved_and_never_runs_or_saves(self) -> None:
        self.page._new_file()
        self.assertTrue(self.page._apply_handoff(self.handoff, "new_file"))
        editor = self.page._current_editor()
        self.assertIsNotNone(editor)
        self.assertIsNone(editor.property("path"))
        self.assertTrue(editor.document().isModified())
        self.assertEqual(self.page.execution.active_run_id, None)
        self.assertFalse((self.workspace.root / "src" / "escape.py").exists())
        self.assertTrue(any(event.event_type == "ai.ide_handoff" for event in self.workspace.events.read_events()))

    def test_insert_preserves_existing_dirty_tab_without_saving(self) -> None:
        self.page._new_file()
        editor = self.page._current_editor()
        editor.setPlainText("prefix\n")
        editor.moveCursor(QTextCursor.MoveOperation.End)
        editor.document().setModified(True)
        self.assertTrue(self.page._apply_handoff(self.handoff, "insert"))
        self.assertIn("generated", editor.toPlainText())
        self.assertTrue(editor.document().isModified())
        self.assertFalse(list((self.workspace.root / "src").glob("*.py")))

    def test_replace_cancellation_and_stale_selection_are_detected(self) -> None:
        self.page._new_file()
        editor = self.page._current_editor()
        editor.setPlainText("old\n")
        editor.document().setModified(False)
        editor.selectAll()
        source = IDEHandoff(
            "response-2",
            self.block,
            source_path=None,
            source_file_hash=sha256_text("old\n"),
            source_selection_hash=sha256_text("old\n"),
        )
        editor.setPlainText("changed\n")
        self.assertTrue(self.page._handoff_stale(source))
        before = editor.toPlainText()
        self.assertFalse(self.page._apply_handoff(source, "cancel"))
        self.assertEqual(editor.toPlainText(), before)

    def test_handoff_does_not_accept_model_path(self) -> None:
        self.page._new_file()
        self.assertTrue(self.page._apply_handoff(self.handoff, "new_file"))
        editor = self.page._current_editor()
        self.assertIsNone(editor.property("path"))
        self.assertFalse((Path(self.temp_dir.name) / "escape.py").exists())


if __name__ == "__main__":
    unittest.main()
