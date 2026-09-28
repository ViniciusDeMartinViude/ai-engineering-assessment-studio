"""Capture clean offline screenshots for the student visual guide."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "student-guide"


def capture() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    with tempfile.TemporaryDirectory(prefix="assessment-guide-") as workspace_dir:
        os.environ["AI_ASSESSMENT_WORKSPACE"] = workspace_dir

        from PySide6.QtWidgets import QApplication, QLabel

        from ai_assessment import app as application

        def capture_exec(qt_app: QApplication) -> int:
            windows = [window for window in QApplication.topLevelWidgets() if window.windowTitle() == "AI Engineering Assessment Studio"]
            if len(windows) != 1:
                raise RuntimeError(f"Expected one application window, found {len(windows)}")
            window = windows[0]
            window.resize(1440, 900)
            window.show()
            qt_app.processEvents()
            for label in window.findChildren(QLabel, "workspacePath"):
                label.setText(r"Workspace: C:\AIEngineering\candidates\C014")
            OUTPUT.mkdir(parents=True, exist_ok=True)
            pages = (
                (0, "01-home.png"),
                (1, "02-dataset.png"),
                (2, "03-training.png"),
                (3, "04-vision.png"),
                (4, "05-calibration.png"),
                (5, "06-robot.png"),
                (6, "07-ide.png"),
                (7, "08-ai-assistant.png"),
                (8, "09-results.png"),
                (9, "10-submission.png"),
            )
            for index, filename in pages:
                window.nav.setCurrentRow(index)
                qt_app.processEvents()
                if not window.grab().save(str(OUTPUT / filename)):
                    raise RuntimeError(f"Could not save {filename}")
            window.shutdown()
            return 0

        QApplication.exec = capture_exec
        return application.main([])


if __name__ == "__main__":
    raise SystemExit(capture())
