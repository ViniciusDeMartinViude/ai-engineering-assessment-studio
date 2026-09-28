from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .core.workspace import CandidateWorkspace
from .modules.pages import PAGES, PageDefinition


DEFAULT_CANDIDATE_ID = "C014"


def default_workspace_root() -> Path:
    configured = os.environ.get("AI_ASSESSMENT_WORKSPACE")
    if configured:
        return Path(configured)
    return Path.cwd() / "candidate_workspaces" / DEFAULT_CANDIDATE_ID


def load_workspace(path: Path, candidate_id: str | None = None) -> CandidateWorkspace:
    if (path / "session.json").exists():
        workspace = CandidateWorkspace.open(path)
        if (
            candidate_id is not None
            and candidate_id != workspace.session.candidate_id
        ):
            raise ValueError(
                "Candidate ID mismatch: workspace contains "
                f"{workspace.session.candidate_id!r}, but --candidate-id requested "
                f"{candidate_id!r}."
            )
        workspace.record_event("workspace.opened", {"source": "app"})
        return workspace
    return CandidateWorkspace.create(
        path,
        candidate_id=candidate_id or DEFAULT_CANDIDATE_ID,
        mode="training",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Engineering Assessment Studio")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=default_workspace_root(),
        help="Candidate workspace folder. Defaults to ./candidate_workspaces/C014.",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help=(
            "Candidate identifier for a newly created workspace. "
            f"Defaults to {DEFAULT_CANDIDATE_ID}; when reopening, omit it or match session.json."
        ),
    )
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="Create/open the workspace and print a compact status without launching Qt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        workspace = load_workspace(args.workspace, args.candidate_id)
    except ValueError as exc:
        parser.error(str(exc))

    if args.headless_check:
        pending = len(workspace.events.pending_events())
        print(
            f"workspace={workspace.root} candidate={workspace.session.candidate_id} "
            f"session={workspace.session.session_id} pending_events={pending}"
        )
        return 0

    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QFrame,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QStackedWidget,
            QVBoxLayout,
            QWidget,
        )
        from .modules.calibration import CalibrationPage
        from .modules.dataset import DatasetPage
        from .modules.ide import IDEPage
        from .modules.robot import RobotPage
        from .modules.results import ResultsPage
        from .modules.training import TrainingPage
        from .modules.vision import VisionPage
    except ImportError as exc:
        print(
            "PySide6, NumPy, OpenCV, and PyYAML are required to launch the desktop shell. "
            "Run with --headless-check to verify workspace services.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    class Page(QWidget):
        def __init__(self, definition: PageDefinition) -> None:
            super().__init__()
            layout = QVBoxLayout(self)
            layout.setContentsMargins(28, 28, 28, 28)
            layout.setSpacing(14)

            title = QLabel(definition.title)
            title.setObjectName("pageTitle")
            title.setAlignment(Qt.AlignmentFlag.AlignLeft)

            milestone = QLabel(definition.milestone)
            milestone.setObjectName("milestone")
            milestone.setFixedWidth(56)
            milestone.setAlignment(Qt.AlignmentFlag.AlignCenter)

            heading = QHBoxLayout()
            heading.addWidget(title)
            heading.addStretch(1)
            heading.addWidget(milestone)

            summary = QLabel(definition.summary)
            summary.setWordWrap(True)
            summary.setObjectName("summary")

            workspace_label = QLabel(f"Workspace: {workspace.root}")
            workspace_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            workspace_label.setWordWrap(True)
            workspace_label.setObjectName("workspacePath")

            layout.addLayout(heading)
            layout.addWidget(summary)
            layout.addSpacing(12)
            layout.addWidget(workspace_label)
            layout.addStretch(1)

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("AI Engineering Assessment Studio")
            self.resize(1160, 760)

            container = QWidget()
            root = QHBoxLayout(container)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            sidebar_frame = QFrame()
            sidebar_frame.setObjectName("sidebar")
            sidebar_layout = QVBoxLayout(sidebar_frame)
            sidebar_layout.setContentsMargins(18, 20, 18, 20)
            sidebar_layout.setSpacing(12)

            product = QLabel("Assessment Studio")
            product.setObjectName("product")
            candidate = QLabel(
                f"{workspace.session.candidate_id}  |  {workspace.session.mode.title()}"
            )
            candidate.setObjectName("candidate")

            self.nav = QListWidget()
            self.nav.setObjectName("navigation")
            for page in PAGES:
                self.nav.addItem(QListWidgetItem(page.title))

            sidebar_layout.addWidget(product)
            sidebar_layout.addWidget(candidate)
            sidebar_layout.addWidget(self.nav, 1)

            self.stack = QStackedWidget()
            self.calibration_page = None
            self.dataset_page = None
            self.vision_page = None
            self.robot_page = None
            self.training_page = None
            self.results_page = None
            self.ide_page = None
            for page in PAGES:
                if page.title == "Calibration":
                    self.calibration_page = CalibrationPage(workspace)
                    self.stack.addWidget(self.calibration_page)
                elif page.title == "Dataset":
                    self.dataset_page = DatasetPage(workspace)
                    self.stack.addWidget(self.dataset_page)
                elif page.title == "Vision":
                    self.vision_page = VisionPage(workspace)
                    self.stack.addWidget(self.vision_page)
                elif page.title == "Robot":
                    self.robot_page = RobotPage(workspace)
                    self.stack.addWidget(self.robot_page)
                elif page.title == "Training":
                    self.training_page = TrainingPage(workspace)
                    self.stack.addWidget(self.training_page)
                elif page.title == "Results":
                    self.results_page = ResultsPage(workspace)
                    self.stack.addWidget(self.results_page)
                elif page.title == "IDE":
                    self.ide_page = IDEPage(workspace)
                    self.stack.addWidget(self.ide_page)
                else:
                    self.stack.addWidget(Page(page))
            self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
            self.nav.setCurrentRow(0)

            root.addWidget(sidebar_frame)
            root.addWidget(self.stack, 1)
            self.setCentralWidget(container)
            self.statusBar().showMessage(self.status_message())

        def status_message(self) -> str:
            total = len(workspace.events.read_events())
            pending = len(workspace.events.pending_events())
            return f"{total} local events, {pending} awaiting acknowledgement"

        def shutdown(self) -> None:
            if self.calibration_page is not None:
                self.calibration_page.shutdown()
            if self.dataset_page is not None:
                self.dataset_page.shutdown()
            if self.vision_page is not None:
                self.vision_page.shutdown()
            if self.robot_page is not None:
                self.robot_page.shutdown()
            if self.training_page is not None:
                self.training_page.shutdown()
            if self.results_page is not None:
                self.results_page.shutdown()
            if self.ide_page is not None:
                self.ide_page.shutdown()

    app = QApplication([sys.argv[0]])
    app.setStyleSheet(
        """
        QMainWindow, QWidget {
            background: #f8faf9;
            color: #17211d;
            font-family: Segoe UI, Arial, sans-serif;
            font-size: 14px;
        }
        #sidebar {
            background: #ecf1ef;
            border-right: 1px solid #d5ded9;
            min-width: 250px;
            max-width: 250px;
        }
        #product {
            font-size: 20px;
            font-weight: 700;
        }
        #candidate {
            color: #53635c;
        }
        #navigation {
            background: transparent;
            border: 0;
            outline: 0;
        }
        #navigation::item {
            min-height: 34px;
            padding: 6px 10px;
            border-radius: 6px;
        }
        #navigation::item:selected {
            background: #d7e7df;
            color: #0c3525;
        }
        #pageTitle {
            font-size: 30px;
            font-weight: 700;
        }
        #milestone {
            background: #174c36;
            color: white;
            border-radius: 6px;
            padding: 5px;
            font-weight: 700;
        }
        #summary {
            color: #394942;
            font-size: 16px;
        }
        #workspacePath {
            color: #56665f;
            background: #eef4f1;
            border: 1px solid #d6e1dc;
            border-radius: 6px;
            padding: 10px;
        }
        """
    )
    window = MainWindow()
    app.aboutToQuit.connect(window.shutdown)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
