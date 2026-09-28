"""M6 candidate-visible training comparison and model selection page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.workspace import CandidateWorkspace
from ..services.results import ResultsService
from ..services.training import TrainingError


class ResultsPage(QWidget):
    def __init__(self, workspace: CandidateWorkspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.service = ResultsService(workspace)
        self._records: list[dict[str, Any]] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(9)
        eyebrow = QLabel("RESULTS / M6")
        eyebrow.setObjectName("resultsEyebrow")
        title = QLabel("Results Studio")
        title.setObjectName("resultsTitle")
        subtitle = QLabel("Compare recorded validation runs and select one exact workspace checkpoint for Vision. These are candidate-visible results; organizer scores and locks are not present here.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("resultsSubtitle")
        root.addWidget(eyebrow)
        root.addWidget(title)
        root.addWidget(subtitle)
        self.status = QLabel("No training results loaded.")
        self.status.setObjectName("resultsStatus")
        root.addWidget(self.status)
        actions = QHBoxLayout()
        refresh = QPushButton("Refresh results")
        refresh.clicked.connect(self.refresh)
        self.select_button = QPushButton("Select production model")
        self.select_button.setObjectName("resultsPrimary")
        self.select_button.clicked.connect(self._select_model)
        actions.addWidget(refresh)
        actions.addWidget(self.select_button)
        actions.addStretch(1)
        root.addLayout(actions)

        runs = QGroupBox("Candidate-visible validation comparison")
        runs_layout = QVBoxLayout(runs)
        self.runs_table = QTableWidget(0, 9)
        self.runs_table.setHorizontalHeaderLabels(["Run", "Status", "mAP50", "mAP50-95", "Precision", "Recall", "Latency ms", "Checkpoints", "Selected"])
        self.runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.runs_table.itemSelectionChanged.connect(self._show_details)
        runs_layout.addWidget(self.runs_table)
        root.addWidget(runs, 1)

        details = QHBoxLayout()
        metric_box = QGroupBox("Selected run details")
        metric_layout = QVBoxLayout(metric_box)
        self.detail_label = QLabel("Select a completed run.")
        self.detail_label.setWordWrap(True)
        self.per_class = QTableWidget(0, 5)
        self.per_class.setHorizontalHeaderLabels(["Class", "mAP50", "mAP50-95", "Precision", "Recall"])
        self.per_class.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        metric_layout.addWidget(self.detail_label)
        metric_layout.addWidget(self.per_class)
        artifact_box = QGroupBox("Artifacts")
        artifact_layout = QVBoxLayout(artifact_box)
        self.artifact_view = QPlainTextEdit()
        self.artifact_view.setReadOnly(True)
        self.confusion_preview = QLabel("Confusion matrix: unavailable")
        self.confusion_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confusion_preview.setMinimumHeight(130)
        artifact_layout.addWidget(self.artifact_view)
        artifact_layout.addWidget(self.confusion_preview)
        details.addWidget(metric_box, 1)
        details.addWidget(artifact_box, 1)
        root.addLayout(details, 1)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            #resultsPage { background: #f5f8f7; color: #17211d; }
            QLabel#resultsEyebrow { color: #17745d; font-size: 12px; font-weight: 800; }
            QLabel#resultsTitle { color: #153c2d; font-size: 28px; font-weight: 800; }
            QLabel#resultsSubtitle { color: #566b62; font-size: 13px; }
            QLabel#resultsStatus { background: #e7f4ed; color: #18583e; padding: 8px; border-radius: 6px; }
            QGroupBox { background: #ffffff; border: 1px solid #d6e2dc; border-radius: 7px; margin-top: 8px; padding: 10px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background: #ffffff; color: #17304d; border: 1px solid #8fa5b5; border-radius: 5px; padding: 7px 13px; min-height: 32px; font-weight: 600; }
            QPushButton:hover { background: #e8f3f2; border-color: #087f75; }
            QPushButton#resultsPrimary { background: #17745d; color: white; font-weight: 800; }
            QTableWidget, QPlainTextEdit { background: white; border: 1px solid #d5e2dc; }
            """
        )

    def refresh(self) -> None:
        try:
            self._records = self.service.runs()
            selected = self.service.selected_model()
        except TrainingError as error:
            self.status.setText(str(error))
            return
        selected_run = selected.get("run_id") if selected else None
        self.runs_table.setRowCount(0)
        for record in self._records:
            row = self.runs_table.rowCount()
            self.runs_table.insertRow(row)
            metrics = record.get("metrics", {}) or {}
            values = [
                str(record.get("run_id", "")),
                str(record.get("status", "")),
                self._metric(metrics.get("map50")),
                self._metric(metrics.get("map50_95")),
                self._metric(metrics.get("precision")),
                self._metric(metrics.get("recall")),
                self._metric(metrics.get("latency_ms")),
                str(len(record.get("checkpoints", []) or [])),
                "yes" if selected_run == record.get("run_id") else "",
            ]
            for column, value in enumerate(values):
                self.runs_table.setItem(row, column, QTableWidgetItem(value))
        if selected_run:
            self.status.setText(f"Production model selected from {selected_run}. Candidate-visible only; no organizer lock or signature.")
        elif self._records:
            self.status.setText("Validation results loaded. Select a completed checkpoint to make it available to Vision.")
        else:
            self.status.setText("No recorded training runs yet.")

    @staticmethod
    def _metric(value: Any) -> str:
        return "unavailable" if value is None else f"{float(value):.4f}"

    def _selected_record(self) -> dict[str, Any] | None:
        selected = self.runs_table.selectedItems()
        if not selected:
            return None
        run_id = selected[0].text()
        return next((record for record in self._records if record.get("run_id") == run_id), None)

    def _show_details(self) -> None:
        record = self._selected_record()
        self.per_class.setRowCount(0)
        if record is None:
            return
        metrics = record.get("metrics", {}) or {}
        self.detail_label.setText(
            f"Run {record.get('run_id')} | status={record.get('status')} | "
            f"loss rows={len(metrics.get('loss_curve', []) or [])} | "
            f"confusion matrix={metrics.get('confusion_matrix') or 'unavailable'}"
        )
        for item in metrics.get("per_class", []) or []:
            row = self.per_class.rowCount()
            self.per_class.insertRow(row)
            values = [str(item.get("class_name", item.get("class_id", ""))), self._metric(item.get("map50")), self._metric(item.get("map50_95")), self._metric(item.get("precision")), self._metric(item.get("recall"))]
            for column, value in enumerate(values):
                self.per_class.setItem(row, column, QTableWidgetItem(value))
        curves = metrics.get("loss_curve", []) or []
        self.artifact_view.setPlainText(
            "\n".join(json_line for json_line in [
                f"model_dir: {record.get('paths', {}).get('model_dir', 'unavailable')}",
                f"log: {record.get('paths', {}).get('log', 'unavailable')}",
                f"checkpoints: {record.get('checkpoints', [])}",
                f"loss_curve_rows: {len(curves)}",
                f"loss_curve_tail: {json.dumps(curves[-5:], sort_keys=True) if curves else 'unavailable'}",
            ])
        )
        confusion = metrics.get("confusion_matrix")
        confusion_path = Path(str(confusion)) if confusion else None
        if confusion_path and confusion_path.is_file():
            pixmap = QPixmap(str(confusion_path))
            self.confusion_preview.setPixmap(pixmap.scaled(360, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.confusion_preview.setToolTip(str(confusion_path))
        else:
            self.confusion_preview.setPixmap(QPixmap())
            self.confusion_preview.setText("Confusion matrix: unavailable")

    def _select_model(self) -> None:
        record = self._selected_record()
        if record is None:
            self.status.setText("Select a recorded run first.")
            return
        if record.get("status") != "completed":
            self.status.setText("Only a completed run can provide the production model.")
            return
        try:
            selected_path = self.service.select_model(str(record["run_id"]))
            self.status.setText(f"Selected {selected_path.name} from {record['run_id']}; Vision can load this local .pt path.")
            self.refresh()
        except TrainingError as error:
            self.status.setText(f"Model selection failed: {error}")

    def shutdown(self) -> None:
        pass

