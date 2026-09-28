"""M3 Dataset Studio page and Qt workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
)

from ..core.workspace import CandidateWorkspace
from ..services.dataset import (
    CancellationToken,
    DatasetCancelled,
    DatasetError,
    DatasetScan,
    DatasetService,
    ImageRecord,
)
from ..services.print_exports import PrintExportOptions, PrintExportResult, export_print_assets


class DatasetTaskWorker(QObject):
    """Worker hosted on a QThread for scan and export tasks."""

    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, task: Callable[[CancellationToken, Callable[[int, int, str], None]], object], token: CancellationToken) -> None:
        super().__init__()
        self.task = task
        self.token = token

    @Slot()
    def run(self) -> None:
        try:
            result = self.task(self.token, self.progress.emit)
            self.completed.emit(result)
        except DatasetCancelled as error:
            self.failed.emit(str(error))
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            self.finished.emit()


class DatasetPreview(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 300)
        self.image = QImage()
        self.record: ImageRecord | None = None
        self.setObjectName("datasetPreview")

    def set_record(self, record: ImageRecord | None) -> None:
        self.record = record
        self.image = QImage(str(record.image_path)) if record is not None else QImage()
        self.update()

    def _image_rect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        ratio = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * ratio, self.image.height() * ratio
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#eef3f7"))
        if self.image.isNull():
            painter.setPen(QColor("#526b83"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select an image to preview annotations")
            return
        rectangle = self._image_rect()
        painter.drawImage(rectangle, self.image)
        if self.record is None:
            return
        scale_x = rectangle.width() / self.image.width()
        scale_y = rectangle.height() / self.image.height()
        for annotation in self.record.annotations:
            points = [
                QPointF(rectangle.x() + x * scale_x, rectangle.y() + y * scale_y)
                for x, y in annotation.polygon
            ]
            color = QColor("#f39c36") if annotation.source_kind == "box" else QColor("#0a8f83")
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(QPolygonF(points))


class DatasetPage(QWidget):
    """Embedded Dataset Studio with read-only scanning and atomic export."""

    def __init__(self, workspace: CandidateWorkspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.setObjectName("datasetPage")
        self.scan: DatasetScan | None = None
        self.selected_order: list[int] = []
        self._active_thread: QThread | None = None
        self._active_worker: DatasetTaskWorker | None = None
        self._cancel_token: CancellationToken | None = None
        self._task_kind: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(10)
        eyebrow = QLabel("DATASET")
        eyebrow.setObjectName("datasetEyebrow")
        title = QLabel("Dataset Studio")
        title.setObjectName("datasetTitle")
        subtitle = QLabel(
            "Inspect a local YOLO export, select four date classes in order, and create a workspace-owned subset without changing the source."
        )
        subtitle.setObjectName("datasetSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(eyebrow)
        root.addWidget(title)
        root.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dataset_tab(), "Dataset")
        self.tabs.addTab(self._build_print_tab(), "Print exports")
        root.addWidget(self.tabs, 1)
        self.setStyleSheet(
            """
            #datasetPage { background: #f3f7fb; color: #172b4d; }
            QLabel#datasetEyebrow { color: #078b80; font-size: 12px; font-weight: 800; }
            QLabel#datasetTitle { color: #17304d; font-size: 28px; font-weight: 800; }
            QLabel#datasetSubtitle { color: #61748c; font-size: 13px; }
            QFrame#datasetCard { background: #ffffff; border: 1px solid #dce6ef; border-radius: 10px; }
            QLabel#datasetSection { color: #17304d; font-size: 17px; font-weight: 750; }
            QLabel#datasetStatus { background: #e9f8f5; color: #14675f; border-radius: 8px; padding: 9px; }
            QLabel#datasetWarning { background: #fff6e5; color: #795214; border-radius: 8px; padding: 9px; }
            QPushButton { background: #ffffff; color: #17304d; border: 1px solid #8fa5b5; border-radius: 5px; padding: 7px 13px; min-height: 32px; font-weight: 600; }
            QPushButton:hover { background: #e8f3f2; border-color: #087f75; }
            QPushButton#datasetPrimary { background: #087f75; border-color: #087f75; color: white; font-weight: 800; }
            QLineEdit, QSpinBox { background: white; color: #17304d; border: 1px solid #cbd9e6; border-radius: 7px; padding: 5px 8px; }
            QTableWidget, QListWidget, QPlainTextEdit { background: white; color: #17304d; border: 1px solid #d4e2ed; border-radius: 7px; }
            QProgressBar { border: 1px solid #d4e2ed; border-radius: 6px; text-align: center; background: #f2f6fb; }
            QProgressBar::chunk { background: #0b9d91; border-radius: 5px; }
            QTabWidget::pane { border: 0; }
            """
        )

    def _build_dataset_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        source_card = QFrame()
        source_card.setObjectName("datasetCard")
        source_layout = QVBoxLayout(source_card)
        source_layout.setContentsMargins(14, 12, 14, 12)
        source_heading = QLabel("Source dataset")
        source_heading.setObjectName("datasetSection")
        source_layout.addWidget(source_heading)
        source_row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Folder containing data.yaml and train/valid/test images and labels")
        source_row.addWidget(self.source_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_source)
        source_row.addWidget(browse)
        self.scan_button = QPushButton("Scan dataset")
        self.scan_button.setObjectName("datasetPrimary")
        self.scan_button.clicked.connect(self._scan_dataset)
        source_row.addWidget(self.scan_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_task)
        source_row.addWidget(self.cancel_button)
        source_layout.addLayout(source_row)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        source_layout.addWidget(self.progress)
        self.summary = QLabel("No dataset scanned.")
        self.summary.setObjectName("datasetStatus")
        self.summary.setWordWrap(True)
        source_layout.addWidget(self.summary)
        layout.addWidget(source_card)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.setChildrenCollapsible(False)
        main_split.addWidget(self._build_class_panel())
        main_split.addWidget(self._build_browser_panel())
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 2)
        layout.addWidget(main_split, 1)

        diagnostics_card = QFrame()
        diagnostics_card.setObjectName("datasetCard")
        diagnostics_layout = QVBoxLayout(diagnostics_card)
        diagnostics_layout.setContentsMargins(12, 10, 12, 10)
        diagnostics_layout.addWidget(QLabel("Diagnostics"))
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setMaximumHeight(110)
        diagnostics_layout.addWidget(self.diagnostics)
        layout.addWidget(diagnostics_card)
        return page

    def _build_class_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("datasetCard")
        panel.setMinimumWidth(650)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.addWidget(QLabel("Classes and split counts"))
        self.class_table = QTableWidget(0, 5)
        self.class_table.setHorizontalHeaderLabels(("Select", "ID / Class", "Train img / obj", "Val img / obj", "Test img / obj"))
        self.class_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.class_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        for column, width in enumerate((62, 160, 125, 125, 125)):
            header.resizeSection(column, width)
        layout.addWidget(self.class_table, 1)

        layout.addWidget(QLabel("Selected order (new IDs 0-3)"))
        self.order_list = QListWidget()
        self.order_list.setMaximumHeight(115)
        layout.addWidget(self.order_list)
        order_buttons = QHBoxLayout()
        self.up_button = QPushButton("Move up")
        self.up_button.clicked.connect(lambda: self._move_selected(-1))
        self.down_button = QPushButton("Move down")
        self.down_button.clicked.connect(lambda: self._move_selected(1))
        order_buttons.addWidget(self.up_button)
        order_buttons.addWidget(self.down_button)
        layout.addLayout(order_buttons)
        self.selection_note = QLabel("Select exactly four classes. This is a local selection, not an organizer lock or signature.")
        self.selection_note.setObjectName("datasetWarning")
        self.selection_note.setWordWrap(True)
        layout.addWidget(self.selection_note)
        self.export_button = QPushButton("Export working subset")
        self.export_button.setObjectName("datasetPrimary")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_subset)
        layout.addWidget(self.export_button)
        return panel

    def _build_browser_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("datasetCard")
        panel.setMinimumWidth(380)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.addWidget(QLabel("Image browser and annotations"))
        browser_split = QSplitter(Qt.Orientation.Vertical)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._preview_row)
        browser_split.addWidget(self.image_list)
        self.preview = DatasetPreview()
        browser_split.addWidget(self.preview)
        browser_split.setStretchFactor(0, 1)
        browser_split.setStretchFactor(1, 3)
        layout.addWidget(browser_split, 1)
        self.annotation_note = QLabel("Orange outlines are bounding boxes; teal outlines are segmentation polygons.")
        self.annotation_note.setWordWrap(True)
        layout.addWidget(self.annotation_note)
        return panel

    def _build_print_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        card = QFrame()
        card.setObjectName("datasetCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        heading = QLabel("Print exports")
        heading.setObjectName("datasetSection")
        card_layout.addWidget(heading)
        description = QLabel(
            "Uses the preserved printable-image utility. PNG keeps transparency, JPEG uses a white background, and PDF creates A4 cards sized 6 x 4 cm with cut marks, shape cut lines, class names, and fit rotation. Box-only labels receive rectangular cut lines."
        )
        description.setWordWrap(True)
        card_layout.addWidget(description)
        formats = QHBoxLayout()
        self.print_png = QCheckBox("Transparent PNG")
        self.print_png.setChecked(True)
        self.print_jpeg = QCheckBox("White JPEG")
        self.print_jpeg.setChecked(True)
        self.print_pdf = QCheckBox("A4 6 x 4 cm PDF")
        self.print_pdf.setChecked(True)
        formats.addWidget(self.print_png)
        formats.addWidget(self.print_jpeg)
        formats.addWidget(self.print_pdf)
        formats.addStretch(1)
        card_layout.addLayout(formats)
        options = QGridLayout()
        options.addWidget(QLabel("Padding (%)"), 0, 0)
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 50)
        self.padding_spin.setValue(2)
        options.addWidget(self.padding_spin, 0, 1)
        options.addWidget(QLabel("JPEG quality"), 0, 2)
        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(50, 100)
        self.jpeg_quality_spin.setValue(95)
        options.addWidget(self.jpeg_quality_spin, 0, 3)
        card_layout.addLayout(options)
        self.print_button = QPushButton("Generate print exports")
        self.print_button.setObjectName("datasetPrimary")
        self.print_button.clicked.connect(self._export_prints)
        card_layout.addWidget(self.print_button)
        self.print_status = QLabel("Scan a dataset and select four classes first.")
        self.print_status.setObjectName("datasetStatus")
        self.print_status.setWordWrap(True)
        card_layout.addWidget(self.print_status)
        card_layout.addStretch(1)
        layout.addWidget(card, 1)
        return page

    def _browse_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose YOLO dataset root")
        if path:
            self.source_edit.setText(path)

    def _start_task(self, kind: str, task: Callable[[CancellationToken, Callable[[int, int, str], None]], object], completed: Callable[[object], None]) -> None:
        if self._active_thread is not None:
            return
        self._task_kind = kind
        self._cancel_token = CancellationToken()
        self._active_thread = QThread(self)
        self._active_worker = DatasetTaskWorker(task, self._cancel_token)
        self._active_worker.moveToThread(self._active_thread)
        self._active_thread.started.connect(self._active_worker.run)
        self._active_worker.progress.connect(self._task_progress)
        self._active_worker.completed.connect(completed)
        self._active_worker.failed.connect(self._task_failed)
        self._active_worker.finished.connect(self._active_thread.quit)
        self._active_worker.finished.connect(self._active_worker.deleteLater)
        self._active_thread.finished.connect(self._task_finished)
        self._active_thread.finished.connect(self._active_thread.deleteLater)
        self._active_thread.start()
        self.cancel_button.setEnabled(True)
        self.scan_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.print_button.setEnabled(False)

    def _task_progress(self, current: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(min(current, max(1, total)))
        if self._task_kind == "print":
            self.print_status.setText(message)
        else:
            self.summary.setText(message)

    def _task_failed(self, message: str) -> None:
        if self._task_kind == "print":
            self.print_status.setText(f"Export failed: {message}")
        else:
            self.summary.setText(f"Dataset task failed: {message}")

    def _task_finished(self) -> None:
        self._active_thread = None
        self._active_worker = None
        self._cancel_token = None
        self._task_kind = None
        self.cancel_button.setEnabled(False)
        self.scan_button.setEnabled(True)
        self._refresh_action_state()

    def _cancel_task(self) -> None:
        if self._cancel_token is not None:
            self._cancel_token.cancel()
            self.summary.setText("Cancellation requested; finishing the current file...")

    def _scan_dataset(self) -> None:
        root = Path(self.source_edit.text().strip()).expanduser()
        if not root:
            self.summary.setText("Choose a dataset root first.")
            return
        self._start_task(
            "scan",
            lambda token, progress: DatasetService.scan(root, cancellation=token, progress=progress),
            self._scan_completed,
        )

    def _scan_completed(self, result: object) -> None:
        self.scan = result  # type: ignore[assignment]
        self.selected_order = []
        self._populate_scan()
        scan = self.scan
        if scan is None:
            return
        summary = " | ".join(
            f"{key}: {split.image_count} images / {split.object_count} objects"
            for key, split in scan.splits.items()
        )
        self.summary.setText(
            f"{scan.root} | {len(scan.names)} classes | {summary} | boxes={scan.bbox_count}, polygons={scan.segment_count}"
        )
        self.diagnostics.setPlainText(
            "\n".join(diagnostic.display() for diagnostic in scan.diagnostics)
            or "No diagnostics."
        )
        self.print_status.setText(
            "Scan complete. Select four classes before generating print exports. "
            + ("Box-only labels will produce rectangular cut lines." if scan.bbox_count else "")
        )

    def _populate_scan(self) -> None:
        if self.scan is None:
            return
        stats = self.scan.class_stats()
        self.class_table.setRowCount(0)
        for row, class_id in enumerate(self.scan.class_ids):
            self.class_table.insertRow(row)
            name = QTableWidgetItem(f"{class_id}  {self.scan.names[class_id]}")
            self.class_table.setItem(row, 1, name)
            for column, split_key in enumerate(("train", "val", "test"), start=1):
                values = stats.get(class_id, {}).get(split_key, {"images": 0, "objects": 0})
                self.class_table.setItem(row, column + 1, QTableWidgetItem(f"{values['images']} / {values['objects']}"))
            checkbox = QCheckBox()
            checkbox.setStyleSheet("margin-left: 20px;")
            checkbox.stateChanged.connect(
                lambda state, selected_id=class_id, box=checkbox: self._class_toggled(
                    selected_id, state, box
                )
            )
            self.class_table.setCellWidget(row, 0, checkbox)
        self.image_list.clear()
        for index, record in enumerate(self.scan.records):
            label = f"{record.split}: {record.image_path.name} ({len(record.annotations)} objects)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.image_list.addItem(item)
        if self.image_list.count():
            self.image_list.setCurrentRow(0)

    def _class_toggled(self, class_id: int, state: int, checkbox: QCheckBox | None = None) -> None:
        checked = state == Qt.CheckState.Checked.value
        if checked and class_id not in self.selected_order:
            if len(self.selected_order) >= 4:
                if checkbox is not None:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)
                self.selection_note.setText("Exactly four classes are allowed. Move or unselect an existing class first.")
                return
            self.selected_order.append(class_id)
        elif not checked and class_id in self.selected_order:
            self.selected_order.remove(class_id)
        self._refresh_order()
        self._refresh_action_state()

    def _refresh_order(self) -> None:
        if self.scan is None:
            return
        self.order_list.clear()
        for new_id, original_id in enumerate(self.selected_order):
            self.order_list.addItem(f"{new_id}  <- original {original_id}: {self.scan.names[original_id]}")
        self.selection_note.setText(
            "Selected classes are local only; no organizer lock or signature is claimed."
            if len(self.selected_order) == 4
            else f"Select exactly four classes ({len(self.selected_order)}/4 selected)."
        )

    def _move_selected(self, direction: int) -> None:
        row = self.order_list.currentRow()
        target = row + direction
        if row < 0 or target < 0 or target >= len(self.selected_order):
            return
        self.selected_order[row], self.selected_order[target] = self.selected_order[target], self.selected_order[row]
        self._refresh_order()
        self.order_list.setCurrentRow(target)

    def _refresh_action_state(self) -> None:
        ready = self.scan is not None and len(self.selected_order) == 4 and self._active_thread is None
        self.export_button.setEnabled(ready)
        self.print_button.setEnabled(ready)

    def _preview_row(self, row: int) -> None:
        if self.scan is None or row < 0 or row >= len(self.scan.records):
            self.preview.set_record(None)
            return
        self.preview.set_record(self.scan.records[row])

    def _export_subset(self) -> None:
        if self.scan is None or len(self.selected_order) != 4:
            self.summary.setText("Select exactly four classes before exporting.")
            return
        selected = list(self.selected_order)
        scan = self.scan
        self.workspace.record_event(
            "dataset.selection.changed",
            {"source_root": str(scan.root), "selected_class_ids": selected, "signed": False, "locked": False},
        )
        self._start_task(
            "subset",
            lambda token, progress: DatasetService.export_subset(
                scan, self.workspace.root, selected, cancellation=token, progress=progress
            ),
            self._subset_completed,
        )

    def _subset_completed(self, result: object) -> None:
        export = result
        self.workspace.record_event(
            "dataset.exported",
            {
                "manifest": str(export.manifest_path.relative_to(self.workspace.root)),
                "output_dir": str(export.output_dir.relative_to(self.workspace.root)),
                "selected_class_ids": list(self.selected_order),
                "counts": export.manifest["splits"],
                "exclusions": export.manifest["exclusions"],
            },
        )
        self.summary.setText(
            f"Complete subset exported to {export.output_dir}. "
            f"Mixed-class exclusions: {export.manifest['mixed_class_excluded_total']}."
        )

    def _export_prints(self) -> None:
        if self.scan is None or len(self.selected_order) != 4:
            self.print_status.setText("Select exactly four classes before generating print exports.")
            return
        options = PrintExportOptions(
            export_png=self.print_png.isChecked(),
            export_jpeg=self.print_jpeg.isChecked(),
            export_pdf=self.print_pdf.isChecked(),
            jpeg_quality=self.jpeg_quality_spin.value(),
            padding_percent=float(self.padding_spin.value()),
        )
        scan = self.scan
        selected = list(self.selected_order)
        self._start_task(
            "print",
            lambda token, progress: export_print_assets(
                scan, self.workspace.root, selected, options, cancellation=token, progress=progress
            ),
            self._prints_completed,
        )

    def _prints_completed(self, result: object) -> None:
        export: PrintExportResult = result  # type: ignore[assignment]
        self.workspace.record_event(
            "dataset.print_exported",
            {
                "output_dir": str(export.output_dir.relative_to(self.workspace.root)),
                "selected_class_ids": list(self.selected_order),
                "box_only_objects": export.box_only_objects,
            },
        )
        self.print_status.setText(
            f"Print exports complete: {export.output_dir}. "
            f"Box-only rectangular cutouts: {export.box_only_objects}."
        )

    def shutdown(self) -> None:
        self._cancel_task()
        if self._active_thread is not None:
            self._active_thread.quit()
            self._active_thread.wait(1500)
