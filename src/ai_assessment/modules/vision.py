"""M4 Vision Studio page with bounded camera and inference workers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal, Slot, QThread
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.workspace import CandidateWorkspace
from ..services.calibration import CalibrationError, CalibrationGeometryMismatchError, CalibrationRecord, CalibrationService
from ..services.camera import CameraOwnership
from ..services.vision import (
    Detection,
    FrameResult,
    LatestFrameBuffer,
    ModelLoadError,
    VisionError,
    VisionProfile,
    VisionWorker,
    load_profile,
    normalise_roi,
    robot_coordinates_for_detection,
    save_profile,
    save_snapshot,
    validate_model_path,
)


def _qimage_from_bgr(frame: Any) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = QImage(
        rgb.data,
        int(rgb.shape[1]),
        int(rgb.shape[0]),
        int(rgb.strides[0]),
        QImage.Format.Format_RGB888,
    )
    return image.copy()


class VisionImageView(QWidget):
    roi_selected = Signal(int, int, int, int)

    def __init__(self, *, interactive_roi: bool = False) -> None:
        super().__init__()
        self.setMinimumSize(360, 260)
        self.setMouseTracking(True)
        self.setCursor(
            Qt.CursorShape.CrossCursor if interactive_roi else Qt.CursorShape.ArrowCursor
        )
        self.interactive_roi = interactive_roi
        self.image = QImage()
        self.roi: tuple[int, int, int, int] | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_current: tuple[float, float] | None = None

    def set_image(self, image: QImage) -> None:
        self.image = image
        self.update()

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self.roi = roi
        self.update()

    def image_rect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        ratio = min(self.width() / self.image.width(), self.height() / self.image.height())
        width = self.image.width() * ratio
        height = self.image.height() * ratio
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _image_point(self, x: float, y: float) -> tuple[float, float] | None:
        rectangle = self.image_rect()
        if rectangle.isNull() or not rectangle.contains(x, y):
            return None
        return (
            max(0.0, min(self.image.width() - 1.0, (x - rectangle.x()) * self.image.width() / rectangle.width())),
            max(0.0, min(self.image.height() - 1.0, (y - rectangle.y()) * self.image.height() / rectangle.height())),
        )

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#eaf1f7"))
        if self.image.isNull():
            painter.setPen(QColor("#526b83"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Camera preview unavailable")
            return
        rectangle = self.image_rect()
        painter.drawImage(rectangle, self.image)
        if self.roi is not None:
            x, y, width, height = self.roi
            roi_rect = QRectF(
                rectangle.x() + x * rectangle.width() / self.image.width(),
                rectangle.y() + y * rectangle.height() / self.image.height(),
                width * rectangle.width() / self.image.width(),
                height * rectangle.height() / self.image.height(),
            )
            painter.setPen(QPen(QColor("#f39c36"), 2))
            painter.drawRect(roi_rect)
        if self._drag_start is not None and self._drag_current is not None:
            start = self._widget_from_image(self._drag_start)
            current = self._widget_from_image(self._drag_current)
            painter.setPen(QPen(QColor("#087f75"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(start, current).normalized())

    def _widget_from_image(self, point: tuple[float, float]) -> QPointF:
        rectangle = self.image_rect()
        return QPointF(
            rectangle.x() + point[0] * rectangle.width() / self.image.width(),
            rectangle.y() + point[1] * rectangle.height() / self.image.height(),
        )

    def mousePressEvent(self, event: Any) -> None:
        if not self.interactive_roi or self.image.isNull():
            return
        point = self._image_point(event.position().x(), event.position().y())
        if point is not None:
            self._drag_start = point
            self._drag_current = point
            self.update()

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag_start is not None:
            point = self._image_point(event.position().x(), event.position().y())
            if point is not None:
                self._drag_current = point
                self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._drag_start is None:
            return
        point = self._image_point(event.position().x(), event.position().y())
        if point is not None:
            self._drag_current = point
        if self._drag_current is not None:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_current
            x, y = int(round(min(x1, x2))), int(round(min(y1, y2)))
            width, height = int(round(abs(x2 - x1))), int(round(abs(y2 - y1)))
            if width >= 4 and height >= 4:
                self.roi_selected.emit(x, y, width, height)
        self._drag_start = None
        self._drag_current = None
        self.update()


class VisionPage(QWidget):
    """Embedded raw/processed camera and local-model vision workspace."""

    def __init__(self, workspace: CandidateWorkspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.setObjectName("visionPage")
        self.buffer = LatestFrameBuffer()
        self._thread: QThread | None = None
        self._worker: VisionWorker | None = None
        self._last_result: FrameResult | None = None
        self._camera_metadata: dict[str, Any] = {}
        try:
            self._profile = load_profile(workspace.root)
            self._profile_error: str | None = None
        except VisionError as error:
            self._profile = VisionProfile()
            self._profile_error = str(error)
        self._calibration: CalibrationRecord | None = None
        self._build_ui()
        self._apply_profile()
        self._load_calibration()
        if self._profile_error:
            self.status.setText(f"Camera profile unavailable: {self._profile_error}")
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._poll_frame)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(9)
        eyebrow = QLabel("VISION / M4")
        eyebrow.setObjectName("visionEyebrow")
        title = QLabel("Vision Studio")
        title.setObjectName("visionTitle")
        subtitle = QLabel(
            "View the full camera frame, draw an inference ROI, apply transparent image corrections, and inspect local-model detections without moving the robot."
        )
        subtitle.setObjectName("visionSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(eyebrow)
        root.addWidget(title)
        root.addWidget(subtitle)

        controls = QFrame()
        controls.setObjectName("visionCard")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(14, 12, 14, 12)
        controls_layout.setSpacing(10)

        source_row = QHBoxLayout()
        source_row.setSpacing(10)

        source_group = QGroupBox("Input source")
        source_group.setObjectName("visionControlGroup")
        source_layout = QGridLayout(source_group)
        source_layout.setContentsMargins(11, 12, 11, 10)
        source_layout.setHorizontalSpacing(9)
        source_layout.setVerticalSpacing(7)
        source_layout.addWidget(QLabel("Camera index"), 0, 0)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 20)
        source_layout.addWidget(self.camera_index, 0, 1)
        source_layout.setColumnStretch(1, 1)
        source_row.addWidget(source_group)

        model_group = QGroupBox("Local model")
        model_group.setObjectName("visionControlGroup")
        model_layout = QGridLayout(model_group)
        model_layout.setContentsMargins(11, 12, 11, 10)
        model_layout.setHorizontalSpacing(8)
        model_layout.setVerticalSpacing(7)
        model_layout.addWidget(QLabel("Weights file"), 0, 0)
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Optional .pt or .onnx file; preview works without a model")
        self.model_path.editingFinished.connect(self._model_path_changed)
        model_layout.addWidget(self.model_path, 0, 1, 1, 2)
        choose_model = QPushButton("Browse model...")
        choose_model.setObjectName("visionSecondary")
        choose_model.clicked.connect(self._browse_model)
        model_layout.addWidget(choose_model, 1, 1)
        self.load_model_button = QPushButton("Load model")
        self.load_model_button.setObjectName("visionAction")
        self.load_model_button.clicked.connect(self._load_model)
        model_layout.addWidget(self.load_model_button, 1, 2)
        model_layout.setColumnStretch(1, 1)
        source_row.addWidget(model_group, 1)

        capture_group = QGroupBox("Capture")
        capture_group.setObjectName("visionControlGroup")
        capture_layout = QHBoxLayout(capture_group)
        capture_layout.setContentsMargins(11, 12, 11, 10)
        capture_layout.setSpacing(8)
        self.start_button = QPushButton("Start camera")
        self.start_button.setObjectName("visionPrimary")
        self.start_button.clicked.connect(self._start_camera)
        capture_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("visionStop")
        self.stop_button.clicked.connect(self._stop_camera)
        self.stop_button.setEnabled(False)
        capture_layout.addWidget(self.stop_button)
        source_row.addWidget(capture_group)
        controls_layout.addLayout(source_row)

        tuning_row = QHBoxLayout()
        tuning_row.setSpacing(10)

        inference_group = QGroupBox("Inference adjustments")
        inference_group.setObjectName("visionControlGroup")
        inference_layout = QGridLayout(inference_group)
        inference_layout.setContentsMargins(11, 12, 11, 10)
        inference_layout.setHorizontalSpacing(8)
        inference_layout.setVerticalSpacing(7)
        self.confidence = self._double_control(0.05, 0.99, 0.5, 2)
        self.brightness = QSpinBox()
        self.brightness.setRange(-100, 100)
        self.contrast = self._double_control(0.2, 3.0, 1.0, 2)
        for column, (label, widget) in enumerate(
            (("Confidence", self.confidence), ("Brightness", self.brightness), ("Contrast", self.contrast))
        ):
            inference_layout.addWidget(QLabel(label), 0, column * 2)
            inference_layout.addWidget(widget, 0, column * 2 + 1)
        for column in range(3):
            inference_layout.setColumnStretch(column * 2 + 1, 1)
        tuning_row.addWidget(inference_group, 1)

        camera_group = QGroupBox("Camera properties")
        camera_group.setObjectName("visionControlGroup")
        camera_layout = QGridLayout(camera_group)
        camera_layout.setContentsMargins(11, 12, 11, 10)
        camera_layout.setHorizontalSpacing(8)
        camera_layout.setVerticalSpacing(7)
        self.auto_exposure = QCheckBox("Auto exposure")
        self.auto_exposure.setChecked(True)
        self.exposure = self._double_control(-20, 20, -6, 2)
        self.auto_focus = QCheckBox("Auto focus")
        self.auto_focus.setChecked(True)
        self.focus = self._double_control(0, 1000, 0, 1)
        self.auto_white_balance = QCheckBox("Auto white balance")
        self.auto_white_balance.setChecked(True)
        self.white_balance = self._double_control(2000, 10000, 4500, 0)
        for row, (auto_control, label, value_control) in enumerate(
            (
                (self.auto_exposure, "Exposure", self.exposure),
                (self.auto_focus, "Focus", self.focus),
                (self.auto_white_balance, "White balance", self.white_balance),
            )
        ):
            camera_layout.addWidget(auto_control, row, 0)
            camera_layout.addWidget(QLabel(label), row, 1)
            camera_layout.addWidget(value_control, row, 2)
        camera_layout.setColumnStretch(2, 1)
        tuning_row.addWidget(camera_group, 1)
        controls_layout.addLayout(tuning_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.save_profile_button = QPushButton("Save camera profile")
        self.save_profile_button.setObjectName("visionSecondary")
        self.save_profile_button.clicked.connect(self._save_profile)
        action_row.addWidget(self.save_profile_button)
        self.snapshot_button = QPushButton("Save snapshot")
        self.snapshot_button.setObjectName("visionAction")
        self.snapshot_button.clicked.connect(self._save_snapshot)
        self.snapshot_button.setEnabled(False)
        action_row.addWidget(self.snapshot_button)
        clear_roi = QPushButton("Clear ROI")
        clear_roi.setObjectName("visionSecondary")
        clear_roi.clicked.connect(self._clear_roi)
        action_row.addWidget(clear_roi)
        self.reload_calibration_button = QPushButton("Reload calibration")
        self.reload_calibration_button.setObjectName("visionSecondary")
        self.reload_calibration_button.clicked.connect(self._load_calibration)
        action_row.addWidget(self.reload_calibration_button)
        action_row.addStretch(1)
        self.device_label = QLabel("Device: preview only")
        self.device_label.setObjectName("visionDevice")
        action_row.addWidget(self.device_label)
        self.camera_readback = QLabel("Camera requested/read-back values appear after start.")
        self.camera_readback.setWordWrap(True)
        self.camera_readback.setObjectName("visionReadback")
        action_row.addWidget(self.camera_readback, 1)
        controls_layout.addLayout(action_row)
        root.addWidget(controls)

        center = QSplitter(Qt.Orientation.Horizontal)
        center.setChildrenCollapsible(False)
        image_card = QFrame()
        image_card.setObjectName("visionCard")
        image_layout = QVBoxLayout(image_card)
        image_layout.setContentsMargins(12, 10, 12, 10)
        image_split = QSplitter(Qt.Orientation.Horizontal)
        raw_holder = QWidget()
        raw_layout = QVBoxLayout(raw_holder)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        raw_layout.addWidget(QLabel("RAW FULL FRAME"))
        self.raw_view = VisionImageView(interactive_roi=True)
        self.raw_view.roi_selected.connect(self._set_roi)
        raw_layout.addWidget(self.raw_view, 1)
        processed_holder = QWidget()
        processed_layout = QVBoxLayout(processed_holder)
        processed_layout.setContentsMargins(0, 0, 0, 0)
        processed_layout.addWidget(QLabel("PROCESSED ROI + YOLO"))
        self.processed_view = VisionImageView()
        processed_layout.addWidget(self.processed_view, 1)
        image_split.addWidget(raw_holder)
        image_split.addWidget(processed_holder)
        image_split.setStretchFactor(0, 1)
        image_split.setStretchFactor(1, 1)
        image_layout.addWidget(image_split, 1)
        self.roi_label = QLabel("ROI: full frame")
        self.roi_label.setObjectName("visionHint")
        image_layout.addWidget(self.roi_label)
        center.addWidget(image_card)

        detection_card = QFrame()
        detection_card.setObjectName("visionCard")
        detection_layout = QVBoxLayout(detection_card)
        detection_layout.setContentsMargins(12, 10, 12, 10)
        detection_layout.addWidget(QLabel("Detections and centers"))
        self.detection_table = QTableWidget(0, 7)
        self.detection_table.setHorizontalHeaderLabels(
            ("ID", "Class", "Confidence", "BBox full px", "Center full px", "Frame", "Robot X/Y")
        )
        self.detection_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.detection_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        detection_layout.addWidget(self.detection_table, 1)
        self.calibration_warning = QLabel("No matching saved calibration loaded.")
        self.calibration_warning.setObjectName("visionWarning")
        self.calibration_warning.setWordWrap(True)
        detection_layout.addWidget(self.calibration_warning)
        self.status = QLabel("Start the camera for a live preview. A model is optional.")
        self.status.setObjectName("visionStatus")
        self.status.setWordWrap(True)
        detection_layout.addWidget(self.status)
        center.addWidget(detection_card)
        center.setStretchFactor(0, 3)
        center.setStretchFactor(1, 2)
        root.addWidget(center, 1)

        self.setStyleSheet(
            """
            #visionPage { background: #f3f7fb; color: #172b4d; }
            QLabel#visionEyebrow { color: #078b80; font-size: 12px; font-weight: 800; }
            QLabel#visionTitle { color: #17304d; font-size: 28px; font-weight: 800; }
            QLabel#visionSubtitle { color: #61748c; font-size: 13px; }
            QFrame#visionCard { background: #ffffff; border: 1px solid #dce6ef; border-radius: 10px; }
            QLabel#visionStatus { background: #e9f8f5; color: #14675f; border-radius: 8px; padding: 9px; }
            QLabel#visionWarning { background: #fff0ed; color: #9f2d20; border-radius: 8px; padding: 9px; }
            QLabel#visionHint { background: #fff6e5; color: #795214; border-radius: 8px; padding: 7px; }
            QLabel#visionDevice { color: #14675f; font-weight: 800; }
             QLabel#visionReadback { color: #61748c; }
             QGroupBox#visionControlGroup { background: #f8fbfc; border: 1px solid #c7d8df; border-radius: 7px; margin-top: 8px; padding: 9px; }
             QGroupBox#visionControlGroup::title { color: #2a4c63; font-size: 12px; font-weight: 800; padding: 0 4px; }
            QPushButton { background: #ffffff; color: #17304d; border: 1px solid #8fa5b5; border-radius: 5px; padding: 7px 13px; min-height: 32px; font-weight: 600; }
            QPushButton:hover { background: #e8f3f2; border-color: #087f75; }
            QPushButton#visionPrimary { background: #087f75; border-color: #087f75; color: white; font-weight: 800; }
             QPushButton#visionPrimary:hover { background: #0a988d; border-color: #0a988d; }
             QPushButton#visionStop { background: #fff8f6; color: #a13b2f; border-color: #e0aaa2; }
             QPushButton#visionStop:hover { background: #ffe9e5; border-color: #b74b3e; }
             QPushButton#visionAction { background: #e8f3f2; color: #075f59; border-color: #76bdb7; font-weight: 700; }
             QPushButton#visionAction:hover { background: #d3e9e6; border-color: #087f75; }
            QLineEdit, QSpinBox, QDoubleSpinBox { background: white; color: #17304d; border: 1px solid #cbd9e6; border-radius: 7px; padding: 5px 8px; }
            QTableWidget { background: white; color: #17304d; border: 1px solid #d4e2ed; border-radius: 7px; }
            """
        )
        for widget in (self.confidence, self.brightness, self.contrast, self.exposure, self.focus, self.white_balance,
                       self.auto_exposure, self.auto_focus, self.auto_white_balance):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._settings_changed)
            else:
                widget.stateChanged.connect(self._settings_changed)

    @staticmethod
    def _double_control(minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        control.setDecimals(decimals)
        control.setSingleStep(0.1 if decimals else 1.0)
        return control

    def _apply_profile(self) -> None:
        profile = self._profile
        self.camera_index.setValue(profile.camera_index)
        self.confidence.setValue(profile.confidence)
        self.brightness.setValue(int(profile.brightness))
        self.contrast.setValue(profile.contrast)
        requested = profile.requested_camera
        self.auto_exposure.setChecked(float(requested.get("auto_exposure", 1.0)) != 0.0)
        self.exposure.setValue(float(requested.get("exposure", self.exposure.value())))
        self.auto_focus.setChecked(float(requested.get("autofocus", 1.0)) != 0.0)
        self.focus.setValue(float(requested.get("focus", self.focus.value())))
        self.auto_white_balance.setChecked(
            float(requested.get("auto_white_balance", 1.0)) != 0.0
        )
        self.white_balance.setValue(
            float(requested.get("white_balance", self.white_balance.value()))
        )
        if profile.model_path and Path(profile.model_path).is_file():
            self.model_path.setText(profile.model_path)
        else:
            selected_candidate: Path | None = None
            selection_path = self.workspace.resolve_inside("results", "selected_model.json")
            if selection_path.is_file():
                try:
                    selection = json.loads(selection_path.read_text(encoding="utf-8"))
                    if isinstance(selection, dict):
                        selected_candidate = self.workspace.resolve_inside(selection.get("path", ""))
                except (OSError, TypeError, ValueError):
                    selected_candidate = None
            candidates = ([selected_candidate] if selected_candidate else []) + [Path.cwd() / "yolo26n.pt", self.workspace.root / "models" / "yolo26n.pt"]
            for candidate in candidates:
                if candidate is None:
                    continue
                if candidate.is_file():
                    self.model_path.setText(str(candidate.resolve()))
                    break
        if profile.roi:
            self._set_roi_from_mapping(profile.roi, log_event=False)

    def _settings(self) -> dict[str, Any]:
        return {
            "roi": list(self.raw_view.roi) if self.raw_view.roi else None,
            "brightness": float(self.brightness.value()),
            "contrast": float(self.contrast.value()),
            "confidence": float(self.confidence.value()),
            "requested_camera": {
                "auto_exposure": 1.0 if self.auto_exposure.isChecked() else 0.0,
                "exposure": float(self.exposure.value()),
                "autofocus": 1.0 if self.auto_focus.isChecked() else 0.0,
                "focus": float(self.focus.value()),
                "auto_white_balance": 1.0 if self.auto_white_balance.isChecked() else 0.0,
                "white_balance": float(self.white_balance.value()),
            },
            "model_path": self.model_path.text().strip() or None,
        }

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose local YOLO model", "", "YOLO models (*.pt *.onnx)")
        if path:
            self.model_path.setText(path)
            self._model_path_changed()

    def _model_path_changed(self) -> None:
        if self._worker is not None:
            raw_path = self.model_path.text().strip()
            self._worker.request_model_load(Path(raw_path) if raw_path else None)
        self._save_profile()

    def _load_model(self) -> None:
        raw_path = self.model_path.text().strip()
        try:
            path = validate_model_path(Path(raw_path))
        except (ModelLoadError, VisionError) as error:
            self.status.setText(str(error))
            return
        if self._worker is None:
            self.status.setText("Model selected. Start the camera to load it off the UI thread; preview does not require a model.")
            return
        self._worker.request_model_load(path)
        self._save_profile()

    def _start_camera(self) -> None:
        if self._thread is not None:
            return
        owner = CameraOwnership.current_owner()
        if owner not in (None, "vision"):
            self.status.setText(f"Camera is already in use by {owner}. Stop it before starting Vision Studio.")
            return
        settings = self._settings()
        self.buffer.clear()
        self._worker = VisionWorker(self.camera_index.value(), settings, self.buffer)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.metadata_ready.connect(self._metadata_ready)
        self._worker.status.connect(self.status.setText)
        self._worker.model_status.connect(self._model_status)
        self._worker.error.connect(self._worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.workspace.record_event("vision.camera.started", {"camera_index": self.camera_index.value()})
        self._save_profile()

    def _stop_camera(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.status.setText("Stopping camera and releasing the device...")

    @Slot()
    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.workspace.record_event("vision.camera.stopped", {})

    @Slot(object)
    def _metadata_ready(self, metadata: dict[str, Any]) -> None:
        self._camera_metadata = dict(metadata)
        requested = metadata.get("requested_camera", {})
        readback = metadata.get("readback_camera", {})
        values = []
        for name in ("exposure", "focus", "white_balance"):
            entry = readback.get(name, {})
            values.append(f"{name}: {requested.get(name, '--')} -> {entry.get('actual', '--')}")
        self.camera_readback.setText(" | ".join(values))
        self.status.setText("Camera live. Preview is active; draw an ROI or load a local model.")
        self._profile.backend = metadata.get("backend")
        self._profile.readback_camera = readback
        self._profile.frame_size = (
            int(metadata.get("readback_width", 0)),
            int(metadata.get("readback_height", 0)),
        ) if metadata.get("readback_width") and metadata.get("readback_height") else None
        self._save_profile()

    @Slot(str, str)
    def _model_status(self, state: str, value: str) -> None:
        if state == "loaded":
            self.device_label.setText(f"Device: {value}")
            self._profile.model_device = value
            self.workspace.record_event("vision.model.loaded", {"model": self.model_path.text(), "device": value})
        else:
            self.device_label.setText("Device: model unavailable")
            self.workspace.record_event("vision.model.load_failed", {"model": self.model_path.text(), "error": value}, outcome="failed")

    @Slot(str)
    def _worker_error(self, message: str) -> None:
        self.status.setText(message)

    def _settings_changed(self, *_args: Any) -> None:
        settings = self._settings()
        if self._worker is not None:
            self._worker.update_settings(settings)
        self._save_profile()

    def _set_roi(self, x: int, y: int, width: int, height: int) -> None:
        self._set_roi_from_mapping({"x": x, "y": y, "width": width, "height": height})

    def _set_roi_from_mapping(self, roi: dict[str, Any], *, log_event: bool = True) -> None:
        try:
            frame_size = self._last_result.full_frame_size if self._last_result else self._profile.frame_size
            if frame_size is None:
                self.status.setText("Wait for a frame before setting an ROI.")
                return
            normalised = normalise_roi((roi["x"], roi["y"], roi["width"], roi["height"]), frame_size)
        except (KeyError, TypeError, ValueError, VisionError) as error:
            self.status.setText(f"Invalid ROI: {error}")
            return
        self.raw_view.set_roi(normalised)
        self.roi_label.setText(f"ROI: x={normalised[0]} y={normalised[1]} width={normalised[2]} height={normalised[3]}")
        self._settings_changed()
        if log_event:
            self.workspace.record_event("vision.roi.changed", {"roi": list(normalised)})

    def _clear_roi(self) -> None:
        self.raw_view.set_roi(None)
        self.roi_label.setText("ROI: full frame")
        self._settings_changed()
        self.workspace.record_event("vision.roi.changed", {"roi": None})

    @Slot()
    def _poll_frame(self) -> None:
        result = self.buffer.take_latest()
        if result is None:
            return
        self._last_result = result
        self.raw_view.set_image(_qimage_from_bgr(result.raw_bgr))
        self.raw_view.set_roi(result.roi)
        self.processed_view.set_image(_qimage_from_bgr(result.annotated_bgr))
        self.snapshot_button.setEnabled(True)
        self.device_label.setText(f"Device: {result.device}")
        self._populate_detections(result)

    def _populate_detections(self, result: FrameResult) -> None:
        self.detection_table.setRowCount(0)
        geometry_warning = ""
        for row, detection in enumerate(result.detections):
            self.detection_table.insertRow(row)
            robot_text = "--"
            if self._calibration is not None:
                try:
                    robot = robot_coordinates_for_detection(self._calibration, detection, result.full_frame_size)
                    robot_text = f"{robot[0]:.1f}, {robot[1]:.1f}"
                except CalibrationGeometryMismatchError as error:
                    geometry_warning = str(error)
                except CalibrationError as error:
                    geometry_warning = str(error)
            values = (
                str(detection.class_id),
                detection.class_name,
                f"{detection.confidence:.1%}",
                ", ".join(str(int(round(value))) for value in detection.bbox_xyxy),
                ", ".join(str(int(round(value))) for value in detection.center_px),
                str(detection.frame_id),
                robot_text,
            )
            for column, value in enumerate(values):
                self.detection_table.setItem(row, column, QTableWidgetItem(value))
        if self._calibration is None:
            self.calibration_warning.setText("No matching saved calibration loaded. Robot coordinates are not shown.")
        elif geometry_warning:
            self.calibration_warning.setText(geometry_warning)
        else:
            self.calibration_warning.setText("Saved calibration matches the current full-frame geometry.")

    def _load_calibration(self) -> None:
        path = self.workspace.resolve_inside("calibration", "calibration.json")
        if not path.is_file():
            self._calibration = None
            self.calibration_warning.setText("No saved M2 calibration found. Robot coordinates are not shown.")
            return
        try:
            self._calibration = CalibrationService.load(path)
            self.calibration_warning.setText("Saved M2 calibration loaded; geometry will be checked per frame.")
        except (OSError, CalibrationError, ValueError) as error:
            self._calibration = None
            self.calibration_warning.setText(f"Calibration unavailable: {error}")

    def _save_profile(self) -> None:
        try:
            model_text = self.model_path.text().strip() or None
            model_path = Path(model_text) if model_text else None
            self._profile.camera_index = self.camera_index.value()
            self._profile.confidence = self.confidence.value()
            self._profile.brightness = self.brightness.value()
            self._profile.contrast = self.contrast.value()
            self._profile.roi = (
                {"x": self.raw_view.roi[0], "y": self.raw_view.roi[1], "width": self.raw_view.roi[2], "height": self.raw_view.roi[3]}
                if self.raw_view.roi else None
            )
            self._profile.requested_camera = self._settings()["requested_camera"]
            resolved_model = str(model_path.resolve()) if model_path else None
            if resolved_model != self._profile.model_path:
                self._profile.model_sha256 = None
            self._profile.model_path = resolved_model
            save_profile(self.workspace.root, self._profile)
        except (OSError, VisionError):
            pass

    def _save_snapshot(self) -> None:
        if self._last_result is None:
            self.status.setText("Wait for a frame before saving a snapshot.")
            return
        try:
            paths = save_snapshot(
                self.workspace.root,
                self._last_result,
                {
                    "camera": self._camera_metadata,
                    "confidence": self.confidence.value(),
                    "brightness": self.brightness.value(),
                    "contrast": self.contrast.value(),
                    "model_path": self.model_path.text().strip() or None,
                    "model_device": self._last_result.device,
                },
            )
            self.workspace.record_event(
                "vision.snapshot.created",
                {"files": [str(path.relative_to(self.workspace.root)) for path in paths], "frame_id": self._last_result.frame_id},
            )
            self.status.setText(f"Snapshot saved: {paths[0].parent}")
        except (OSError, VisionError) as error:
            self.status.setText(f"Snapshot failed: {error}")

    def shutdown(self) -> None:
        if hasattr(self, "_timer"):
            self._timer.stop()
        self._stop_camera()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._save_profile()
