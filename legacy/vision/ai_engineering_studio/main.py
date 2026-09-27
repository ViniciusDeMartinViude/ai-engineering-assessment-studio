"""AI Engineering Studio: live vision, MaxArm simulator, HTTP console and calibration."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from PySide6.QtCore import QObject, QPoint, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from calibration.camera_robot_calibrator import APP_STYLE, Calibrator
from calibration.calibration_math import transform_point
from dataset.exporter import MainWindow as DatasetExporter
from maxarm_sim.api import ApiServerThread
from maxarm_sim.config import load_positions
from maxarm_sim.model import RobotModel
from maxarm_sim.render import RobotCanvas


def image_from_bgr(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                  QImage.Format.Format_RGB888).copy()


class ImagePanel(QWidget):
    roi_selected = Signal(tuple)

    def __init__(self, caption: str):
        super().__init__()
        self.caption = caption
        self.image = QImage()
        self.roi = None
        self.anchor = None
        self.drag = None
        self.setMinimumSize(280, 220)
        self.setMouseTracking(True)

    def set_image(self, image: QImage):
        self.image = image
        self.update()

    def image_rect(self):
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        w, h = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def source_point(self, point: QPoint):
        rect = self.image_rect()
        if rect.isEmpty() or not rect.contains(point):
            return None
        return (max(0, min(self.image.width() - 1,
                 int((point.x() - rect.x()) * self.image.width() / rect.width()))),
                max(0, min(self.image.height() - 1,
                 int((point.y() - rect.y()) * self.image.height() / rect.height()))))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#14273a"))
        if self.image.isNull():
            painter.setPen(QColor("#b2c8d8"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.caption)
            return
        rect = self.image_rect()
        painter.drawImage(rect, self.image)
        roi = self.drag or self.roi
        if roi:
            x, y, w, h = roi
            box = QRectF(rect.x() + x * rect.width() / self.image.width(),
                         rect.y() + y * rect.height() / self.image.height(),
                         w * rect.width() / self.image.width(),
                         h * rect.height() / self.image.height())
            painter.setPen(QPen(QColor("#ffb43c"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(box)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.anchor = self.source_point(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if self.anchor:
            end = self.source_point(event.position().toPoint())
            if end:
                a, b = self.anchor, end
                self.drag = (min(a[0], b[0]), min(a[1], b[1]),
                             abs(a[0] - b[0]), abs(a[1] - b[1]))
                self.update()

    def mouseReleaseEvent(self, event):
        if self.anchor:
            end = self.source_point(event.position().toPoint())
            if end and abs(end[0] - self.anchor[0]) >= 12 and abs(end[1] - self.anchor[1]) >= 12:
                self.roi = self.drag
                self.roi_selected.emit(self.roi)
            self.anchor = self.drag = None
            self.update()


class CameraWorker(QThread):
    frame_ready = Signal(object, object, object, object)  # raw, processed, detections, raw ndarray
    status = Signal(str)

    def __init__(self, camera: int, model_path: str, confidence: float,
                 brightness: float, contrast: float, camera_settings: dict | None = None):
        super().__init__()
        self.camera = camera
        self.model_path = model_path
        self.confidence = confidence
        self.brightness = brightness
        self.contrast = contrast
        self.camera_settings = camera_settings or {}
        self.roi = None
        self.inference_paused = False
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(self.camera)
        if not cap.isOpened() and sys.platform == "win32":
            cap.release()
            cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            self.status.emit(f"Could not open camera {self.camera}")
            cap.release()
            return
        settings = self.camera_settings
        for key, prop in (("Auto Exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
                          ("Exposure", cv2.CAP_PROP_EXPOSURE),
                          ("Auto Focus", cv2.CAP_PROP_AUTOFOCUS),
                          ("Focus", cv2.CAP_PROP_FOCUS),
                          ("Auto White Balance", cv2.CAP_PROP_AUTO_WB),
                          ("WB Temperature", cv2.CAP_PROP_WB_TEMPERATURE)):
            if key not in settings:
                continue
            if key == "Exposure" and settings.get("Auto Exposure"):
                continue
            if key == "Focus" and settings.get("Auto Focus"):
                continue
            if key == "WB Temperature" and settings.get("Auto White Balance"):
                continue
            value = float(settings[key])
            if key == "Auto Exposure":
                value = 0.75 if value else 0.25
            try:
                success = cap.set(prop, value)
                self.status.emit(f"{key}: requested {value:g}, set={success}, readback={cap.get(prop):g}")
            except Exception as exc:
                self.status.emit(f"{key}: {exc}")
        model = None
        try:
            if self.model_path:
                self.status.emit("Loading YOLO model…")
                from ultralytics import YOLO
                model = YOLO(self.model_path)
            self.status.emit(f"Camera {self.camera} running" + (" with YOLO" if model else " (preview only)"))
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.status.emit("Camera frame read failed")
                    break
                height, width = frame.shape[:2]
                roi = self.roi
                if roi:
                    x, y, w, h = roi
                    x1, y1 = max(0, min(width - 1, x)), max(0, min(height - 1, y))
                    x2, y2 = min(width, x + w), min(height, y + h)
                    if x2 <= x1 or y2 <= y1:
                        x1, y1, x2, y2 = 0, 0, width, height
                else:
                    x1, y1, x2, y2 = 0, 0, width, height
                crop = frame[y1:y2, x1:x2].copy()
                processed = np.clip(crop.astype(np.float32) * self.contrast + self.brightness,
                                    0, 255).astype(np.uint8)
                detections = []
                annotated = processed.copy()
                if model and not self.inference_paused:
                    result = model.predict(processed, conf=self.confidence, verbose=False)[0]
                    annotated = result.plot()
                    if result.boxes is not None:
                        for box in result.boxes:
                            a, b, c, d = box.xyxy[0].cpu().tolist()
                            cid = int(box.cls[0])
                            names = result.names
                            name = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
                            detections.append({"id": cid, "name": str(name),
                                               "confidence": float(box.conf[0]),
                                               "u": int(round(x1 + (a + c) / 2)),
                                               "v": int(round(y1 + (b + d) / 2))})
                            cv2.circle(annotated, (int((a+c)/2), int((b+d)/2)), 5, (0, 255, 255), -1)
                self.frame_ready.emit(image_from_bgr(frame), image_from_bgr(annotated),
                                      detections, frame)
                if model is None:
                    self.msleep(25)
        except Exception as exc:
            self.status.emit(f"Camera / YOLO error: {exc}")
        finally:
            cap.release()
            self.status.emit("Camera stopped")


class Events(QObject):
    log = Signal(str)
    sequence_finished = Signal()


class Studio(QMainWindow):
    def __init__(self, sim_port: int):
        super().__init__()
        self.setWindowTitle("AI Engineering Studio · Vision + MaxArm")
        self.resize(1640, 980)
        self.model = RobotModel(load_positions())
        self.server = ApiServerThread(self.model, "127.0.0.1", sim_port)
        self.server.start()
        self.sim_url = f"http://127.0.0.1:{sim_port}"
        self.camera_worker = None
        self.io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot-commands")
        self.events = Events()
        self.events.log.connect(self.log)
        self.events.sequence_finished.connect(self.sequence_finished)
        self.cancel_sequence = threading.Event()
        self.sequence_active = False
        self.last_detection_at = 0.0
        self.last_frame_size = None
        self.last_raw_frame = None
        self.pending_pick = None
        self.camera_settings = {}
        self.calibrator = Calibrator()
        self.calibrator.setMinimumSize(0, 0)
        self.dataset_exporter = DatasetExporter()
        self._build_ui()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_simulator)
        self.refresh_timer.start(50)
        self.log(f"Simulator API starting at {self.sim_url} (supports /docs)")

    def _build_ui(self):
        outer = QWidget()
        layout = QVBoxLayout(outer)
        layout.setContentsMargins(16, 12, 16, 12)
        top = QHBoxLayout()
        title = QLabel("AI ENGINEERING  /  STUDIO")
        title.setObjectName("title")
        top.addWidget(title)
        top.addStretch()
        self.status = QLabel("Ready · simulator")
        self.status.setObjectName("status")
        top.addWidget(self.status)
        layout.addLayout(top)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(outer)
        self._build_dashboard()
        self._build_calibration()
        self.tabs.addTab(self.dataset_exporter.takeCentralWidget(), "Dataset prints")
        self._build_help()
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f3f7fb; color: #19334e; font: 12px 'Segoe UI'; }
            QLabel#title { font-size: 23px; font-weight: 800; color: #087f75; }
            QLabel#status { background: #e2f5ef; border-radius: 8px; padding: 8px 14px; font-weight: 700; }
            QFrame#card { background: white; border: 1px solid #d9e5ef; border-radius: 12px; }
            QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
              min-height: 26px; padding: 4px 7px; border: 1px solid #bbcfdf;
              background: white; color: #19334e; border-radius: 6px; }
            QPushButton:hover, QComboBox:hover { background: #ddf5f0; border-color: #087f75; }
            QPushButton#primary { background: #087f75; color: white; font-weight: 700; }
            QPlainTextEdit { background: #102537; color: #d5f6ee; font: 12px Consolas; }
            QTabBar::tab { min-width: 130px; padding: 9px 13px; }
            QTabBar::tab:selected { background: #087f75; color: white; }
        """)

    @staticmethod
    def card(caption: str):
        widget = QFrame()
        widget.setObjectName("card")
        box = QVBoxLayout(widget)
        label = QLabel(caption)
        label.setStyleSheet("font-size:16px;font-weight:700;color:#087f75")
        box.addWidget(label)
        return widget, box

    def _build_dashboard(self):
        dashboard = QWidget()
        root = QVBoxLayout(dashboard)
        settings, sl = self.card("01  Camera and model")
        line1 = QHBoxLayout()
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 10)
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Select trained .pt / .onnx model, or leave empty for camera preview")
        if Path("yolo26n.pt").exists():
            self.model_path.setText(str(Path("yolo26n.pt").resolve()))
        choose = QPushButton("Select model…")
        choose.clicked.connect(self.choose_model)
        for item in (QLabel("Camera"), self.camera_index, QLabel("YOLO model"),
                     self.model_path, choose):
            line1.addWidget(item, 1 if item is self.model_path else 0)
        sl.addLayout(line1)
        settings_line = QHBoxLayout()
        self.settings_path = QLineEdit()
        self.settings_path.setPlaceholderText("Optional saved camera_settings.json · camera controls + software brightness/contrast")
        settings_button = QPushButton("Load camera settings…")
        settings_button.clicked.connect(self.choose_camera_settings)
        settings_line.addWidget(self.settings_path, 1)
        settings_line.addWidget(settings_button)
        sl.addLayout(settings_line)
        line2 = QHBoxLayout()
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.05, 0.99)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.5)
        self.brightness = QSpinBox()
        self.brightness.setRange(-100, 100)
        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.2, 3.0)
        self.contrast.setSingleStep(0.1)
        self.contrast.setValue(1)
        self.start_camera_button = QPushButton("Start camera")
        self.start_camera_button.setObjectName("primary")
        self.start_camera_button.clicked.connect(self.start_camera)
        self.stop_camera_button = QPushButton("Stop")
        self.stop_camera_button.clicked.connect(self.stop_camera)
        clear_roi = QPushButton("Clear ROI")
        clear_roi.clicked.connect(self.clear_roi)
        for item in (QLabel("Confidence"), self.confidence, QLabel("Brightness"), self.brightness,
                     QLabel("Contrast"), self.contrast, self.start_camera_button,
                     self.stop_camera_button, clear_roi):
            line2.addWidget(item)
        line2.addStretch()
        sl.addLayout(line2)
        root.addWidget(settings)

        center = QSplitter(Qt.Orientation.Horizontal)
        vision, vl = self.card("02  Live vision · drag a rectangle on the camera to select ROI")
        images = QSplitter(Qt.Orientation.Horizontal)
        self.raw_view = ImagePanel("Start camera to see the raw image")
        self.raw_view.roi_selected.connect(self.set_roi)
        self.processed_view = ImagePanel("Processed ROI and YOLO detections")
        for label, panel in (("RAW CAMERA", self.raw_view), ("PROCESSED + YOLO", self.processed_view)):
            holder = QWidget()
            box = QVBoxLayout(holder)
            box.addWidget(QLabel(label))
            box.addWidget(panel, 1)
            images.addWidget(holder)
        vl.addWidget(images, 1)
        self.detected = QLabel("No detections yet")
        self.detected.setMinimumHeight(32)
        vl.addWidget(self.detected)
        center.addWidget(vision)

        sim, sim_layout = self.card("03  MaxArm · local interactive simulator")
        self.canvas = RobotCanvas(self.model, self.model.positions)
        self.canvas.setMinimumSize(340, 280)
        sim_layout.addWidget(self.canvas, 1)
        self.sim_state = QLabel("P1 · high · suction off")
        sim_layout.addWidget(self.sim_state)
        center.addWidget(sim)
        center.setStretchFactor(0, 3)
        center.setStretchFactor(1, 2)
        root.addWidget(center, 1)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        robot, rb = self.card("04  Robot controls")
        route = QHBoxLayout()
        self.target_mode = QComboBox()
        self.target_mode.addItems(["Simulator", "Physical robot"])
        self.target_mode.currentIndexChanged.connect(self.target_changed)
        self.robot_url = QLineEdit(self.sim_url)
        self.robot_url.setPlaceholderText("http://192.168.1.167:8080")
        route.addWidget(self.target_mode)
        route.addWidget(self.robot_url, 1)
        rb.addLayout(route)
        move_row = QHBoxLayout()
        self.position = QComboBox()
        self.position.addItems([f"P{i}" for i in range(1, 6)])
        self.height = QComboBox()
        self.height.addItems(["high", "low"])
        self.duration = QSpinBox()
        self.duration.setRange(100, 10000)
        self.duration.setSingleStep(100)
        self.duration.setValue(1500)
        move_button = QPushButton("Move")
        move_button.setObjectName("primary")
        move_button.clicked.connect(lambda: self.submit({"command": "move", "location": self.position.currentText(),
                                                        "height": self.height.currentText(), "duration_ms": self.duration.value()}))
        for widget in (self.position, self.height, QLabel("ms"), self.duration, move_button):
            move_row.addWidget(widget)
        rb.addLayout(move_row)
        buttons = QHBoxLayout()
        for label, payload in (("Suction ON", {"command":"suction","state":"on"}),
                               ("Suction OFF", {"command":"suction","state":"off"})):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, p=payload: self.submit(p))
            buttons.addWidget(button)
        check = QPushButton("Health")
        check.clicked.connect(lambda: self.submit("GET /health"))
        buttons.addWidget(check)
        reset = QPushButton("Reset simulator")
        reset.clicked.connect(lambda: self.submit("POST /sim/reset", force_sim=True))
        buttons.addWidget(reset)
        rb.addLayout(buttons)
        auto_row = QHBoxLayout()
        self.auto = QCheckBox("Auto sort after detection")
        self.auto.toggled.connect(self.auto_toggled)
        self.trigger_class = QLineEdit("Ajwa")
        self.trigger_class.setToolTip("Case insensitive class name or numeric ID")
        self.destination = QComboBox()
        self.destination.addItems([f"P{i}" for i in range(2, 6)])
        self.destination.setCurrentText("P5")
        self.delay = QSpinBox()
        self.delay.setRange(0, 20)
        self.delay.setValue(5)
        for item in (self.auto, QLabel("Class"), self.trigger_class, QLabel("Place at"),
                     self.destination, QLabel("Wait / step (s)"), self.delay):
            auto_row.addWidget(item)
        rb.addLayout(auto_row)
        coordinates_row = QHBoxLayout()
        self.use_calibrated_pick = QCheckBox("Pick at detected X/Y (XYZ API)")
        self.pick_low_z = QSpinBox()
        self.pick_low_z.setRange(-100, 300)
        self.pick_low_z.setValue(59)
        self.pick_high_z = QSpinBox()
        self.pick_high_z.setRange(-100, 300)
        self.pick_high_z.setValue(89)
        for item in (self.use_calibrated_pick, QLabel("Low Z mm"), self.pick_low_z,
                     QLabel("High Z mm"), self.pick_high_z):
            coordinates_row.addWidget(item)
        rb.addLayout(coordinates_row)
        cancel = QPushButton("Cancel pending sequence")
        cancel.clicked.connect(self.cancel_pending)
        rb.addWidget(cancel)
        foot = QLabel("Auto sort: 5 s countdown, pickup high → low → suction on → pickup high → destination high → low → suction off → high.\n"
                      "Physical robot mode sends live commands. Cancel cannot stop movement already accepted by its controller.")
        foot.setWordWrap(True)
        rb.addWidget(foot)
        bottom.addWidget(robot)

        console, cl = self.card("05  Command terminal · HTTP / JSON")
        self.terminal = QPlainTextEdit()
        self.terminal.setReadOnly(True)
        cl.addWidget(self.terminal, 1)
        terminal_row = QHBoxLayout()
        self.command_line = QLineEdit()
        self.command_line.setPlaceholderText('move P2 high 1500  |  suction on  |  xyz -3 -130 89 1500  |  {"command":"move",...}')
        self.command_line.returnPressed.connect(self.run_line)
        send = QPushButton("Send")
        send.setObjectName("primary")
        send.clicked.connect(self.run_line)
        terminal_row.addWidget(self.command_line, 1)
        terminal_row.addWidget(send)
        cl.addLayout(terminal_row)
        bottom.addWidget(console)
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 1)
        root.addWidget(bottom, 1)
        self.tabs.addTab(dashboard, "Dashboard")

    def _build_calibration(self):
        # Embed the original calibration UI; only this window owns the video capture.
        page = self.calibrator.takeCentralWidget()
        page.setStyleSheet(APP_STYLE)
        self.calibrator.start_button.hide()
        self.calibrator.previous_camera_button.hide()
        self.calibrator.next_camera_button.hide()
        self.calibrator.camera_label.setText("Shared dashboard camera")
        self.calibrator.freeze_button.clicked.disconnect()
        self.calibrator.freeze_button.clicked.connect(self.toggle_calibration_freeze)
        self.calibrator.status.setText("Start the dashboard camera or open an image, freeze, then mark four points.")
        tools_row = QHBoxLayout()
        self.test_z = QSpinBox()
        self.test_z.setRange(-100, 300)
        self.test_z.setValue(89)
        try_button = QPushButton("Move simulator to tested point")
        try_button.clicked.connect(self.move_to_test_point)
        tools_row.addWidget(QLabel("Test Z (mm)"))
        tools_row.addWidget(self.test_z)
        tools_row.addWidget(try_button)
        tools_row.addStretch()
        page.layout().addLayout(tools_row)
        self.tabs.addTab(page, "Camera calibration")
        self.calibrator.view.clicked.connect(self.on_calibration_click)

    def _build_help(self):
        page = QWidget()
        box = QVBoxLayout(page)
        help_text = QPlainTextEdit()
        help_text.setReadOnly(True)
        help_text.setPlainText(
            "AI Engineering Studio\n\n"
            "1. Start camera. Select a local YOLO weights file to enable detections. "
            "Drag on the raw camera view to select a region; brightness and contrast apply before inference.\n\n"
            "2. Simulator starts on localhost. Click a position, choose high/low, then Move. "
            "Use the terminal to send commands; `help` lists examples. Rotate the simulator with the mouse.\n\n"
            "3. In Camera calibration, freeze a live frame or open an image. Mark four pixel points "
            "and enter their robot X/Y values. Choose affine 2×3 or perspective 3×3, calculate, test "
            "by clicking, and save a JSON calibration. The dashboard uses the same camera pixels.\n\n"
            "4. Click a test point and use Move simulator to tested point, choosing Z yourself. "
            "Optionally, enable Pick at detected X/Y so auto sorting uses calibration for pickup. "
            "Calibration maps only X/Y. The built in simulator accepts XYZ movement. Physical "
            "firmware must also support the XYZ JSON command for that feature.\n\n"
            "5. Dataset prints imports your YOLO test split and exports transparent PNGs, white "
            "JPEGs and A4 sheets with 6 × 4 cm cutouts and class names.\n\n"
            "6. Auto sort is off initially. When enabled, the named class triggers a countdown "
            "and pick/place sequence. A stationary detection must disappear for at least 0.75 s "
            "before it can trigger another sequence. Verify your real workspace before selecting a physical robot.\n\n"
            "Simulator API: GET /health, /positions, /suction, /sim/state; POST /command. "
            "Interactive simulator docs are available at the selected localhost port /docs."
        )
        box.addWidget(help_text)
        self.tabs.addTab(page, "Guide")

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", "", "YOLO models (*.pt *.onnx)")
        if path:
            self.model_path.setText(path)

    def choose_camera_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open camera settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            settings = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(settings, dict):
                raise ValueError("Expected a JSON object")
            contrast = float(settings.get("Contrast", 1.0))
            if contrast > 10:
                contrast /= 100
            self.brightness.setValue(int(settings.get("Brightness", 0)))
            self.contrast.setValue(contrast)
            self.camera_settings = settings
            self.settings_path.setText(path)
            self.log(f"Camera settings loaded; restart camera to apply hardware controls: {Path(path).name}")
        except (ValueError, OSError, TypeError) as exc:
            QMessageBox.warning(self, "Camera settings", str(exc))

    def start_camera(self):
        if self.camera_worker and self.camera_worker.isRunning():
            return
        path = self.model_path.text().strip()
        if path and not Path(path).is_file():
            QMessageBox.warning(self, "Model", "Select an existing .pt or .onnx model file.")
            return
        self.camera_worker = CameraWorker(self.camera_index.value(), path, self.confidence.value(),
                                          self.brightness.value(), self.contrast.value(), self.camera_settings)
        self.camera_worker.frame_ready.connect(self.on_frame)
        self.camera_worker.status.connect(self.on_camera_status)
        self.camera_worker.finished.connect(lambda: self.start_camera_button.setEnabled(True))
        self.camera_worker.start()
        self.start_camera_button.setEnabled(False)

    def stop_camera(self):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker.wait(5000)
            if self.camera_worker.isRunning():
                self.log("Camera worker is still stopping; wait for inference to complete.")
            else:
                self.camera_worker = None
                self.start_camera_button.setEnabled(True)

    def on_camera_status(self, message):
        self.log(message)
        self.status.setText(message)
        if self.camera_worker and not self.camera_worker.isRunning():
            self.start_camera_button.setEnabled(True)

    def set_roi(self, roi):
        if self.camera_worker:
            self.camera_worker.roi = roi
        self.log(f"ROI set to x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")

    def clear_roi(self):
        self.raw_view.roi = None
        self.raw_view.update()
        if self.camera_worker:
            self.camera_worker.roi = None
        self.log("ROI cleared")

    def on_frame(self, raw, processed, detections, frame):
        self.raw_view.set_image(raw)
        self.processed_view.set_image(processed)
        self.last_raw_frame = frame
        size = (raw.width(), raw.height())
        if self.last_frame_size and size != self.last_frame_size:
            self.calibrator.clear_points()
            self.log("Camera resolution changed; calibration points cleared")
        self.last_frame_size = size
        if not self.calibrator.frozen:
            self.calibrator.last_frame = frame
            self.calibrator.view.set_image(raw)
        if detections:
            best = max(detections, key=lambda item: item["confidence"])
            message = f"{best['name']} · class {best['id']} · {best['confidence']:.0%} · pixel ({best['u']}, {best['v']})"
            if self.calibrator.h is not None and size == (self.calibrator.view.image.width(), self.calibrator.view.image.height()):
                try:
                    x, y = transform_point(self.calibrator.h, best["u"], best["v"])
                    message += f"  →  robot X {round(x)}, Y {round(y)}"
                except ValueError:
                    pass
            self.detected.setText(message)
            trigger = self.trigger_class.text().strip().casefold()
            if self.auto.isChecked() and trigger and not self.sequence_active and time.monotonic() - self.last_detection_at > 0.75:
                matches = [d for d in detections if d["name"].casefold() == trigger or str(d["id"]) == trigger]
                if matches:
                    self.start_sequence(max(matches, key=lambda item: item["confidence"]))
        else:
            self.detected.setText("No detection" if not self.sequence_active else "Sorting in progress")
            if not self.sequence_active and self.last_detection_at == float("inf"):
                self.last_detection_at = time.monotonic()
        if detections and not self.sequence_active:
            self.last_detection_at = float("inf")

    def toggle_calibration_freeze(self):
        if not self.calibrator.frozen:
            if self.calibrator.view.image.isNull():
                self.calibrator.status.setText("Start the camera or open an image first.")
                return
            self.calibrator.frozen = True
            self.calibrator.freeze_button.setText("Resume live")
            self.calibrator.status.setText("Frame frozen. Select four points, then calculate the matrix.")
        else:
            if any(p is not None for p in self.calibrator.points):
                result = QMessageBox.question(self, "Resume live", "Clear points and matrix and resume live feed?")
                if result != QMessageBox.StandardButton.Yes:
                    return
                self.calibrator.clear_points()
            self.calibrator.frozen = False
            self.calibrator.freeze_button.setText("Freeze frame")

    def on_calibration_click(self, u, v):
        if not self.calibrator.frozen and not self.calibrator.view.image.isNull():
            self.calibrator.frozen = True
            self.calibrator.freeze_button.setText("Resume live")

    def move_to_test_point(self):
        h = self.calibrator.h
        point = self.calibrator.view.test_point
        if h is None or point is None:
            self.log("Calibrate four points and click a test point first")
            return
        try:
            x, y = transform_point(h, *point)
            if self.last_frame_size and self.last_frame_size != (
                    self.calibrator.view.image.width(), self.calibrator.view.image.height()):
                raise ValueError("Calibration image size differs from live camera size")
            self.submit({"command":"move", "x":round(x, 2), "y":round(y, 2),
                         "z":self.test_z.value(), "duration_ms":self.duration.value()}, force_sim=True)
        except ValueError as exc:
            self.log(f"Cannot move to test point: {exc}")

    def target_changed(self):
        physical = self.target_mode.currentIndex() == 1
        self.robot_url.setText("" if physical else self.sim_url)
        self.robot_url.setReadOnly(not physical)
        if physical:
            self.auto.setChecked(False)
        self.status.setText("Physical robot · enter its API address" if physical else "Local simulator")

    def auto_toggled(self, enabled):
        if enabled and self.target_mode.currentIndex() == 1:
            result = QMessageBox.question(self, "Enable live sorting",
                                          "Detection will send movement and suction commands to the physical robot. Enable?")
            if result != QMessageBox.StandardButton.Yes:
                self.auto.blockSignals(True)
                self.auto.setChecked(False)
                self.auto.blockSignals(False)

    def endpoint(self, force_sim=False):
        base = self.sim_url if force_sim else self.robot_url.text().strip()
        if not base.startswith(("http://", "https://")):
            raise ValueError("Robot URL must begin with http:// or https://")
        return base.rstrip("/")

    def log(self, message):
        self.terminal.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _request(self, base, operation):
        for attempt in range(6):
            try:
                if isinstance(operation, str) and operation.startswith("GET "):
                    result = requests.get(base + operation[4:], timeout=5)
                elif isinstance(operation, str) and operation.startswith("POST "):
                    result = requests.post(base + operation[5:], timeout=5)
                else:
                    result = requests.post(base + "/command", json=operation, timeout=5)
                break
            except requests.ConnectionError:
                if base != self.sim_url or attempt == 5:
                    raise
                time.sleep(0.2)
        try:
            data = result.json()
        except ValueError:
            data = result.text[:300]
        if not result.ok or isinstance(data, dict) and data.get("ok") is False:
            raise RuntimeError(f"HTTP {result.status_code}: {data}")
        return data

    def submit(self, operation, force_sim=False):
        try:
            base = self.endpoint(force_sim)
        except ValueError as exc:
            self.log(str(exc))
            return
        self.events.log.emit(f"→ {operation}")
        def task():
            try:
                data = self._request(base, operation)
                self.events.log.emit("← " + json.dumps(data, ensure_ascii=False))
            except Exception as exc:
                self.events.log.emit(f"Request failed: {exc}")
        self.io.submit(task)

    def run_line(self):
        line = self.command_line.text().strip()
        self.command_line.clear()
        if not line:
            return
        try:
            if line.startswith("{"):
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("JSON command must be an object")
            else:
                parts = shlex.split(line)
                cmd = parts[0].lower()
                if cmd == "help":
                    self.log("move P1 high [ms] | suction on/off | xyz X Y Z [ms] | health | positions | state | JSON body")
                    return
                if cmd in ("health", "positions", "state") and len(parts) == 1:
                    payload = "GET " + {"health":"/health", "positions":"/positions", "state":"/sim/state"}[cmd]
                elif cmd == "move" and len(parts) in (3, 4):
                    payload = {"command":"move", "location":parts[1].upper(), "height":parts[2].lower(),
                               "duration_ms": int(parts[3]) if len(parts) == 4 else self.duration.value()}
                elif cmd == "suction" and len(parts) == 2:
                    payload = {"command":"suction", "state":parts[1].lower()}
                elif cmd == "xyz" and len(parts) in (4, 5):
                    payload = {"command":"move", **dict(zip(("x","y","z"),map(float,parts[1:4]))),
                               "duration_ms":int(parts[4]) if len(parts) == 5 else self.duration.value()}
                else:
                    raise ValueError("Unknown command. Type help for examples.")
            self.submit(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            self.log(f"Command error: {exc}")

    def start_sequence(self, detection):
        if self.sequence_active:
            return
        self.pending_pick = None
        if self.use_calibrated_pick.isChecked():
            if self.calibrator.h is None or self.last_frame_size != (
                    self.calibrator.view.image.width(), self.calibrator.view.image.height()):
                self.log("Auto pick needs a calibration captured at the current camera resolution")
                return
            try:
                self.pending_pick = transform_point(self.calibrator.h, detection["u"], detection["v"])
            except ValueError as exc:
                self.log(f"Cannot map detection center: {exc}")
                return
        try:
            base = self.endpoint()
        except ValueError as exc:
            self.log(str(exc))
            return
        self.sequence_active = True
        self.cancel_sequence.clear()
        if self.camera_worker:
            self.camera_worker.inference_paused = True
        self.status.setText("SORTING · 5 second countdown")
        self.status.setStyleSheet("background:#b33131;color:white;padding:8px")
        self.log("Detection matched. Starting 5 second countdown")
        self.countdown = 5
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(lambda: self.countdown_tick(base))
        self.countdown_timer.start(1000)

    def countdown_tick(self, base):
        if self.cancel_sequence.is_set():
            self.countdown_timer.stop()
            self.sequence_finished()
            return
        self.countdown -= 1
        if self.countdown > 0:
            self.status.setText(f"SORTING · starting in {self.countdown} s")
            return
        self.countdown_timer.stop()
        destination = self.destination.currentText()
        duration = self.duration.value()
        wait_seconds = self.delay.value()
        if self.pending_pick is not None:
            x, y = self.pending_pick
            def pickup(height):
                return {"command":"move", "x":round(x, 2), "y":round(y, 2),
                        "z": self.pick_high_z.value() if height == "high" else self.pick_low_z.value(),
                        "duration_ms": duration}
        else:
            def pickup(height):
                return {"command":"move", "location":"P1", "height":height, "duration_ms":duration}
        commands = [
            pickup("high"),
            pickup("low"),
            {"command":"suction", "state":"on"},
            pickup("high"),
            {"command":"move", "location":destination, "height":"high", "duration_ms":duration},
            {"command":"move", "location":destination, "height":"low", "duration_ms":duration},
            {"command":"suction", "state":"off"},
            {"command":"move", "location":destination, "height":"high", "duration_ms":duration},
        ]
        self.io.submit(self._run_sequence, base, commands, wait_seconds)

    def _run_sequence(self, base, commands, delay):
        try:
            for i, command in enumerate(commands):
                if self.cancel_sequence.is_set():
                    self.events.log.emit("Sequence canceled before next command")
                    break
                self.events.log.emit(f"Step {i+1}/{len(commands)} → {command}")
                answer = self._request(base, command)
                self.events.log.emit("← " + json.dumps(answer, ensure_ascii=False))
                # The simulator acknowledges movement immediately; never issue next motion while busy.
                motion_wait = command.get("duration_ms", 0) / 1000 + 0.15
                if i < len(commands) - 1:
                    self.cancel_sequence.wait(max(delay, motion_wait))
        except Exception as exc:
            self.events.log.emit(f"Sequence stopped after request failure: {exc}")
        finally:
            self.events.sequence_finished.emit()

    def cancel_pending(self):
        self.cancel_sequence.set()
        self.log("Cancel requested; commands already accepted by the robot may finish")

    def sequence_finished(self):
        self.sequence_active = False
        if self.camera_worker:
            self.camera_worker.inference_paused = False
        self.last_detection_at = float("inf")
        self.status.setStyleSheet("")
        self.status.setText("Ready · " + self.target_mode.currentText().lower())

    def refresh_simulator(self):
        self.canvas.update()
        state = self.model.snapshot()
        xyz = state["current_xyz"]
        self.sim_state.setText(f"{state['location']} · {state['height']} · "
                               f"XYZ ({xyz[0]:.0f}, {xyz[1]:.0f}, {xyz[2]:.0f}) mm · "
                               f"suction {'ON' if state['suction_on'] else 'OFF'}"
                               + (" · moving" if state["moving"] else ""))
        if self.server.startup_error and not getattr(self, "_server_error_shown", False):
            self._server_error_shown = True
            self.log("Simulator API failed: " + self.server.startup_error)
        if self.server.server and self.server.server.started and not getattr(self, "_server_ready_shown", False):
            self._server_ready_shown = True
            self.log("Simulator API ready")

    def closeEvent(self, event):
        self.cancel_sequence.set()
        if self.countdown_timer.isActive() if hasattr(self, "countdown_timer") else False:
            self.countdown_timer.stop()
        self.stop_camera()
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.wait()
        self.io.shutdown(wait=False, cancel_futures=True)
        self.server.stop()
        self.server.join(timeout=3)
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="AI Engineering integrated vision and robot studio")
    parser.add_argument("--sim-port", type=int, default=8080)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = Studio(args.sim_port)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
