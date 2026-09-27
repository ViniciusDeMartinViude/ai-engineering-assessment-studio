"""Interactive four-point camera-to-robot planar calibration.

Run: python camera_robot_calibrator.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QButtonGroup, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QWidget,
)

from calibration_math import fit_affine, fit_homography, transform_point


APP_STYLE = """
QMainWindow, QWidget#root { background: #f3f7fb; color: #172b4d; }
QFrame#card { background: #ffffff; border: 1px solid #dce6ef; border-radius: 16px; }
QLabel#eyebrow { color: #078b80; font-size: 12px; font-weight: 800; letter-spacing: 1px; }
QLabel#pageTitle { color: #17304d; font-size: 29px; font-weight: 800; }
QLabel#pageSubtitle { color: #61748c; font-size: 13px; }
QLabel#cardTitle { color: #17304d; font-size: 18px; font-weight: 750; }
QLabel#fieldTitle { color: #61748c; font-size: 12px; font-weight: 750; }
QLabel#pixelValue { color: #17304d; font-size: 15px; font-weight: 700; }
QLabel#hint { background: #fff6e5; color: #795214; border-radius: 9px; padding: 12px; font-size: 13px; }
QLabel#status { background: #e9f8f5; color: #14675f; border-radius: 9px; padding: 11px; font-size: 13px; }
QLabel#testResult { background: #087f75; color: white; border-radius: 12px; padding: 18px; font-size: 32px; font-weight: 800; }
QLabel#matrixNote { color: #61748c; font-size: 12px; }
QDoubleSpinBox { background: #ffffff; color: #17304d; border: 1px solid #cbd9e6;
    border-radius: 8px; padding: 5px 10px; min-height: 31px; font-size: 13px; }
QDoubleSpinBox { font-size: 15px; font-weight: 650; }
QDoubleSpinBox:focus { border: 2px solid #0b9d91; }
QLabel#cameraSelector { background: #f0f7fa; color: #17304d; border: 1px solid #cbd9e6;
    border-radius: 8px; padding: 8px; font-size: 14px; font-weight: 750; }
QPushButton#cameraStep { min-width: 33px; max-width: 33px; font-size: 20px; padding: 4px; }
QRadioButton#matrixChoice { background: #f5f8fc; color: #254260; border: 1px solid #cfdeeb;
    border-radius: 9px; padding: 9px 13px; font-size: 13px; font-weight: 700; }
QRadioButton#matrixChoice:hover { background: #e7f3f7; border-color: #60bcb5; }
QRadioButton#matrixChoice:checked { background: #ddf6f1; border: 2px solid #0b9d91; color: #086c63; }
QRadioButton#matrixChoice::indicator { width: 17px; height: 17px; }
QPushButton { background: #f5f8fc; color: #254260; border: 1px solid #cfdeeb;
    border-radius: 9px; padding: 8px 12px; min-height: 31px; font-size: 13px; font-weight: 650; }
QPushButton:hover { background: #e7f3f7; border-color: #60bcb5; }
QPushButton#primaryButton { background: #087f75; border-color: #087f75; color: white; font-weight: 800; }
QPushButton#primaryButton:hover { background: #076e65; }
QPushButton#pointButton:checked { background: #ddf6f1; border: 2px solid #0b9d91; color: #086c63; font-weight: 800; }
QPushButton:disabled { background: #e9eef3; color: #94a3b3; border-color: #e1e9f0; }
QPlainTextEdit#matrix { background: #f2f6fb; color: #17304d; border: 1px solid #d4e2ed;
    border-radius: 10px; padding: 12px; selection-background-color: #b6e9e2; }
QScrollArea { background: transparent; border: none; }
"""


def format_matrix(matrix):
    """Align rounded values in a bracketed, monospaced 2×3 or 3×3 display."""
    values = [["0.00" if abs(value) < 0.005 else f"{value:.2f}" for value in row] for row in matrix]
    widths = [max(len(row[col]) for row in values) for col in range(3)]
    left = ("⎡", "⎣") if len(values) == 2 else ("⎡", "⎢", "⎣")
    right = ("⎤", "⎦") if len(values) == 2 else ("⎤", "⎥", "⎦")
    return "\n".join(
        f"{left[i]}  {'   '.join(value.rjust(widths[j]) for j, value in enumerate(row))}  {right[i]}"
        for i, row in enumerate(values)
    )


class ImageView(QWidget):
    clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(540, 360)
        self.image = QImage()
        self.points = [None] * 4
        self.test_point = None
        self.hover_pixel = None
        self.hover_widget_x = 0
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, image):
        if image.isNull() or (not self.image.isNull() and self.image.size() != image.size()):
            self.hover_pixel = None
        self.image = image
        self.update()

    def image_rect(self):
        if self.image.isNull():
            return QRectF()
        ratio = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * ratio, self.image.height() * ratio
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def pixel_at(self, position):
        rectangle = self.image_rect()
        if not rectangle.contains(position):
            return None
        u = max(0.0, min(self.image.width() - 1.0,
                         (position.x() - rectangle.x()) * self.image.width() / rectangle.width()))
        v = max(0.0, min(self.image.height() - 1.0,
                         (position.y() - rectangle.y()) * self.image.height() / rectangle.height()))
        return u, v

    def mouseMoveEvent(self, event):
        pixel = self.pixel_at(event.position())
        self.hover_pixel = tuple(round(value) for value in pixel) if pixel is not None else None
        self.hover_widget_x = event.position().x()
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.hover_pixel = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        pixel = self.pixel_at(event.position())
        if event.button() == Qt.MouseButton.LeftButton and pixel is not None:
            u, v = pixel
            self.clicked.emit(u, v)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#eaf1f7"))
        if self.image.isNull():
            painter.setPen(QColor("#526b83"))
            painter.setFont(QFont("Segoe UI", 15))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Start a camera or open an image\nThen select four reference points")
            return
        rectangle = self.image_rect()
        painter.drawImage(rectangle, self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for index, point in enumerate(self.points):
            if point is not None:
                self._draw_marker(painter, rectangle, point, str(index + 1), QColor("#ffad33"))
        if self.test_point is not None:
            self._draw_marker(painter, rectangle, self.test_point, "T", QColor("#ffd447"))
        if self.hover_pixel is not None:
            self._draw_magnifier(painter)

    def _draw_magnifier(self, painter):
        u, v = self.hover_pixel
        # Draw a 35×35 pixel tile at 5×. Pixels outside the image are filled
        # with the same background so the crosshair stays centered at edges.
        span, scale = 35, 5
        tile = QImage(span, span, QImage.Format.Format_RGB32)
        tile.fill(QColor("#eaf1f7"))
        tile_painter = QPainter(tile)
        tile_painter.drawImage(QPoint(span // 2 - u, span // 2 - v), self.image)
        tile_painter.end()

        side, top = span * scale + 20, 12
        left = 12 if self.hover_widget_x > self.width() / 2 else self.width() - side - 12
        card = QRectF(left, top, side, side + 42)
        image_target = QRectF(left + 10, top + 34, span * scale, span * scale)
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor("#087f75"), 2))
        painter.drawRoundedRect(card, 9, 9)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor("#17304d"))
        font = QFont("Segoe UI", 10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(left + 10, top + 4, side - 20, 25),
                         Qt.AlignmentFlag.AlignVCenter,
                         f"{scale}×   u={u}   v={v}")
        painter.save()
        painter.setClipRect(image_target)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(image_target, tile)
        painter.restore()
        painter.setPen(QPen(QColor("#cbd9e6"), 1))
        painter.drawRect(image_target)
        cx, cy = image_target.center().x(), image_target.center().y()
        for color, thickness in ((QColor("#ffffff"), 4), (QColor("#f16b35"), 2)):
            painter.setPen(QPen(color, thickness))
            painter.drawLine(int(cx - 12), int(cy), int(cx + 12), int(cy))
            painter.drawLine(int(cx), int(cy - 12), int(cx), int(cy + 12))

    def _draw_marker(self, painter, rectangle, point, label, color):
        px = rectangle.x() + point[0] * rectangle.width() / self.image.width()
        py = rectangle.y() + point[1] * rectangle.height() / self.image.height()
        painter.setPen(QPen(QColor("#111111"), 4))
        painter.drawEllipse(int(px - 6), int(py - 6), 12, 12)
        painter.setPen(QPen(color, 3))
        painter.drawEllipse(int(px - 6), int(py - 6), 12, 12)
        painter.setPen(QPen(color, 2))
        painter.drawLine(int(px - 12), int(py), int(px + 12), int(py))
        painter.drawLine(int(px), int(py - 12), int(px), int(py + 12))
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(int(px + 13), int(py - 10), label)


def numeric_box():
    box = QDoubleSpinBox()
    box.setRange(-1_000_000_000, 1_000_000_000)
    box.setDecimals(1)
    box.setSingleStep(0.1)
    box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    box.setAlignment(Qt.AlignmentFlag.AlignRight)
    box.setMinimumWidth(105)
    return box


class Calibrator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera → Robot Coordinate Calibration")
        self.resize(1420, 890)
        self.setMinimumSize(1160, 720)
        self.setStyleSheet(APP_STYLE)
        self.capture = None
        self.last_frame = None
        self.frozen = False
        self.points = [None] * 4
        self.h = None
        self.robot_values_exact = [(0.0, 0.0)] * 4
        self.edit_point_mode = True
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.read_frame)

        self.view = ImageView()
        self.view.clicked.connect(self.on_image_clicked)
        self.status = QLabel("Choose a camera or an image. Freeze the frame before selecting points.")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)

        self.camera_number = 0
        self.camera_label = QLabel("Camera 0")
        self.camera_label.setObjectName("cameraSelector")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.previous_camera_button = QPushButton("−")
        self.previous_camera_button.setObjectName("cameraStep")
        self.previous_camera_button.clicked.connect(lambda: self.change_camera(-1))
        self.next_camera_button = QPushButton("+")
        self.next_camera_button.setObjectName("cameraStep")
        self.next_camera_button.clicked.connect(lambda: self.change_camera(1))
        self.update_camera_controls()
        self.start_button = QPushButton("Start camera")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_camera)
        self.freeze_button = QPushButton("Freeze frame")
        self.freeze_button.clicked.connect(self.toggle_freeze)
        self.open_button = QPushButton("Open image…")
        self.open_button.clicked.connect(self.open_image)
        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        for widget in (self.previous_camera_button, self.camera_label, self.next_camera_button):
            source_row.addWidget(widget)
        zoom_tip = QLabel("Hover image for 5× zoom")
        zoom_tip.setObjectName("fieldTitle")
        source_row.addWidget(zoom_tip)
        source_row.addStretch()
        source_actions = QHBoxLayout()
        source_actions.setSpacing(8)
        for widget in (self.start_button, self.freeze_button, self.open_button):
            source_actions.addWidget(widget)

        left = QFrame()
        left.setObjectName("card")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(20, 18, 20, 20)
        left_layout.setSpacing(14)
        view_title = QLabel("01  Camera workspace")
        view_title.setObjectName("cardTitle")
        left_layout.addWidget(view_title)
        left_layout.addLayout(source_row)
        left_layout.addLayout(source_actions)
        left_layout.addWidget(self.view, 1)
        self.test_result = QLabel("ROBOT X: —     Y: —")
        self.test_result.setObjectName("testResult")
        self.test_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.test_result.setMinimumHeight(84)
        left_layout.addWidget(self.test_result)
        left_layout.addWidget(self.status)

        instructions = QLabel(
            "Click four markers on the flat work surface and enter each marker's "
            "robot X/Y. After calibration, click the image to see a robot position. "
            "Select a numbered row to edit a marker."
        )
        instructions.setObjectName("hint")
        instructions.setWordWrap(True)
        table = QGridLayout()
        table.setHorizontalSpacing(9)
        table.setVerticalSpacing(6)
        for col, label in enumerate(("POINT", "PIXEL U", "PIXEL V", "ROBOT X", "ROBOT Y")):
            title = QLabel(label)
            title.setObjectName("fieldTitle")
            table.addWidget(title, 0, col)
        self.select_buttons = []
        self.pixel_labels = []
        self.robot_fields = []
        for row in range(4):
            button = QPushButton(f"{row + 1:02d}")
            button.setObjectName("pointButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked, n=row: self.select_row(n, editing=True))
            self.select_buttons.append(button)
            u_label, v_label = QLabel("—"), QLabel("—")
            for pixel_label in (u_label, v_label):
                pixel_label.setObjectName("pixelValue")
                pixel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pixel_labels.append((u_label, v_label))
            x, y = numeric_box(), numeric_box()
            x.valueChanged.connect(self.invalidate)
            y.valueChanged.connect(self.invalidate)
            self.robot_fields.append((x, y))
            for col, widget in enumerate((button, u_label, v_label, x, y)):
                table.addWidget(widget, row + 1, col)
        self.selected_row = 0
        self.select_row(0)

        self.homography_choice = QRadioButton("3 × 3  Perspective")
        self.affine_choice = QRadioButton("2 × 3  Affine")
        for choice in (self.homography_choice, self.affine_choice):
            choice.setObjectName("matrixChoice")
        self.model_group = QButtonGroup(self)
        self.model_group.setExclusive(True)
        self.model_group.addButton(self.homography_choice)
        self.model_group.addButton(self.affine_choice)
        self.homography_choice.setChecked(True)
        self.homography_choice.toggled.connect(lambda checked: self.invalidate() if checked else None)
        self.affine_choice.toggled.connect(lambda checked: self.invalidate() if checked else None)
        model_options = QHBoxLayout()
        model_options.setSpacing(8)
        model_options.addWidget(self.homography_choice)
        model_options.addWidget(self.affine_choice)
        self.calibrate_button = QPushButton("Calculate matrix")
        self.calibrate_button.setObjectName("primaryButton")
        self.calibrate_button.clicked.connect(self.calibrate)
        self.clear_button = QPushButton("Clear points")
        self.clear_button.clicked.connect(self.clear_points)
        self.matrix_text = QPlainTextEdit()
        self.matrix_text.setObjectName("matrix")
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setPlaceholderText("The matrix will appear here.")
        matrix_font = QFont("Consolas", 18)
        matrix_font.setStyleHint(QFont.StyleHint.Monospace)
        self.matrix_text.setFont(matrix_font)
        self.matrix_text.setMinimumHeight(120)
        self.matrix_text.setMaximumHeight(120)
        self.matrix_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.matrix_note = QLabel("Display rounded to 2 decimals; calculations and saved values use full precision.")
        self.matrix_note.setObjectName("matrixNote")
        self.matrix_note.setWordWrap(True)

        self.save_button = QPushButton("Save calibration…")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_session)
        self.load_button = QPushButton("Load calibration…")
        self.load_button.clicked.connect(self.load_session)
        action_row = QHBoxLayout()
        action_row.addWidget(self.save_button)
        action_row.addWidget(self.load_button)

        right = QFrame()
        right.setObjectName("card")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(24, 18, 24, 18)
        right_layout.setSpacing(8)
        config_title = QLabel("02  Calibrate the plane")
        config_title.setObjectName("cardTitle")
        right_layout.addWidget(config_title)
        right_layout.addWidget(instructions)
        right_layout.addLayout(table)
        model_label = QLabel("MATRIX TYPE")
        model_label.setObjectName("fieldTitle")
        right_layout.addWidget(model_label)
        right_layout.addLayout(model_options)
        buttons = QHBoxLayout()
        buttons.addWidget(self.calibrate_button)
        buttons.addWidget(self.clear_button)
        right_layout.addLayout(buttons)
        matrix_title = QLabel("CAMERA PIXEL → ROBOT PLANE")
        matrix_title.setObjectName("fieldTitle")
        right_layout.addWidget(matrix_title)
        right_layout.addWidget(self.matrix_text)
        right_layout.addWidget(self.matrix_note)
        right_layout.addLayout(action_row)
        right_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(520)
        scroll.setWidget(right)

        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 14, 24, 14)
        layout.setSpacing(10)
        eyebrow = QLabel("VISION  /  ROBOTICS")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Camera to robot calibration")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Capture four points, calculate the transform, then click anywhere on the work surface.")
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(eyebrow)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        content = QHBoxLayout()
        content.setSpacing(16)
        content.addWidget(left, 3)
        content.addWidget(scroll, 2)
        layout.addLayout(content, 1)
        self.setCentralWidget(root)

    def select_row(self, index, editing=False):
        self.selected_row = index
        if editing:
            self.edit_point_mode = True
            self.status.setText(f"Editing point {index + 1}. Click its position in the image.")
        for i, button in enumerate(self.select_buttons):
            button.setChecked(i == index)

    def invalidate(self, *_):
        self.h = None
        self.edit_point_mode = True
        self.save_button.setEnabled(False)
        self.matrix_text.clear()
        self.test_result.setText("ROBOT X: —     Y: —")
        self.view.test_point = None
        self.view.update()

    def selected_transform(self):
        return "homography" if self.homography_choice.isChecked() else "affine"

    def update_camera_controls(self):
        self.camera_label.setText(f"Camera {self.camera_number}")
        self.previous_camera_button.setEnabled(self.capture is None and self.camera_number > 0)
        self.next_camera_button.setEnabled(self.capture is None and self.camera_number < 9)

    def change_camera(self, delta):
        if self.capture is not None:
            return
        self.camera_number = max(0, min(9, self.camera_number + delta))
        self.update_camera_controls()
        self.status.setText(f"Camera {self.camera_number} selected. Press Start camera to preview it.")

    def stop_camera(self):
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.start_button.setText("Start camera")
        self.update_camera_controls()

    def start_camera(self):
        if self.capture is not None:
            self.stop_camera()
            self.status.setText("Camera stopped. The displayed image remains available.")
            return
        index = self.camera_number
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
        if not cap.isOpened() and sys.platform == "win32":
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            QMessageBox.warning(self, "Camera", f"Could not open camera {index}.")
            return
        self.stop_camera()
        self.clear_points()
        self.last_frame = None
        self.view.set_image(QImage())
        self.capture = cap
        self.update_camera_controls()
        self.frozen = False
        self.freeze_button.setText("Freeze frame")
        self.start_button.setText("Stop camera")
        self.timer.start()
        self.status.setText("Camera live. Freeze the image before clicking calibration points.")

    def read_frame(self):
        if self.capture is None or self.frozen:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.stop_camera()
            self.status.setText("Camera read failed. Reconnect or choose another camera.")
            return
        if self.last_frame is not None and frame.shape[:2] != self.last_frame.shape[:2]:
            self.clear_points()
        self.last_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        self.view.set_image(image.copy())

    def toggle_freeze(self):
        if self.capture is None:
            self.status.setText("Start a camera to use Freeze frame.")
            return
        if not self.frozen and self.last_frame is None:
            self.status.setText("Waiting for the first camera frame.")
            return
        if self.frozen and any(point is not None for point in self.points):
            answer = QMessageBox.question(
                self, "Resume live video", "Resuming video clears the selected points and matrix. Continue?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.clear_points()
        self.frozen = not self.frozen
        self.freeze_button.setText("Resume live" if self.frozen else "Freeze frame")
        self.status.setText("Frame frozen; select four points." if self.frozen else "Camera live; freeze to select points.")

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open calibration image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            QMessageBox.warning(self, "Image", "Could not read this image.")
            return
        self.stop_camera()
        self.clear_points()
        self.last_frame = frame
        self.frozen = True
        self.freeze_button.setText("Freeze frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.view.set_image(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy())
        self.status.setText(f"Loaded {Path(path).name} ({frame.shape[1]} × {frame.shape[0]} pixels).")

    def on_image_clicked(self, u, v):
        # A camera pixel is discrete; keep the underlying point and the displayed
        # point identical instead of storing a hidden subpixel fraction.
        u, v = int(round(u)), int(round(v))
        if self.capture is not None and not self.frozen:
            self.toggle_freeze()
        if self.h is not None and not self.edit_point_mode:
            try:
                x, y = transform_point(self.h, u, v)
                self.test_result.setText(f"ROBOT X: {round(x):,}     Y: {round(y):,}")
                self.status.setText(f"Test pixel ({u}, {v}); robot position shown above.")
            except ValueError as error:
                self.test_result.setText("ROBOT X: —     Y: —")
                self.status.setText(str(error))
                return
            self.view.test_point = (u, v)
            self.view.update()
            return
        row = self.selected_row
        self.points[row] = (u, v)
        self.view.points = self.points[:]
        self.pixel_labels[row][0].setText(str(u))
        self.pixel_labels[row][1].setText(str(v))
        self.invalidate()
        self.view.update()
        self.status.setText(f"Point {row + 1}: pixel ({u}, {v}). Enter its robot X/Y.")
        for next_row in range(row + 1, 4):
            if self.points[next_row] is None:
                self.select_row(next_row)
                break

    def clear_points(self):
        self.points = [None] * 4
        self.view.points = self.points[:]
        self.view.test_point = None
        for labels in self.pixel_labels:
            for label in labels:
                label.setText("—")
        self.select_row(0)
        self.edit_point_mode = True
        self.invalidate()
        self.view.update()

    def calibrate(self):
        if any(point is None for point in self.points):
            QMessageBox.warning(self, "Calibration", "Click all four camera points first.")
            return
        robot = [(x.value(), y.value()) for x, y in self.robot_fields]
        try:
            fit = fit_homography if self.selected_transform() == "homography" else fit_affine
            self.h = fit(self.points, robot)
        except ValueError as error:
            QMessageBox.warning(self, "Calibration", str(error))
            return
        self.matrix_text.setPlainText(format_matrix(self.h))
        self.edit_point_mode = False
        for button in self.select_buttons:
            button.setChecked(False)
        self.robot_values_exact = robot
        self.save_button.setEnabled(True)
        self.status.setText("Matrix ready. Click the image to see robot X/Y; select a numbered row to edit.")

    def save_session(self):
        if self.h is None:
            QMessageBox.warning(self, "Save", "Calculate a matrix before saving.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration", "camera_robot_calibration.json", "JSON (*.json)")
        if not path:
            return
        try:
            image = None
            if not self.view.image.isNull():
                # Store the frozen source image so the saved pixel positions remain meaningful.
                from PySide6.QtCore import QBuffer, QIODevice
                buffer = QBuffer()
                buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                if not self.view.image.save(buffer, "PNG"):
                    raise ValueError("Could not encode the image.")
                image = base64.b64encode(bytes(buffer.data())).decode("ascii")
            data = {
                "schema_version": 2,
                "transform": self.selected_transform(),
                "coordinate_convention": "pixel origin top-left; u right, v down; robot X/Y in entered units",
                "image_width": self.view.image.width(),
                "image_height": self.view.image.height(),
                "camera_points_uv": self.points,
                "robot_points_xy": self.robot_values_exact,
                "matrix": self.h.tolist(),
                "snapshot_png_base64": image,
            }
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.status.setText(f"Saved {Path(path).name}.")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Save", str(error))

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            version = data.get("schema_version")
            if version not in (1, 2):
                raise ValueError("Unsupported calibration file version.")
            transform_type = "homography" if version == 1 else data["transform"]
            if transform_type not in ("homography", "affine"):
                raise ValueError("Unknown transform type.")
            image_bytes = base64.b64decode(data["snapshot_png_base64"], validate=True)
            image = QImage.fromData(image_bytes, "PNG")
            if image.isNull() or image.width() != data["image_width"] or image.height() != data["image_height"]:
                raise ValueError("Missing or invalid source image.")
            pixels = np.asarray(data["camera_points_uv"], dtype=float)
            robot = np.asarray(data["robot_points_xy"], dtype=float)
            if pixels.shape != (4, 2) or robot.shape != (4, 2):
                raise ValueError("Expected four camera and robot point pairs.")
            # Recalculate instead of trusting a potentially edited saved matrix.
            fitted = (fit_homography if transform_type == "homography" else fit_affine)(pixels, robot)
            stored = np.asarray(data["matrix_3x3"] if version == 1 else data["matrix"], dtype=float)
            if stored.shape != fitted.shape or not np.allclose(fitted, stored, rtol=1e-7, atol=1e-7):
                raise ValueError("Saved matrix does not match the saved points.")
        except (OSError, KeyError, ValueError, TypeError, IndexError) as error:
            QMessageBox.warning(self, "Load", f"Cannot load calibration: {error}")
            return
        self.stop_camera()
        self.clear_points()
        self.last_frame = None
        self.frozen = True
        self.freeze_button.setText("Freeze frame")
        self.view.set_image(image)
        (self.homography_choice if transform_type == "homography" else self.affine_choice).setChecked(True)
        self.points = [tuple(point) for point in pixels]
        self.view.points = self.points[:]
        for row, ((u, v), (x, y)) in enumerate(zip(pixels, robot)):
            self.pixel_labels[row][0].setText(str(round(u)))
            self.pixel_labels[row][1].setText(str(round(v)))
            for box, value in zip(self.robot_fields[row], (x, y)):
                box.blockSignals(True)
                box.setValue(float(value))
                box.blockSignals(False)
        self.h = fitted
        self.robot_values_exact = [tuple(pair) for pair in robot]
        self.matrix_text.setPlainText(format_matrix(fitted))
        self.edit_point_mode = False
        for button in self.select_buttons:
            button.setChecked(False)
        self.save_button.setEnabled(True)
        self.view.update()
        self.status.setText(f"Loaded {Path(path).name}. Calibration ready.")

    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = Calibrator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
