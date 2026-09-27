from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from .config import BASE_RADIUS_MM, NOZZLE_LENGTH_MM, Positions
from .model import RobotModel


class RobotCanvas(QWidget):
    camera_changed = Signal(float, float, float)

    def __init__(self, model: RobotModel, positions: Positions, parent: QWidget | None = None):
        super().__init__(parent)
        self.model = model
        self.positions = positions
        self.camera_yaw = math.radians(-28.0)
        self.camera_elevation = math.radians(27.0)
        self.zoom = 1.35
        self._last_mouse: QPoint | None = None
        self.setMinimumSize(650, 560)
        self.setMouseTracking(True)
        self.setToolTip("Drag to rotate the camera. Use the mouse wheel to zoom.")

    def reset_camera(self) -> None:
        self.camera_yaw = math.radians(-28.0)
        self.camera_elevation = math.radians(27.0)
        self.zoom = 1.35
        self.update()

    def _scale(self) -> float:
        return min(self.width() / 680.0, self.height() / 570.0) * self.zoom

    def _project(self, point: tuple[float, float, float]) -> tuple[QPointF, float]:
        x, y, z = point
        cy = math.cos(self.camera_yaw)
        sy = math.sin(self.camera_yaw)
        x1 = cy * x - sy * y
        depth_axis = sy * x + cy * y
        ce = math.cos(self.camera_elevation)
        se = math.sin(self.camera_elevation)
        vertical = z * ce - depth_axis * se
        depth = depth_axis * ce + z * se
        scale = self._scale()
        center_x = self.width() * 0.51
        center_y = self.height() * 0.67
        return QPointF(center_x + x1 * scale, center_y - vertical * scale), depth

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#F8FAFD"))
        gradient.setColorAt(1.0, QColor("#DDE6F1"))
        painter.fillRect(self.rect(), gradient)

        snapshot = self.model.snapshot()
        self._draw_floor(painter)
        self._draw_positions(painter, snapshot)
        self._draw_objects(painter, snapshot)
        self._draw_robot(painter, snapshot)
        self._draw_overlay(painter, snapshot)

    def _draw_floor(self, painter: QPainter) -> None:
        major_pen = QPen(QColor(112, 130, 154, 90), 1.2)
        minor_pen = QPen(QColor(112, 130, 154, 38), 1.0)
        extent = 330
        for coordinate in range(-300, 301, 30):
            painter.setPen(major_pen if coordinate % 90 == 0 else minor_pen)
            p1, _ = self._project((coordinate, -extent, 0))
            p2, _ = self._project((coordinate, extent, 0))
            painter.drawLine(p1, p2)
            p3, _ = self._project((-extent, coordinate, 0))
            p4, _ = self._project((extent, coordinate, 0))
            painter.drawLine(p3, p4)

        self._draw_axis(painter, (0, 0, 0), (90, 0, 0), QColor("#E5484D"), "X")
        self._draw_axis(painter, (0, 0, 0), (0, -90, 0), QColor("#30A46C"), "−Y")
        self._draw_axis(painter, (0, 0, 0), (0, 0, 90), QColor("#3E63DD"), "Z")

    def _draw_axis(
        self,
        painter: QPainter,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        color: QColor,
        label: str,
    ) -> None:
        a, _ = self._project(start)
        b, _ = self._project(end)
        painter.setPen(QPen(color, 2.2))
        painter.drawLine(a, b)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(b + QPointF(5, -4), label)

    def _draw_positions(self, painter: QPainter, snapshot: dict[str, Any]) -> None:
        current_target = snapshot["location"]
        for name, heights in self.positions.items():
            x, y, _z = heights["low"]
            center, _ = self._project((x, y, 1.0))
            radius = 15.0 * self._scale()
            rect = QRectF(center.x() - radius, center.y() - radius * 0.45, radius * 2, radius * 0.9)
            active = name == current_target
            fill = QColor("#9EB1FF" if active else "#E1E7F0")
            painter.setBrush(fill)
            painter.setPen(QPen(QColor("#3659E3" if active else "#8A98AC"), 2 if active else 1))
            painter.drawEllipse(rect)
            painter.setPen(QColor("#173B8F" if active else "#48566A"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(QRectF(center.x() - 22, center.y() + 7, 44, 20), Qt.AlignmentFlag.AlignCenter, name)

    def _draw_objects(self, painter: QPainter, snapshot: dict[str, Any]) -> None:
        objects = sorted(snapshot["objects"], key=lambda obj: self._project(tuple(obj["position"]))[1])
        for obj in objects:
            center, _ = self._project(tuple(obj["position"]))
            radius = max(6.0, 10.0 * self._scale())
            shadow = QRectF(center.x() - radius, center.y() + radius * 0.25, radius * 2, radius * 0.7)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(20, 27, 40, 42))
            painter.drawEllipse(shadow)
            body = QRectF(center.x() - radius, center.y() - radius * 0.65, radius * 2, radius * 1.3)
            painter.setBrush(QColor(obj["color"]))
            painter.setPen(QPen(QColor("#633B2A"), 1.2))
            painter.drawEllipse(body)
            if obj["attached"]:
                painter.setPen(QPen(QColor("#159455"), 2, Qt.PenStyle.DashLine))
                painter.drawEllipse(body.adjusted(-4, -4, 4, 4))

    def _draw_robot(self, painter: QPainter, snapshot: dict[str, Any]) -> None:
        joints = snapshot["joints"]
        shoulder = tuple(joints["shoulder"])
        elbow = tuple(joints["elbow"])
        wrist = tuple(joints["wrist"])
        base_bottom = (0.0, 0.0, 0.0)
        base_top = (0.0, 0.0, shoulder[2])

        # Soft floor shadow of the arm.
        shadow_points = [base_bottom, (elbow[0], elbow[1], 1.0), (wrist[0], wrist[1], 1.0)]
        painter.setPen(QPen(QColor(20, 29, 45, 38), 13, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for first, second in zip(shadow_points, shadow_points[1:]):
            a, _ = self._project(first)
            b, _ = self._project(second)
            painter.drawLine(a, b)

        self._draw_base(painter, base_bottom, base_top)

        # Main carbon-fibre links.
        self._draw_link(painter, shoulder, elbow, QColor("#252D3A"), 18)
        self._draw_link(painter, elbow, wrist, QColor("#303B4B"), 16)

        # Parallelogram brace: a visual approximation of MaxArm's linkage mechanism.
        base_yaw = math.radians(snapshot["joint_angles_degrees"]["base"])
        offset = 12.0
        side = (offset * math.cos(base_yaw), offset * math.sin(base_yaw), 0.0)
        shoulder_brace = (shoulder[0] + side[0], shoulder[1] + side[1], shoulder[2] + 7)
        elbow_brace = (elbow[0] + side[0], elbow[1] + side[1], elbow[2] + 7)
        wrist_brace = (wrist[0] + side[0], wrist[1] + side[1], wrist[2] + 7)
        self._draw_link(painter, shoulder_brace, elbow_brace, QColor("#E2A52B"), 5)
        self._draw_link(painter, elbow_brace, wrist_brace, QColor("#E2A52B"), 5)

        for point, size in ((shoulder, 13), (elbow, 12), (wrist, 10)):
            self._draw_joint(painter, point, size)

        nozzle_bottom = (wrist[0], wrist[1], wrist[2] - NOZZLE_LENGTH_MM)
        self._draw_link(painter, wrist, nozzle_bottom, QColor("#687386"), 8)
        tip, _ = self._project(nozzle_bottom)
        painter.setBrush(QColor("#39A96B" if snapshot["suction_on"] else "#E5484D"))
        painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
        painter.drawEllipse(QRectF(tip.x() - 7, tip.y() - 4, 14, 8))

    def _draw_base(
        self,
        painter: QPainter,
        bottom: tuple[float, float, float],
        top: tuple[float, float, float],
    ) -> None:
        base, _ = self._project(bottom)
        radius = BASE_RADIUS_MM * self._scale()
        base_rect = QRectF(base.x() - radius, base.y() - radius * 0.3, radius * 2, radius * 0.6)
        painter.setBrush(QColor("#1D2430"))
        painter.setPen(QPen(QColor("#596579"), 2))
        painter.drawEllipse(base_rect)
        top_point, _ = self._project(top)
        painter.setPen(QPen(QColor("#303947"), 28, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(base, top_point)
        painter.setPen(QPen(QColor("#768398"), 3))
        painter.drawLine(base + QPointF(-5, 0), top_point + QPointF(-5, 0))

    def _draw_link(
        self,
        painter: QPainter,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        color: QColor,
        width: float,
    ) -> None:
        a, _ = self._project(start)
        b, _ = self._project(end)
        painter.setPen(QPen(QColor(255, 255, 255, 70), width + 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(a, b)
        painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(a, b)

    def _draw_joint(self, painter: QPainter, point: tuple[float, float, float], radius: float) -> None:
        center, _ = self._project(point)
        scaled = max(7.0, radius * self._scale())
        painter.setBrush(QColor("#465266"))
        painter.setPen(QPen(QColor("#F0B429"), 3))
        painter.drawEllipse(QRectF(center.x() - scaled, center.y() - scaled, scaled * 2, scaled * 2))
        painter.setBrush(QColor("#C9D1DE"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(center.x() - 3, center.y() - 3, 6, 6))

    def _draw_overlay(self, painter: QPainter, snapshot: dict[str, Any]) -> None:
        box = QRectF(18, 18, 306, 92)
        painter.setBrush(QColor(255, 255, 255, 225))
        painter.setPen(QPen(QColor("#C9D3E2"), 1))
        painter.drawRoundedRect(box, 10, 10)
        painter.setPen(QColor("#182033"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        state = "MOVING" if snapshot["moving"] else "READY"
        text_flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(QRectF(32, 28, 275, 22), text_flags, f"MaxArm · {state}")
        painter.setFont(QFont("Consolas", 9))
        x, y, z = snapshot["current_xyz"]
        painter.drawText(
            QRectF(32, 52, 275, 20),
            text_flags,
            f"XYZ  {x:7.1f} {y:7.1f} {z:7.1f} mm",
        )
        angles = snapshot["joint_angles_degrees"]
        painter.drawText(
            QRectF(32, 73, 275, 20),
            text_flags,
            f"Joints {angles['base']:6.1f}° {angles['shoulder']:6.1f}° {angles['elbow']:6.1f}°",
        )
        painter.setPen(QColor("#667085"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(
            QRectF(18, self.height() - 30, 360, 18),
            text_flags,
            "Drag to rotate · Mouse wheel to zoom",
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._last_mouse is not None and event.buttons() & Qt.MouseButton.LeftButton:
            current = event.position().toPoint()
            delta = current - self._last_mouse
            self._last_mouse = current
            self.camera_yaw += delta.x() * 0.009
            self.camera_elevation = max(
                math.radians(8),
                min(math.radians(72), self.camera_elevation + delta.y() * 0.006),
            )
            self.update()
            self.camera_changed.emit(self.camera_yaw, self.camera_elevation, self.zoom)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # type: ignore[override]
        self._last_mouse = None
        self.unsetCursor()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        self.zoom = max(0.65, min(2.5, self.zoom * factor))
        self.update()
        self.camera_changed.emit(self.camera_yaw, self.camera_elevation, self.zoom)
