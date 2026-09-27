"""Threaded OpenCV capture and image loading for the calibration page."""

from __future__ import annotations

import threading
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Signal, Slot, QThread


class CameraOwnership:
    """Process-local ownership guard for the single shared webcam."""

    _lock = threading.Lock()
    _owner: str | None = None

    @classmethod
    def acquire(cls, owner: str) -> bool:
        with cls._lock:
            if cls._owner is not None and cls._owner != owner:
                return False
            cls._owner = owner
            return True

    @classmethod
    def release(cls, owner: str) -> None:
        with cls._lock:
            if cls._owner == owner:
                cls._owner = None

    @classmethod
    def current_owner(cls) -> str | None:
        with cls._lock:
            return cls._owner


class CameraWorker(QObject):
    frame_ready = Signal(object)
    metadata_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, camera_index: int) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.stop_requested = threading.Event()

    @Slot()
    def run(self) -> None:
        capture = None
        try:
            if sys.platform == "win32":
                capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened() and sys.platform == "win32":
                capture.release()
                capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                self.error.emit(f"Could not open camera {self.camera_index}.")
                return
            self.metadata_ready.emit(
                {
                    "source_type": "camera",
                    "camera_index": self.camera_index,
                    "backend": capture.getBackendName(),
                    "readback_width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                    "readback_height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    "readback_fps": capture.get(cv2.CAP_PROP_FPS),
                }
            )
            while not self.stop_requested.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.error.emit("Camera frame read failed.")
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb)
                QThread.msleep(33)
        except Exception as error:  # OpenCV backends can raise platform-specific errors.
            self.error.emit(str(error))
        finally:
            if capture is not None:
                capture.release()
            self.finished.emit()

    def stop(self) -> None:
        self.stop_requested.set()


class ImageLoadWorker(QObject):
    image_ready = Signal(object, object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    @Slot()
    def run(self) -> None:
        try:
            image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
            if image is None:
                self.error.emit(f"Could not read image: {self.path.name}")
                return
            self.image_ready.emit(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                {"source_type": "saved_image", "path": str(self.path)},
            )
        except Exception as error:
            self.error.emit(str(error))
        finally:
            self.finished.emit()
