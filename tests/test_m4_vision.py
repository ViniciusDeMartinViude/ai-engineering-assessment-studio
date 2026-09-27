from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_assessment.services.calibration import CalibrationGeometryMismatchError, CalibrationService
from ai_assessment.services.camera import CameraOwnership
from ai_assessment.services.vision import (
    Detection,
    LatestFrameBuffer,
    ModelLoadError,
    RawDetection,
    VisionProfile,
    VisionWorker,
    apply_software_correction,
    map_bbox_to_full_frame,
    process_frame,
    robot_coordinates_for_detection,
    save_profile,
    load_profile,
    validate_model_path,
)


class FakeAdapter:
    class_names = {2: "Ajwa"}
    device = "cpu"

    def __init__(self, bbox: tuple[float, float, float, float] = (5.0, 6.0, 15.0, 16.0)) -> None:
        self.bbox = bbox
        self.seen: np.ndarray | None = None

    def infer(self, image_bgr: np.ndarray, confidence: float) -> tuple[RawDetection, ...]:
        del confidence
        self.seen = image_bgr.copy()
        return (RawDetection(2, 0.91, self.bbox, model_size=(image_bgr.shape[1], image_bgr.shape[0])),)


class FakeCapture:
    def __init__(self, opened: bool, frames: list[np.ndarray] | None = None) -> None:
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def get(self, prop: int) -> float:
        del prop
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        del prop, value
        return False

    def getBackendName(self) -> str:
        return "fake"

    def release(self) -> None:
        self.released = True


class VisionServiceTests(unittest.TestCase):
    def test_camera_ownership_prevents_calibration_and_vision_competing(self) -> None:
        CameraOwnership.release("calibration")
        CameraOwnership.release("vision")
        self.assertTrue(CameraOwnership.acquire("calibration"))
        self.assertFalse(CameraOwnership.acquire("vision"))
        self.assertEqual(CameraOwnership.current_owner(), "calibration")
        CameraOwnership.release("calibration")
        self.assertTrue(CameraOwnership.acquire("vision"))
        CameraOwnership.release("vision")

    def test_resize_and_padding_are_reversed_before_roi_offset(self) -> None:
        # A 200x100 ROI letterboxed into 400x400 has scale 2 and 100px vertical padding.
        mapped = map_bbox_to_full_frame(
            (40.0, 160.0, 120.0, 240.0),
            (100, 50, 200, 100),
            (200, 100),
            (400, 400),
        )
        self.assertEqual(mapped, (120.0, 80.0, 160.0, 120.0))

    def test_correction_order_is_contrast_then_brightness(self) -> None:
        frame = np.full((1, 1, 3), 100, dtype=np.uint8)
        corrected = apply_software_correction(frame, brightness=10, contrast=2)
        self.assertEqual(int(corrected[0, 0, 0]), 210)

    def test_detection_structure_reports_full_frame_coordinates(self) -> None:
        adapter = FakeAdapter()
        result = process_frame(
            np.zeros((80, 100, 3), dtype=np.uint8),
            frame_id=7,
            roi=(10, 20, 50, 40),
            brightness=0,
            contrast=1,
            confidence=0.5,
            adapter=adapter,
            captured_at="2026-09-27T00:00:00Z",
        )
        detection = result.detections[0]
        self.assertEqual(detection.class_id, 2)
        self.assertEqual(detection.class_name, "Ajwa")
        self.assertEqual(detection.frame_id, 7)
        self.assertEqual(detection.coordinate_space, "full_frame")
        self.assertEqual(detection.bbox_xyxy, (15.0, 26.0, 25.0, 36.0))
        self.assertEqual(detection.center_px, (20.0, 31.0))
        self.assertEqual(adapter.seen.shape[:2], (40, 50))

    def test_latest_frame_buffer_drops_stale_results(self) -> None:
        buffer = LatestFrameBuffer()
        for frame_id in (1, 2, 3):
            result = process_frame(
                np.zeros((4, 4, 3), dtype=np.uint8),
                frame_id=frame_id,
                roi=None,
                brightness=0,
                contrast=1,
                confidence=0.5,
                adapter=None,
            )
            buffer.publish(result)
        self.assertEqual(buffer.take_latest().frame_id, 3)
        self.assertIsNone(buffer.take_latest())

    def test_geometry_mismatch_is_not_converted_to_robot_coordinates(self) -> None:
        camera_points = ((0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0))
        robot_points = tuple((u + 1.0, v + 2.0) for u, v in camera_points)
        record = CalibrationService.fit("affine", camera_points, robot_points, (640, 480))
        detection = Detection(0, "Ajwa", 0.9, (10.0, 20.0, 30.0, 40.0), (20.0, 30.0), 1, "now")
        with self.assertRaises(CalibrationGeometryMismatchError):
            robot_coordinates_for_detection(record, detection, (1280, 720))

    def test_profile_roundtrip_preserves_model_and_processing_settings(self) -> None:
        profile = VisionProfile(
            camera_index=2,
            backend="fake",
            requested_camera={"exposure": -6.0},
            readback_camera={"exposure": {"actual": -5.0}},
            frame_size=(1280, 720),
            roi={"x": 10, "y": 20, "width": 300, "height": 200},
            confidence=0.73,
            brightness=12,
            contrast=1.4,
            model_path="C:/models/date.pt",
            model_sha256="abc",
            model_device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_profile(Path(directory), profile)
            loaded = load_profile(Path(directory))
        self.assertEqual(path.name, "vision_profile.json")
        self.assertEqual(loaded.to_dict(), profile.to_dict())

    def test_camera_and_model_failures_are_reported_and_camera_released(self) -> None:
        camera_errors: list[str] = []
        unopened = FakeCapture(False)
        camera_worker = VisionWorker(0, {}, LatestFrameBuffer(), capture_factory=lambda index: unopened)
        camera_worker.error.connect(camera_errors.append)
        camera_worker.run()
        self.assertTrue(any("Could not open camera" in message for message in camera_errors))
        self.assertTrue(unopened.released)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "date.pt"
            model_path.write_bytes(b"fake")
            model_errors: list[str] = []
            capture = FakeCapture(True, [np.zeros((20, 20, 3), dtype=np.uint8)])

            def failing_loader(path: Path) -> FakeAdapter:
                raise ModelLoadError(f"bad model: {path.name}")

            worker = VisionWorker(
                0,
                {"model_path": str(model_path)},
                LatestFrameBuffer(),
                adapter_loader=failing_loader,
                capture_factory=lambda index: capture,
            )
            worker.error.connect(model_errors.append)
            worker.run()
            self.assertTrue(any("bad model" in message for message in model_errors))
            self.assertTrue(capture.released)

    def test_camera_preview_runs_without_a_model(self) -> None:
        capture = FakeCapture(True, [np.zeros((20, 30, 3), dtype=np.uint8)])
        buffer = LatestFrameBuffer()
        errors: list[str] = []
        worker = VisionWorker(0, {}, buffer, capture_factory=lambda index: capture)
        worker.error.connect(errors.append)
        worker.run()
        result = buffer.take_latest()
        self.assertIsNotNone(result)
        self.assertEqual(result.device, "preview only")
        self.assertEqual(result.detections, ())
        self.assertTrue(capture.released)

    def test_model_reference_rejects_missing_or_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ModelLoadError):
                validate_model_path(root / "missing.pt")
            bad = root / "weights.txt"
            bad.write_text("x", encoding="utf-8")
            with self.assertRaises(ModelLoadError):
                validate_model_path(bad)


if __name__ == "__main__":
    unittest.main()
