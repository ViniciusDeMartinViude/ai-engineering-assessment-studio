from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_assessment.services.calibration import (
    CalibrationError,
    CalibrationGeometryMismatchError,
    CalibrationService,
    fit_affine,
    fit_homography,
    transform_point,
)


class CalibrationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_points = ((0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0))

    def test_known_affine_mapping(self) -> None:
        robot_points = tuple(
            (2.0 * u + 3.0 * v + 10.0, -u + 4.0 * v + 5.0)
            for u, v in self.camera_points
        )

        matrix = fit_affine(self.camera_points, robot_points)

        np.testing.assert_allclose(
            matrix,
            ((2.0, 3.0, 10.0), (-1.0, 4.0, 5.0)),
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            transform_point(matrix, 25.0, 30.0),
            (150.0, 100.0),
            rtol=0.0,
            atol=1e-10,
        )

    def test_known_homography_mapping(self) -> None:
        expected = np.array(
            [[1.2, 0.1, 4.0], [0.05, 0.9, -2.0], [0.0005, -0.0008, 1.0]]
        )
        robot_points = tuple(
            transform_point(expected, u, v) for u, v in self.camera_points
        )

        matrix = fit_homography(self.camera_points, robot_points)

        np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=1e-8)
        np.testing.assert_allclose(
            transform_point(matrix, 40.0, 30.0),
            transform_point(expected, 40.0, 30.0),
            rtol=0.0,
            atol=1e-8,
        )

    def test_degenerate_points_are_rejected(self) -> None:
        collinear = ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0))

        with self.assertRaises(CalibrationError):
            fit_affine(collinear, self.camera_points)
        with self.assertRaises(CalibrationError):
            fit_homography(self.camera_points, collinear)

    def test_save_load_preserves_full_precision(self) -> None:
        robot_points = tuple(
            (2.0 * u + 3.0 * v + 10.0, -u + 4.0 * v + 5.0)
            for u, v in self.camera_points
        )
        record = CalibrationService.fit(
            "affine",
            self.camera_points,
            robot_points,
            (1920, 1080),
            camera_metadata={"source_type": "saved_image", "camera_index": None},
            roi=None,
            snapshot_png=b"png-placeholder",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "calibration.json"
            CalibrationService.save(path, record)
            loaded = CalibrationService.load(path)
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded.schema_version, 3)
        self.assertEqual(loaded.source_schema_version, 3)
        self.assertEqual(loaded.snapshot_png, b"png-placeholder")
        self.assertEqual(loaded.camera_metadata["source_type"], "saved_image")
        np.testing.assert_array_equal(np.asarray(loaded.matrix), np.asarray(record.matrix))
        self.assertEqual(raw["matrix"], [list(row) for row in record.matrix])

    def test_schema_one_and_two_legacy_files_load(self) -> None:
        homography = fit_homography(
            self.camera_points,
            tuple((u + 5.0, v + 7.0) for u, v in self.camera_points),
        )
        affine = fit_affine(
            self.camera_points,
            tuple((2.0 * u + 1.0, 3.0 * v + 2.0) for u, v in self.camera_points),
        )
        snapshot = base64.b64encode(b"png-placeholder").decode("ascii")
        common = {
            "coordinate_convention": "pixel origin top-left",
            "image_width": 640,
            "image_height": 480,
            "camera_points_uv": [list(point) for point in self.camera_points],
            "snapshot_png_base64": snapshot,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            schema_one = root / "legacy-v1.json"
            schema_one.write_text(
                json.dumps(
                    {
                        **common,
                        "schema_version": 1,
                        "robot_points_xy": [
                            [u + 5.0, v + 7.0] for u, v in self.camera_points
                        ],
                        "matrix_3x3": homography.tolist(),
                    }
                ),
                encoding="utf-8",
            )
            schema_two = root / "legacy-v2.json"
            schema_two.write_text(
                json.dumps(
                    {
                        **common,
                        "schema_version": 2,
                        "transform": "affine",
                        "robot_points_xy": [
                            [2.0 * u + 1.0, 3.0 * v + 2.0]
                            for u, v in self.camera_points
                        ],
                        "matrix": affine.tolist(),
                    }
                ),
                encoding="utf-8",
            )

            loaded_one = CalibrationService.load(schema_one)
            loaded_two = CalibrationService.load(schema_two)

        self.assertEqual(loaded_one.source_schema_version, 1)
        self.assertEqual(loaded_one.transform, "homography")
        self.assertEqual(loaded_two.source_schema_version, 2)
        self.assertEqual(loaded_two.transform, "affine")

    def test_geometry_mismatch_blocks_application(self) -> None:
        robot_points = tuple((u + 1.0, v + 2.0) for u, v in self.camera_points)
        record = CalibrationService.fit(
            "affine", self.camera_points, robot_points, (640, 480)
        )

        with self.assertRaises(CalibrationGeometryMismatchError):
            CalibrationService.apply(record, (20.0, 30.0), (1280, 720))
        np.testing.assert_allclose(
            CalibrationService.apply(record, (20.0, 30.0), (640, 480)),
            (21.0, 32.0),
            rtol=0.0,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
