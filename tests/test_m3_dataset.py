from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from ai_assessment.services.dataset import DatasetService
from ai_assessment.services.print_exports import PrintExportOptions, export_print_assets


class DatasetServiceTests(unittest.TestCase):
    def _fixture(self) -> Path:
        root = Path(self.temp_dir.name) / "source"
        root.mkdir(parents=True)
        names = ["Ajwa", "Galaxy", "Medjool", "Meneifi", "Excluded"]
        (root / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "train": "train/images",
                    "val": "valid/images",
                    "test": "test/images",
                    "nc": 5,
                    "names": names,
                    "roboflow": {"version": 16},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        for split in ("train", "valid", "test"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)

        white = np.full((80, 100, 3), 255, dtype=np.uint8)
        self._write_image(root / "train/images/good.jpg", white)
        (root / "train/labels/good.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n"
            "1 0.2 0.2 0.1 0.1 0.3 0.2 0.2 0.3\n",
            encoding="utf-8",
        )
        self._write_image(root / "train/images/mixed.jpg", white)
        (root / "train/labels/mixed.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n4 0.2 0.2 0.1 0.1\n",
            encoding="utf-8",
        )
        self._write_image(root / "train/images/invalid.jpg", white)
        (root / "train/labels/invalid.txt").write_text(
            "9 0.5 0.5 0.4 0.4\n0 malformed\n",
            encoding="utf-8",
        )
        self._write_image(root / "valid/images/valid.jpg", white)
        (root / "valid/labels/valid.txt").write_text(
            "2 0.6 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
        self._write_image(root / "valid/images/missing-label.jpg", white)
        self._write_image(root / "test/images/test.jpg", white)
        (root / "test/labels/test.txt").write_text(
            "3 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.3\n",
            encoding="utf-8",
        )
        (root / "test/labels/orphan.txt").write_text(
            "3 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )
        return root

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise AssertionError(f"could not write fixture image {path}")

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_scan_supports_boxes_segments_and_diagnostics(self) -> None:
        scan = DatasetService.scan(self._fixture())

        self.assertEqual(scan.names[0], "Ajwa")
        self.assertEqual(scan.source_version, "16")
        self.assertEqual(scan.splits["train"].bbox_objects, 3)
        self.assertEqual(scan.splits["train"].segment_objects, 1)
        self.assertEqual(scan.splits["test"].segment_objects, 1)
        codes = {diagnostic.code for diagnostic in scan.diagnostics}
        self.assertIn("invalid_class_id", codes)
        self.assertIn("malformed_annotation", codes)
        self.assertIn("missing_label", codes)
        self.assertIn("missing_image", codes)
        self.assertEqual(scan.class_stats()[0]["train"], {"images": 2, "objects": 2})

    def test_export_remaps_classes_preserves_splits_and_manifest_hashes(self) -> None:
        source = self._fixture()
        scan = DatasetService.scan(source)
        workspace = Path(self.temp_dir.name) / "candidate"
        workspace.mkdir()

        result = DatasetService.export_subset(scan, workspace, [3, 1, 0, 2])

        self.assertTrue(result.manifest_path.is_file())
        self.assertEqual(result.manifest["status"], "complete")
        self.assertEqual(result.manifest["source_version"], "16")
        self.assertEqual(result.manifest["selected_order"], [3, 1, 0, 2])
        self.assertEqual(result.manifest["mixed_class_excluded_total"], 1)
        self.assertEqual(result.manifest["splits"]["train"]["images"], 1)
        self.assertEqual(result.manifest["splits"]["val"]["images"], 1)
        self.assertEqual(result.manifest["splits"]["test"]["images"], 1)
        self.assertTrue((result.output_dir / "train/images/good.jpg").is_file())
        self.assertFalse((result.output_dir / "train/images/mixed.jpg").exists())
        self.assertTrue((result.output_dir / "valid/images/valid.jpg").is_file())
        self.assertTrue((result.output_dir / "test/images/test.jpg").is_file())
        self.assertFalse((result.output_dir / "train/images/test.jpg").exists())

        train_label = (result.output_dir / "train/labels/good.txt").read_text(
            encoding="utf-8"
        )
        self.assertTrue(train_label.startswith("2 0.5 0.5 0.4 0.4"))
        self.assertIn("1 0.2 0.2 0.1 0.1 0.3 0.2 0.2 0.3", train_label)
        output_yaml = yaml.safe_load(
            (result.output_dir / "data.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(output_yaml["names"], ["Meneifi", "Galaxy", "Ajwa", "Medjool"])
        self.assertEqual(output_yaml["nc"], 4)
        self.assertNotIn("path", output_yaml)

        for file_entry in result.manifest["files"]:
            path = result.output_dir / file_entry["path"]
            self.assertTrue(path.is_file(), file_entry["path"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), file_entry["sha256"]
            )
        self.assertFalse(
            any(path.name.startswith(".m3-staging-") for path in result.output_dir.parent.iterdir())
        )

    def test_print_export_writes_transparent_and_white_assets(self) -> None:
        source = self._fixture()
        scan = DatasetService.scan(source)
        workspace = Path(self.temp_dir.name) / "candidate"
        workspace.mkdir()

        result = export_print_assets(
            scan,
            workspace,
            [0, 1, 2, 3],
            PrintExportOptions(export_pdf=False),
        )

        self.assertGreater(len(result.files), 1)
        self.assertEqual(result.box_only_objects, 3)
        self.assertTrue(any(entry["path"].endswith(".png") for entry in result.files))
        self.assertTrue(any(entry["path"].endswith(".jpg") for entry in result.files))
        manifest = (result.output_dir / "manifest.json").read_text(encoding="utf-8")
        self.assertIn('"schema_version": "m3.print.v1"', manifest)


if __name__ == "__main__":
    unittest.main()
