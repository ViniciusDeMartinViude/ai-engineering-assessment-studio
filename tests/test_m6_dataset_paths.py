from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from ai_assessment.services.training_runner import _prepare_dataset_yaml, main


class DatasetPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="dataset path test ")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name) / "C014" / "dataset" / "subset-export"
        for split in ("train", "valid", "test"):
            (self.root / split / "images").mkdir(parents=True)
        self.source = self.root / "data.yaml"
        self.original = (
            "path: .\ntrain: train/images\nval: valid/images\n"
            "test: test/images\nnc: 4\nnames: [Ajwa, Galaxy, Medjool, Meneifi]\n"
        )
        self.source.write_text(self.original, encoding="utf-8")
        self.result_dir = Path(self.temp_dir.name) / "C014" / "results" / "training" / "run_01"

    def test_legacy_subset_gets_absolute_runtime_root_without_modifying_export(self) -> None:
        runtime = _prepare_dataset_yaml(self.source, self.result_dir)
        config = yaml.safe_load(runtime.read_text(encoding="utf-8"))
        self.assertEqual(Path(config["path"]), self.root.resolve())
        self.assertEqual(config["names"], ["Ajwa", "Galaxy", "Medjool", "Meneifi"])
        for split in ("train", "val", "test"):
            self.assertTrue((Path(config["path"]) / config[split]).is_dir())
        self.assertEqual(self.source.read_text(encoding="utf-8"), self.original)

    def test_new_subset_without_path_also_works(self) -> None:
        self.source.write_text(self.original.replace("path: .\n", ""), encoding="utf-8")
        runtime = _prepare_dataset_yaml(self.source, self.result_dir)
        self.assertEqual(Path(yaml.safe_load(runtime.read_text(encoding="utf-8"))["path"]), self.root.resolve())

    def test_train_and_val_receive_the_same_resolved_yaml(self) -> None:
        calls: list[tuple[str, Path]] = []

        class FakeYOLO:
            def __init__(self, weights: str) -> None:
                self.weights = weights

            def train(self, **kwargs):
                self._check("train", kwargs)

            def val(self, **kwargs):
                self._check("val", kwargs)
                return types.SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.5}, box=None)

            def _check(self, kind: str, kwargs: dict) -> None:
                resolved = Path(kwargs["data"])
                config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
                self.assert_compatible(config)
                calls.append((kind, resolved))

            @staticmethod
            def assert_compatible(config: dict) -> None:
                assert Path(config["path"]).is_absolute()
                for split in ("train", "val", "test"):
                    assert (Path(config["path"]) / config[split]).is_dir()

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO
        result_json = self.result_dir / "runner_result.json"
        args = ["--weights", "base.pt", "--data", str(self.source), "--project", str(self.root.parent),
                "--name", "run_01", "--result-json", str(result_json), "--epochs", "1",
                "--imgsz", "320", "--batch", "1", "--seed", "1", "--workers", "0", "--patience", "1"]
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            self.assertEqual(main(args), 0)
        self.assertEqual([kind for kind, _ in calls], ["train", "val"])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(json.loads(result_json.read_text(encoding="utf-8"))["metrics"]["metrics/mAP50(B)"], 0.5)
        self.assertEqual(self.source.read_text(encoding="utf-8"), self.original)


if __name__ == "__main__":
    unittest.main()
