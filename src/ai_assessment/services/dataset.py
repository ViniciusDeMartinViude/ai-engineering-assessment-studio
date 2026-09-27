"""Read-only YOLO dataset scanning and workspace subset export for M3."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_MANIFEST_VERSION = "m3.dataset.v1"
ProgressCallback = Callable[[int, int, str], None]


class DatasetError(ValueError):
    """Raised for invalid datasets or unsafe dataset exports."""


class DatasetCancelled(DatasetError):
    """Raised when a scan or export is cancelled by the candidate."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class DatasetDiagnostic:
    severity: str
    code: str
    split: str | None
    path: str
    message: str

    def display(self) -> str:
        scope = f"[{self.split}] " if self.split else ""
        return f"{self.severity.upper()} {scope}{self.code}: {self.message} ({self.path})"


@dataclass(frozen=True)
class Annotation:
    class_id: int
    source_kind: str
    polygon: tuple[tuple[float, float], ...]
    raw_value_tokens: tuple[str, ...]


@dataclass
class ImageRecord:
    split: str
    image_path: Path
    label_path: Path | None
    width: int | None
    height: int | None
    annotations: list[Annotation] = field(default_factory=list)
    invalid: bool = False
    missing_label: bool = False


@dataclass
class SplitSummary:
    key: str
    directory_name: str
    image_dir: Path | None
    label_dir: Path | None
    records: list[ImageRecord] = field(default_factory=list)
    missing_image_labels: int = 0
    missing_label_images: int = 0
    invalid_annotations: int = 0
    bbox_objects: int = 0
    segment_objects: int = 0

    @property
    def image_count(self) -> int:
        return len(self.records)

    @property
    def annotated_image_count(self) -> int:
        return sum(bool(record.annotations) and not record.invalid for record in self.records)

    @property
    def object_count(self) -> int:
        return sum(len(record.annotations) for record in self.records)


@dataclass
class DatasetScan:
    root: Path
    yaml_path: Path
    names: dict[int, str]
    splits: dict[str, SplitSummary]
    diagnostics: list[DatasetDiagnostic]
    source_version: str | None
    yaml_data: dict[str, Any]

    @property
    def records(self) -> list[ImageRecord]:
        return [record for split in self.splits.values() for record in split.records]

    @property
    def class_ids(self) -> list[int]:
        return sorted(self.names)

    @property
    def bbox_count(self) -> int:
        return sum(split.bbox_objects for split in self.splits.values())

    @property
    def segment_count(self) -> int:
        return sum(split.segment_objects for split in self.splits.values())

    def class_stats(self) -> dict[int, dict[str, dict[str, int]]]:
        stats: dict[int, dict[str, dict[str, int]]] = {
            class_id: {
                split_name: {"images": 0, "objects": 0}
                for split_name in self.splits
            }
            for class_id in self.names
        }
        for split_name, split in self.splits.items():
            for record in split.records:
                per_class: dict[int, int] = {}
                for annotation in record.annotations:
                    per_class[annotation.class_id] = per_class.get(annotation.class_id, 0) + 1
                for class_id, object_count in per_class.items():
                    stats.setdefault(class_id, {})
                    stats[class_id].setdefault(split_name, {"images": 0, "objects": 0})
                    stats[class_id][split_name]["images"] += 1
                    stats[class_id][split_name]["objects"] += object_count
        return stats


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_workspace_path(workspace_root: Path, target: Path) -> Path:
    root = workspace_root.resolve()
    resolved = target.resolve()
    if resolved != root and root not in resolved.parents:
        raise DatasetError(f"Path is outside candidate workspace: {resolved}")
    return resolved


def _ensure_not_source_output(source_root: Path, output_root: Path) -> None:
    source = source_root.resolve()
    output = output_root.resolve()
    if output == source or output in source.parents:
        raise DatasetError(
            "The export directory is inside the source dataset. Choose a source "
            "outside the candidate workspace."
        )


def _resolve_yaml_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        result: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return result
    return {}


def _candidate_paths(root: Path, yaml_path: Path, value: Any) -> list[Path]:
    raw_values = value if isinstance(value, list) else [value]
    candidates: list[Path] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            continue
        raw_path = Path(raw_value)
        candidates.append((yaml_path.parent / raw_path).resolve())
        stripped = [part for part in raw_path.parts if part not in ("..", ".")]
        if stripped:
            candidates.append((root.joinpath(*stripped)).resolve())
        candidates.append((root / raw_path.name).resolve())
    return candidates


def _resolve_split_dirs(root: Path, yaml_path: Path, yaml_data: dict[str, Any], key: str) -> tuple[Path | None, Path | None, str]:
    yaml_key = key
    value = yaml_data.get(key)
    if value is None and key == "val":
        value = yaml_data.get("valid")
        yaml_key = "valid" if value is not None else "val"
    candidates = _candidate_paths(root, yaml_path, value) if value is not None else []
    candidates.extend(
        [
            (root / key / "images").resolve(),
            (root / ("valid" if key == "val" else key) / "images").resolve(),
        ]
    )
    for candidate in candidates:
        images_dir = candidate
        if images_dir.name.lower() != "images" and (candidate / "images").is_dir():
            images_dir = candidate / "images"
        labels_dir = images_dir.parent / "labels"
        if images_dir.is_dir() and labels_dir.is_dir():
            directory_name = images_dir.parent.name
            return images_dir, labels_dir, directory_name
    return None, None, yaml_key


def _normalised_polygon(values: Sequence[float], width: int, height: int, source_kind: str) -> tuple[tuple[float, float], ...]:
    if source_kind == "box":
        cx, cy, box_width, box_height = values
        if box_width <= 0 or box_height <= 0:
            raise DatasetError("Bounding-box width and height must be positive.")
        x1 = (cx - box_width / 2.0) * width
        y1 = (cy - box_height / 2.0) * height
        x2 = (cx + box_width / 2.0) * width
        y2 = (cy + box_height / 2.0) * height
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        raise DatasetError("Segmentation polygons need at least three points.")
    return tuple((float(x * width), float(y * height)) for x, y in points)


def _parse_label_file(
    label_path: Path,
    image_path: Path,
    width: int,
    height: int,
    class_names: dict[int, str],
    split: str,
    diagnostics: list[DatasetDiagnostic],
) -> tuple[list[Annotation], int, int, bool]:
    annotations: list[Annotation] = []
    invalid_count = 0
    bbox_count = 0
    segment_count = 0
    invalid = False
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        diagnostics.append(DatasetDiagnostic("error", "unreadable_label", split, str(label_path), str(error)))
        return [], 1, 0, True
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except (IndexError, TypeError, ValueError):
            invalid = True
            invalid_count += 1
            diagnostics.append(DatasetDiagnostic("error", "malformed_annotation", split, f"{label_path}:{line_number}", "Class ID and coordinates must be numeric."))
            continue
        if class_id not in class_names:
            invalid = True
            invalid_count += 1
            diagnostics.append(DatasetDiagnostic("error", "invalid_class_id", split, f"{label_path}:{line_number}", f"Class ID {class_id} is not declared in data.yaml."))
            continue
        if not values or not np.isfinite(values).all() or any(value < 0.0 or value > 1.0 for value in values):
            invalid = True
            invalid_count += 1
            diagnostics.append(DatasetDiagnostic("error", "invalid_coordinates", split, f"{label_path}:{line_number}", "YOLO coordinates must be finite normalized values in [0, 1]."))
            continue
        if len(values) == 4:
            source_kind = "box"
            bbox_count += 1
        elif len(values) >= 6 and len(values) % 2 == 0:
            source_kind = "segment"
            segment_count += 1
        else:
            invalid = True
            invalid_count += 1
            diagnostics.append(DatasetDiagnostic("error", "malformed_annotation", split, f"{label_path}:{line_number}", "Expected four box values or an even polygon coordinate list with at least three points."))
            continue
        try:
            polygon = _normalised_polygon(values, width, height, source_kind)
        except DatasetError as error:
            invalid = True
            invalid_count += 1
            diagnostics.append(DatasetDiagnostic("error", "malformed_annotation", split, f"{label_path}:{line_number}", str(error)))
            continue
        annotations.append(
            Annotation(
                class_id=class_id,
                source_kind=source_kind,
                polygon=polygon,
                raw_value_tokens=tuple(parts[1:]),
            )
        )
    return annotations, invalid_count, bbox_count, invalid


class DatasetService:
    """Read-only source scanner and atomic candidate subset exporter."""

    @staticmethod
    def scan(
        root: Path,
        *,
        cancellation: CancellationToken | None = None,
        progress: ProgressCallback | None = None,
    ) -> DatasetScan:
        cancellation = cancellation or CancellationToken()
        dataset_root = root.resolve()
        yaml_path = dataset_root / "data.yaml"
        if not yaml_path.is_file():
            raise DatasetError(f"Dataset root must contain data.yaml: {dataset_root}")
        try:
            yaml_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            raise DatasetError(f"Could not read data.yaml: {error}") from error
        if not isinstance(yaml_data, dict):
            raise DatasetError("data.yaml must contain a mapping.")
        names = _resolve_yaml_names(yaml_data.get("names"))
        diagnostics: list[DatasetDiagnostic] = []
        declared_count = yaml_data.get("nc")
        if declared_count is not None:
            try:
                if int(declared_count) != len(names):
                    diagnostics.append(DatasetDiagnostic("warning", "class_count_mismatch", None, str(yaml_path), f"data.yaml declares nc={declared_count}, but names contains {len(names)} classes."))
            except (TypeError, ValueError):
                diagnostics.append(DatasetDiagnostic("warning", "invalid_class_count", None, str(yaml_path), "data.yaml nc is not an integer."))
        splits: dict[str, SplitSummary] = {}
        for split_index, split_key in enumerate(("train", "val", "test"), start=1):
            if cancellation.is_cancelled():
                raise DatasetCancelled("Dataset scan cancelled.")
            images_dir, labels_dir, directory_name = _resolve_split_dirs(dataset_root, yaml_path, yaml_data, split_key)
            summary = SplitSummary(split_key, directory_name, images_dir, labels_dir)
            splits[split_key] = summary
            if images_dir is None or labels_dir is None:
                diagnostics.append(DatasetDiagnostic("error", "missing_split", split_key, str(dataset_root), f"Could not find {split_key}/images and matching labels folders."))
                if progress:
                    progress(split_index, 3, f"Missing {split_key} split")
                continue
            image_paths = sorted(
                path for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            images_by_stem = {path.stem: path for path in image_paths}
            label_paths = sorted(labels_dir.glob("*.txt"))
            for label_path in label_paths:
                if label_path.stem not in images_by_stem:
                    summary.missing_image_labels += 1
                    diagnostics.append(DatasetDiagnostic("error", "missing_image", split_key, str(label_path), "Label has no matching image."))
            total = max(1, len(image_paths))
            for index, image_path in enumerate(image_paths, start=1):
                if cancellation.is_cancelled():
                    raise DatasetCancelled("Dataset scan cancelled.")
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                label_path = labels_dir / f"{image_path.stem}.txt"
                if image is None:
                    diagnostics.append(DatasetDiagnostic("error", "unreadable_image", split_key, str(image_path), "OpenCV could not read this image."))
                    summary.records.append(ImageRecord(split_key, image_path, label_path if label_path.is_file() else None, None, None, invalid=True))
                    continue
                height, width = image.shape[:2]
                if not label_path.is_file():
                    summary.missing_label_images += 1
                    diagnostics.append(DatasetDiagnostic("error", "missing_label", split_key, str(image_path), "Image has no matching label file."))
                    summary.records.append(ImageRecord(split_key, image_path, None, width, height, missing_label=True))
                    continue
                annotations, invalid_count, bbox_count, invalid = _parse_label_file(
                    label_path, image_path, width, height, names, split_key, diagnostics
                )
                summary.invalid_annotations += invalid_count
                summary.bbox_objects += bbox_count
                summary.segment_objects += len([annotation for annotation in annotations if annotation.source_kind == "segment"])
                summary.records.append(ImageRecord(split_key, image_path, label_path, width, height, annotations, invalid=invalid))
                if progress:
                    progress(split_index, 3, f"Scanning {split_key}: {index}/{total}")
            if progress:
                progress(split_index, 3, f"Scanned {split_key}")
        source_version = _source_version(yaml_data)
        return DatasetScan(dataset_root, yaml_path, names, splits, diagnostics, source_version, yaml_data)

    @staticmethod
    def export_subset(
        scan: DatasetScan,
        workspace_root: Path,
        selected_order: Sequence[int],
        *,
        cancellation: CancellationToken | None = None,
        progress: ProgressCallback | None = None,
    ) -> ExportResult:
        cancellation = cancellation or CancellationToken()
        selected = [int(class_id) for class_id in selected_order]
        if len(selected) != 4 or len(set(selected)) != 4:
            raise DatasetError("Select exactly four distinct classes in the desired order.")
        if any(class_id not in scan.names for class_id in selected):
            raise DatasetError("Every selected class must exist in data.yaml.")
        workspace = workspace_root.resolve()
        output_root = _safe_workspace_path(workspace, workspace / "dataset")
        _ensure_not_source_output(scan.root, output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        staging = output_root / f".m3-staging-{token}"
        final = output_root / f"subset-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{token[:8]}"
        mapping = {original: new for new, original in enumerate(selected)}
        exclusions: dict[str, dict[str, int]] = {}
        exported_counts: dict[str, dict[str, int]] = {}
        total_records = sum(len(split.records) for split in scan.splits.values())
        completed = 0
        try:
            staging.mkdir(parents=True, exist_ok=False)
            for split_key, split in scan.splits.items():
                if cancellation.is_cancelled():
                    raise DatasetCancelled("Dataset export cancelled.")
                image_output = staging / split.directory_name / "images"
                label_output = staging / split.directory_name / "labels"
                image_output.mkdir(parents=True, exist_ok=True)
                label_output.mkdir(parents=True, exist_ok=True)
                split_exclusions = {
                    "mixed_class_images": 0,
                    "non_selected_images": 0,
                    "invalid_images": 0,
                    "missing_label_images": 0,
                    "missing_image_labels": 0,
                }
                split_exported = {"images": 0, "objects": 0}
                split_exclusions["missing_image_labels"] = split.missing_image_labels
                for record in split.records:
                    completed += 1
                    if progress:
                        progress(completed, max(1, total_records), f"Exporting {split_key}: {record.image_path.name}")
                    if cancellation.is_cancelled():
                        raise DatasetCancelled("Dataset export cancelled.")
                    if record.missing_label:
                        split_exclusions["missing_label_images"] += 1
                        continue
                    class_ids = {annotation.class_id for annotation in record.annotations}
                    if record.invalid:
                        split_exclusions["invalid_images"] += 1
                        continue
                    if not class_ids or not class_ids.intersection(mapping):
                        split_exclusions["non_selected_images"] += 1
                        continue
                    if not class_ids.issubset(mapping):
                        split_exclusions["mixed_class_images"] += 1
                        continue
                    destination_image = image_output / record.image_path.name
                    destination_label = label_output / f"{record.image_path.stem}.txt"
                    shutil.copy2(record.image_path, destination_image)
                    destination_label.write_text(
                        "\n".join(
                            f"{mapping[annotation.class_id]} {' '.join(annotation.raw_value_tokens)}"
                            for annotation in record.annotations
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    split_exported["images"] += 1
                    split_exported["objects"] += len(record.annotations)
                exclusions[split_key] = split_exclusions
                exported_counts[split_key] = split_exported
            output_yaml = {
                "path": ".",
                "train": "train/images",
                "val": f"{scan.splits['val'].directory_name}/images",
                "test": f"{scan.splits['test'].directory_name}/images",
                "nc": 4,
                "names": [scan.names[class_id] for class_id in selected],
            }
            (staging / "data.yaml").write_text(
                yaml.safe_dump(output_yaml, sort_keys=False), encoding="utf-8"
            )
            file_hashes = _hash_files(staging)
            manifest = {
                "schema_version": DATASET_MANIFEST_VERSION,
                "status": "complete",
                "source_location": str(scan.root),
                "source_yaml": str(scan.yaml_path),
                "source_version": scan.source_version,
                "source_read_only": True,
                "selected_class_mapping": {
                    str(original): {
                        "new_id": mapping[original],
                        "name": scan.names[original],
                    }
                    for original in selected
                },
                "selected_order": selected,
                "splits": exported_counts,
                "exclusions": exclusions,
                "mixed_class_excluded_total": sum(
                    split_data["mixed_class_images"] for split_data in exclusions.values()
                ),
                "exported_at": _utc_now_iso(),
                "files": file_hashes,
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(staging, final)
            return ExportResult(final, final / "manifest.json", manifest)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise


def _source_version(yaml_data: dict[str, Any]) -> str | None:
    roboflow = yaml_data.get("roboflow")
    if isinstance(roboflow, dict) and roboflow.get("version") is not None:
        return str(roboflow["version"])
    for key in ("version", "dataset_version"):
        if yaml_data.get(key) is not None:
            return str(yaml_data[key])
    return None


def _hash_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    return files
