"""Adapter for the unchanged legacy printable-date export implementation."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .dataset import (
    CancellationToken,
    DatasetError,
    DatasetScan,
    DatasetCancelled,
    _safe_workspace_path,
)


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class PrintExportOptions:
    export_png: bool = True
    export_jpeg: bool = True
    export_pdf: bool = True
    jpeg_quality: int = 95
    padding_percent: float = 2.0
    pdf_filename: str = "printable_dates_A4_maxfit.pdf"


@dataclass(frozen=True)
class PrintExportResult:
    output_dir: Path
    files: tuple[dict[str, Any], ...]
    box_only_objects: int


def export_print_assets(
    scan: DatasetScan,
    workspace_root: Path,
    selected_order: Sequence[int],
    options: PrintExportOptions,
    *,
    cancellation: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> PrintExportResult:
    """Run the legacy transparent/JPEG/A4 implementation in a staging folder."""

    if not (options.export_png or options.export_jpeg or options.export_pdf):
        raise DatasetError("Select at least one print export format.")
    cancellation = cancellation or CancellationToken()
    try:
        from legacy.dataset.date_dataset_image_extractor_maxfit_pdf import (
            PdfEntry,
            create_a4_pdf,
            encode_image_for_pdf,
            extract_object,
            make_jpeg_white,
            make_png,
        )
        import cv2
    except ImportError as error:
        raise DatasetError(
            "The legacy print exporter needs OpenCV, PySide6, NumPy, and reportlab for PDF output."
        ) from error

    workspace = workspace_root.resolve()
    export_root = _safe_workspace_path(workspace, workspace / "exports" / "print")
    export_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging = export_root / f".m3-print-staging-{token}"
    final = export_root / f"print-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{token[:8]}"
    selected = set(int(class_id) for class_id in selected_order)
    objects = [
        (record, annotation)
        for record in scan.records
        if not record.invalid
        for annotation in record.annotations
        if annotation.class_id in selected
    ]
    if not objects:
        raise DatasetError("No selected annotated objects are available for print export.")
    try:
        staging.mkdir(parents=True, exist_ok=False)
        pdf_entries = []
        box_only_objects = 0
        for index, (record, annotation) in enumerate(objects, start=1):
            if cancellation.is_cancelled():
                raise DatasetCancelled("Print export cancelled.")
            image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise DatasetError(f"Could not read source image: {record.image_path}")
            crop, mask, local_points = extract_object(
                image,
                annotation.polygon,
                options.padding_percent,
            )
            class_name = scan.names.get(annotation.class_id, f"class_{annotation.class_id}")
            safe_name = _sanitize_name(class_name)
            class_dir = staging / safe_name
            class_dir.mkdir(parents=True, exist_ok=True)
            base_name = f"{safe_name}_{index:04d}_{_sanitize_name(record.image_path.stem)}"
            if annotation.source_kind == "box":
                box_only_objects += 1
            if options.export_png:
                if not cv2.imwrite(str(class_dir / f"{base_name}.png"), make_png(crop, mask)):
                    raise DatasetError(f"Could not write PNG for {record.image_path.name}")
            white = None
            if options.export_jpeg or options.export_pdf:
                white = make_jpeg_white(crop, mask)
            if options.export_jpeg and white is not None:
                if not cv2.imwrite(
                    str(class_dir / f"{base_name}.jpg"),
                    white,
                    [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality],
                ):
                    raise DatasetError(f"Could not write JPEG for {record.image_path.name}")
            if options.export_pdf and white is not None:
                pdf_entries.append(
                    PdfEntry(
                        image_stream=encode_image_for_pdf(white),
                        polygon=local_points.copy(),
                        image_width=white.shape[1],
                        image_height=white.shape[0],
                        source_kind=annotation.source_kind,
                        class_name=class_name,
                    )
                )
            if progress:
                progress(index, len(objects), f"Printing {index}/{len(objects)}: {class_name}")
        if options.export_pdf:
            create_a4_pdf(staging / options.pdf_filename, pdf_entries)
        files = tuple(_hash_files(staging))
        manifest = {
            "schema_version": "m3.print.v1",
            "source_location": str(scan.root),
            "selected_class_ids": [int(class_id) for class_id in selected_order],
            "box_only_objects": box_only_objects,
            "files": list(files),
            "exported_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.move(str(staging), str(final))
        return PrintExportResult(final, tuple(_hash_files(final)), box_only_objects)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _sanitize_name(value: str) -> str:
    safe = "".join("_" if character in '<>:"/\\|?*' else character for character in value.strip())
    safe = "_".join(safe.split())
    return safe or "class"


def _hash_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        files.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    return files
