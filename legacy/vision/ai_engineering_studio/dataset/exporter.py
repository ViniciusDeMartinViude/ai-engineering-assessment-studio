import sys
import re
import random
import io
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QSpinBox, QCheckBox, QProgressBar,
    QGroupBox
)

try:
    import yaml
except ImportError:
    yaml = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ObjectAnnotation:
    image_path: Path
    class_id: int
    polygon: np.ndarray
    source_kind: str


@dataclass
class PdfEntry:
    image_stream: io.BytesIO
    polygon: np.ndarray
    image_width: int
    image_height: int
    source_kind: str
    class_name: str


def sanitize_name(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text or "class"


def find_dataset_layout(selected: Path):
    selected = selected.resolve()

    candidates = [
        (selected / "test" / "images", selected / "test" / "labels", selected),
        (selected / "images", selected / "labels", selected),
    ]

    if selected.name.lower() == "images":
        candidates.append((selected, selected.parent / "labels", selected.parent))

    for images_dir, labels_dir, logical_root in candidates:
        if images_dir.is_dir() and labels_dir.is_dir():
            return images_dir, labels_dir, logical_root

    raise FileNotFoundError(
        "Could not find YOLO test folders.\n\n"
        "Select either:\n"
        "  • the dataset root containing test/images and test/labels, or\n"
        "  • the test folder containing images and labels."
    )


def load_class_names(selected: Path, logical_root: Path):
    if yaml is None:
        return {}

    search_dirs = []
    for d in [selected, logical_root, logical_root.parent, selected.parent]:
        if d not in search_dirs:
            search_dirs.append(d)

    yaml_files = []
    for d in search_dirs:
        if d.is_dir():
            yaml_files.extend([d / "data.yaml", d / "dataset.yaml"])

    for yfile in yaml_files:
        if not yfile.is_file():
            continue

        try:
            with yfile.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            names = data.get("names", {})
            if isinstance(names, list):
                return {i: str(name) for i, name in enumerate(names)}

            if isinstance(names, dict):
                result = {}
                for k, v in names.items():
                    try:
                        result[int(k)] = str(v)
                    except Exception:
                        pass
                return result
        except Exception:
            pass

    return {}


def yolo_box_to_polygon(values, width, height):
    x, y, w, h = values
    cx, cy = x * width, y * height
    bw, bh = w * width, h * height

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def yolo_segment_to_polygon(values, width, height):
    pts = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts


def scan_annotations(selected_folder: Path):
    images_dir, labels_dir, logical_root = find_dataset_layout(selected_folder)
    class_names = load_class_names(selected_folder, logical_root)

    images_by_stem = {}
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images_by_stem[p.stem] = p

    objects = []
    segmentation_count = 0
    bbox_count = 0
    skipped = 0

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = images_by_stem.get(label_path.stem)
        if image_path is None:
            skipped += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue

        height, width = image.shape[:2]

        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            skipped += 1
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                parts = line.split()
                class_id = int(float(parts[0]))
                values = [float(v) for v in parts[1:]]
            except Exception:
                skipped += 1
                continue

            if len(values) >= 6 and len(values) % 2 == 0:
                try:
                    polygon = yolo_segment_to_polygon(values, width, height)
                    if len(polygon) < 3:
                        skipped += 1
                        continue
                    objects.append(
                        ObjectAnnotation(
                            image_path=image_path,
                            class_id=class_id,
                            polygon=polygon,
                            source_kind="segment",
                        )
                    )
                    segmentation_count += 1
                except Exception:
                    skipped += 1

            elif len(values) == 4:
                polygon = yolo_box_to_polygon(values, width, height)
                objects.append(
                    ObjectAnnotation(
                        image_path=image_path,
                        class_id=class_id,
                        polygon=polygon,
                        source_kind="box",
                    )
                )
                bbox_count += 1
            else:
                skipped += 1

    counts = defaultdict(int)
    for obj in objects:
        counts[obj.class_id] += 1

    for class_id in counts:
        class_names.setdefault(class_id, f"class_{class_id}")

    return {
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "objects": objects,
        "counts": dict(counts),
        "class_names": class_names,
        "segmentation_count": segmentation_count,
        "bbox_count": bbox_count,
        "skipped": skipped,
    }


def extract_object(image, polygon, padding_percent):
    h, w = image.shape[:2]

    pts = np.round(polygon).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    x, y, bw, bh = cv2.boundingRect(pts)

    pad_x = int(round(bw * padding_percent / 100.0))
    pad_y = int(round(bh * padding_percent / 100.0))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    crop = image[y1:y2, x1:x2].copy()
    local_pts = pts.copy().astype(np.float32)
    local_pts[:, 0] -= x1
    local_pts[:, 1] -= y1

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(local_pts).astype(np.int32)], 255, lineType=cv2.LINE_AA)

    return crop, mask, local_pts


def make_png(crop, mask):
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    bgra[mask == 0, 0:3] = 0
    return bgra


def make_jpeg_white(crop, mask):
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    white = np.full_like(crop, 255, dtype=np.uint8)

    result = (
        crop.astype(np.float32) * alpha
        + white.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def encode_image_for_pdf(image_bgr):
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Could not encode image for PDF.")
    return io.BytesIO(encoded.tobytes())


def rotate_image_and_polygon_90cw(image, polygon):
    h, w = image.shape[:2]
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    new_polygon = polygon.copy().astype(np.float32)
    x = polygon[:, 0].copy()
    y = polygon[:, 1].copy()

    new_polygon[:, 0] = h - 1 - y
    new_polygon[:, 1] = x

    return rotated, new_polygon


def compute_fit_scale(img_w, img_h, box_w, box_h):
    return min(box_w / img_w, box_h / img_h)


def choose_best_orientation(image, polygon, box_w, box_h):
    h, w = image.shape[:2]

    scale_normal = compute_fit_scale(w, h, box_w, box_h)
    printed_area_normal = (w * scale_normal) * (h * scale_normal)

    rotated, rotated_polygon = rotate_image_and_polygon_90cw(image, polygon)
    rh, rw = rotated.shape[:2]

    scale_rotated = compute_fit_scale(rw, rh, box_w, box_h)
    printed_area_rotated = (rw * scale_rotated) * (rh * scale_rotated)

    if printed_area_rotated > printed_area_normal:
        return rotated, rotated_polygon

    return image, polygon


def draw_cut_marks(pdf, x, y, w, h, mark_len):
    pdf.setDash()
    pdf.setStrokeColor(colors.black)
    pdf.setLineWidth(0.4)

    pdf.line(x - mark_len, y, x, y)
    pdf.line(x, y - mark_len, x, y)

    pdf.line(x + w, y, x + w + mark_len, y)
    pdf.line(x + w, y - mark_len, x + w, y)

    pdf.line(x - mark_len, y + h, x, y + h)
    pdf.line(x, y + h, x, y + h + mark_len)

    pdf.line(x + w, y + h, x + w + mark_len, y + h)
    pdf.line(x + w, y + h, x + w, y + h + mark_len)


def draw_shape_cut_line(pdf, polygon, draw_x, draw_y, draw_w, draw_h, img_w, img_h, source_kind):
    if polygon is None or len(polygon) < 3:
        return

    scale = min(draw_w / img_w, draw_h / img_h)
    actual_w = img_w * scale
    actual_h = img_h * scale

    offset_x = draw_x + (draw_w - actual_w) / 2.0
    offset_y = draw_y + (draw_h - actual_h) / 2.0

    mapped = []
    for px, py in polygon:
        x_pdf = offset_x + px * scale
        y_pdf = offset_y + actual_h - py * scale
        mapped.append((x_pdf, y_pdf))

    if len(mapped) < 3:
        return

    pdf.saveState()
    pdf.setStrokeColor(colors.red if source_kind == "segment" else colors.darkgray)
    pdf.setLineWidth(0.8)
    pdf.setDash(3, 2)

    path = pdf.beginPath()
    path.moveTo(mapped[0][0], mapped[0][1])
    for x, y in mapped[1:]:
        path.lineTo(x, y)
    path.close()

    pdf.drawPath(path, stroke=1, fill=0)
    pdf.restoreState()


def draw_class_name_in_gap(pdf, class_name, center_x, baseline_y, max_width):
    pdf.saveState()
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 8)

    text = str(class_name).strip()
    while pdf.stringWidth(text, "Helvetica-Bold", 8) > max_width and len(text) > 3:
        text = text[:-2] + "…"

    pdf.drawCentredString(center_x, baseline_y, text)
    pdf.restoreState()


def create_a4_pdf(pdf_path, pdf_entries):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "reportlab is required for PDF export. Please install it with:\n"
            "pip install reportlab"
        )

    page_w, page_h = A4

    card_w = 6.0 * cm
    card_h = 4.0 * cm

    gap_x = 0.55 * cm
    gap_y = 0.70 * cm

    cols = int((page_w + gap_x) // (card_w + gap_x))
    rows = int((page_h + gap_y) // (card_h + gap_y))

    if cols <= 0 or rows <= 0:
        raise RuntimeError("Card size does not fit on an A4 page.")

    grid_w = cols * card_w + (cols - 1) * gap_x
    grid_h = rows * card_h + (rows - 1) * gap_y

    margin_x = (page_w - grid_w) / 2.0
    margin_y = (page_h - grid_h) / 2.0

    mark_len = 0.18 * cm
    inner_pad = 0.10 * cm

    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    pdf.setAuthor("ChatGPT / OpenAI")
    pdf.setTitle("Printable date cards - maximized fit")
    pdf.setLineWidth(0.4)

    cards_per_page = cols * rows

    for index, entry in enumerate(pdf_entries):
        pos = index % cards_per_page

        if pos == 0 and index > 0:
            pdf.showPage()
            pdf.setLineWidth(0.4)

        row = pos // cols
        col = pos % cols

        x = margin_x + col * (card_w + gap_x)
        y = page_h - margin_y - card_h - row * (card_h + gap_y)

        pdf.setDash()
        pdf.setStrokeColor(colors.black)
        pdf.rect(x, y, card_w, card_h)
        draw_cut_marks(pdf, x, y, card_w, card_h, mark_len)

        box_x = x + inner_pad
        box_y = y + inner_pad
        box_w = card_w - 2 * inner_pad
        box_h = card_h - 2 * inner_pad

        img_data = entry.image_stream.getvalue()
        arr = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Could not decode image for PDF placement.")

        image, polygon = choose_best_orientation(
            image,
            entry.polygon,
            box_w,
            box_h,
        )

        h, w = image.shape[:2]
        stream = encode_image_for_pdf(image)
        reader = ImageReader(stream)

        scale = min(box_w / w, box_h / h)
        actual_w = w * scale
        actual_h = h * scale
        draw_x = box_x + (box_w - actual_w) / 2.0
        draw_y = box_y + (box_h - actual_h) / 2.0

        pdf.drawImage(
            reader,
            draw_x,
            draw_y,
            width=actual_w,
            height=actual_h,
            preserveAspectRatio=True,
            anchor='c',
            mask='auto'
        )

        draw_shape_cut_line(
            pdf=pdf,
            polygon=polygon,
            draw_x=draw_x,
            draw_y=draw_y,
            draw_w=actual_w,
            draw_h=actual_h,
            img_w=w,
            img_h=h,
            source_kind=entry.source_kind,
        )

        # Put class name in the vertical gap below the card.
        label_y = y - min(gap_y * 0.55, 0.30 * cm)
        draw_class_name_in_gap(
            pdf,
            entry.class_name,
            x + card_w / 2.0,
            label_y,
            card_w
        )

    pdf.save()


class ExportWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, str)
    failed = Signal(str)

    def __init__(
        self,
        jobs,
        class_names,
        output_dir,
        export_png,
        export_jpeg,
        export_pdf,
        jpeg_quality,
        padding_percent,
        seed,
        pdf_filename,
    ):
        super().__init__()
        self.jobs = jobs
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.export_png = export_png
        self.export_jpeg = export_jpeg
        self.export_pdf = export_pdf
        self.jpeg_quality = jpeg_quality
        self.padding_percent = padding_percent
        self.seed = seed
        self.pdf_filename = pdf_filename

    @Slot()
    def run(self):
        try:
            rng = random.Random(self.seed)
            selected_objects = []

            by_class = defaultdict(list)
            for obj in self.jobs["objects"]:
                by_class[obj.class_id].append(obj)

            for class_id, requested in self.jobs["requested"].items():
                available = by_class.get(class_id, [])
                if requested > len(available):
                    raise ValueError(
                        f"{self.class_names.get(class_id, class_id)}: requested "
                        f"{requested}, but only {len(available)} objects are available."
                    )
                selected_objects.extend(rng.sample(available, requested))

            total = len(selected_objects)
            if total == 0:
                raise ValueError("No images were selected for export.")

            self.output_dir.mkdir(parents=True, exist_ok=True)

            per_class_index = defaultdict(int)
            written = 0
            pdf_entries = []

            for i, obj in enumerate(selected_objects, start=1):
                image = cv2.imread(str(obj.image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Could not read image: {obj.image_path}")

                crop, mask, local_pts = extract_object(
                    image,
                    obj.polygon,
                    self.padding_percent,
                )

                class_name = self.class_names.get(obj.class_id, f"class_{obj.class_id}")
                safe_class_name = sanitize_name(class_name)
                class_dir = self.output_dir / safe_class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                per_class_index[obj.class_id] += 1
                idx = per_class_index[obj.class_id]

                stem = sanitize_name(obj.image_path.stem)
                base_name = f"{safe_class_name}_{idx:03d}_{stem}"

                if self.export_png:
                    png = make_png(crop, mask)
                    out_png = class_dir / f"{base_name}.png"
                    if not cv2.imwrite(str(out_png), png):
                        raise RuntimeError(f"Could not write: {out_png}")

                white_img = None
                if self.export_jpeg or self.export_pdf:
                    white_img = make_jpeg_white(crop, mask)

                if self.export_jpeg:
                    out_jpg = class_dir / f"{base_name}.jpg"
                    if not cv2.imwrite(
                        str(out_jpg),
                        white_img,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    ):
                        raise RuntimeError(f"Could not write: {out_jpg}")

                if self.export_pdf:
                    pdf_entries.append(
                        PdfEntry(
                            image_stream=encode_image_for_pdf(white_img),
                            polygon=local_pts.copy(),
                            image_width=white_img.shape[1],
                            image_height=white_img.shape[0],
                            source_kind=obj.source_kind,
                            class_name=class_name,
                        )
                    )

                written += 1
                self.progress.emit(i, total, class_name)

            pdf_message = ""
            if self.export_pdf:
                self.progress.emit(total, total, "Creating A4 PDF...")
                pdf_path = self.output_dir / self.pdf_filename
                create_a4_pdf(pdf_path, pdf_entries)
                pdf_message = f"\nA4 PDF created: {pdf_path}"

            self.finished.emit(
                written,
                f"Exported {written} individual objects to:\n{self.output_dir}{pdf_message}",
            )

        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Date Image Extractor")
        self.resize(980, 740)

        self.scan_result = None
        self.thread = None
        self.worker = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        dataset_group = QGroupBox("1. Test dataset")
        dataset_layout = QHBoxLayout(dataset_group)

        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("Select dataset root or test folder...")
        dataset_browse = QPushButton("Browse...")
        dataset_scan = QPushButton("Scan dataset")

        dataset_layout.addWidget(self.dataset_edit, 1)
        dataset_layout.addWidget(dataset_browse)
        dataset_layout.addWidget(dataset_scan)
        main_layout.addWidget(dataset_group)

        classes_group = QGroupBox("2. Select date classes and quantities")
        classes_layout = QVBoxLayout(classes_group)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Class ID", "Date / Class", "Available objects", "Generate"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        classes_layout.addWidget(self.table)

        main_layout.addWidget(classes_group, 1)

        export_group = QGroupBox("3. Export options")
        export_layout = QFormLayout(export_group)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Choose an output folder...")
        output_browse = QPushButton("Browse...")
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(output_browse)

        format_row = QHBoxLayout()
        self.png_check = QCheckBox("PNG — transparent background")
        self.png_check.setChecked(True)
        self.jpg_check = QCheckBox("JPEG — white background")
        self.jpg_check.setChecked(True)
        self.pdf_check = QCheckBox("A4 PDF — max-fit 6x4 cm cards")
        self.pdf_check.setChecked(True)
        format_row.addWidget(self.png_check)
        format_row.addWidget(self.jpg_check)
        format_row.addWidget(self.pdf_check)
        format_row.addStretch(1)

        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 50)
        self.padding_spin.setValue(2)
        self.padding_spin.setSuffix(" %")

        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(50, 100)
        self.jpeg_quality_spin.setValue(95)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999999)
        self.seed_spin.setValue(42)

        self.pdf_name_edit = QLineEdit("printable_dates_A4_maxfit.pdf")

        export_layout.addRow("Output folder:", out_row)
        export_layout.addRow("Formats:", format_row)
        export_layout.addRow("Padding around fruit:", self.padding_spin)
        export_layout.addRow("JPEG quality:", self.jpeg_quality_spin)
        export_layout.addRow("Random seed:", self.seed_spin)
        export_layout.addRow("PDF filename:", self.pdf_name_edit)

        main_layout.addWidget(export_group)

        self.status_label = QLabel(
            "The PDF maximizes each date inside the full 6 x 4 cm card, "
            "preserves aspect ratio, and automatically rotates 90° when it gives a larger printed date. "
            "Class names are printed in the gap between cards."
        )
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.generate_button = QPushButton("Generate images")
        self.generate_button.setEnabled(False)

        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.generate_button)
        main_layout.addLayout(bottom)

        dataset_browse.clicked.connect(self.choose_dataset)
        dataset_scan.clicked.connect(self.scan_dataset)
        output_browse.clicked.connect(self.choose_output)
        self.generate_button.clicked.connect(self.generate_images)

        if not REPORTLAB_AVAILABLE:
            self.pdf_check.setChecked(False)
            self.pdf_check.setEnabled(False)
            self.pdf_name_edit.setEnabled(False)

    @Slot()
    def choose_dataset(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select dataset root or test folder"
        )
        if folder:
            self.dataset_edit.setText(folder)

    @Slot()
    def choose_output(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder"
        )
        if folder:
            self.output_edit.setText(folder)

    @Slot()
    def scan_dataset(self):
        folder = self.dataset_edit.text().strip()
        if not folder:
            QMessageBox.warning(self, "Dataset", "Select the dataset folder first.")
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            result = scan_annotations(Path(folder))
            self.scan_result = result
            self.populate_table(result)

            if not self.output_edit.text().strip():
                default_out = Path(folder) / "printable_dates"
                self.output_edit.setText(str(default_out))

            seg = result["segmentation_count"]
            box = result["bbox_count"]
            skipped = result["skipped"]

            msg = (
                f"Found {len(result['objects'])} annotated objects in "
                f"{len(result['counts'])} classes. "
                f"Segmentation objects: {seg}. Bounding-box objects: {box}."
            )
            if skipped:
                msg += f" Skipped entries/files: {skipped}."

            if box:
                msg += (
                    " Warning: bounding-box labels do not contain the exact fruit shape; "
                    "those objects will receive a rectangular cut line in the PDF."
                )

            if not REPORTLAB_AVAILABLE:
                msg += " PDF export is disabled because reportlab is not installed."

            self.status_label.setText(msg)
            self.generate_button.setEnabled(bool(result["objects"]))

        except Exception as exc:
            self.scan_result = None
            self.table.setRowCount(0)
            self.generate_button.setEnabled(False)
            QMessageBox.critical(self, "Scan error", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def populate_table(self, result):
        counts = result["counts"]
        names = result["class_names"]
        class_ids = sorted(counts.keys())

        self.table.setRowCount(len(class_ids))

        for row, class_id in enumerate(class_ids):
            item_id = QTableWidgetItem(str(class_id))
            item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)

            item_name = QTableWidgetItem(names.get(class_id, f"class_{class_id}"))
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)

            item_available = QTableWidgetItem(str(counts[class_id]))
            item_available.setTextAlignment(Qt.AlignCenter)
            item_available.setFlags(item_available.flags() & ~Qt.ItemIsEditable)

            qty = QSpinBox()
            qty.setRange(0, counts[class_id])
            qty.setValue(0)
            qty.setAlignment(Qt.AlignCenter)

            self.table.setItem(row, 0, item_id)
            self.table.setItem(row, 1, item_name)
            self.table.setItem(row, 2, item_available)
            self.table.setCellWidget(row, 3, qty)

        self.table.resizeColumnsToContents()

    def requested_quantities(self):
        requested = {}
        for row in range(self.table.rowCount()):
            class_id = int(self.table.item(row, 0).text())
            spin = self.table.cellWidget(row, 3)
            qty = spin.value()
            if qty > 0:
                requested[class_id] = qty
        return requested

    @Slot()
    def generate_images(self):
        if not self.scan_result:
            QMessageBox.warning(self, "Generate", "Scan the dataset first.")
            return

        requested = self.requested_quantities()
        if not requested:
            QMessageBox.warning(
                self,
                "Generate",
                "Choose at least one class and set its Generate quantity above zero."
            )
            return

        export_png = self.png_check.isChecked()
        export_jpeg = self.jpg_check.isChecked()
        export_pdf = self.pdf_check.isChecked()

        if not export_png and not export_jpeg and not export_pdf:
            QMessageBox.warning(
                self,
                "Generate",
                "Select PNG, JPEG, PDF, or any combination."
            )
            return

        output_dir = self.output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Generate", "Choose an output folder.")
            return

        pdf_filename = self.pdf_name_edit.text().strip() or "printable_dates_A4_maxfit.pdf"
        if not pdf_filename.lower().endswith(".pdf"):
            pdf_filename += ".pdf"

        self.generate_button.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("Generating images...")

        jobs = {
            "objects": self.scan_result["objects"],
            "requested": requested,
        }

        self.thread = QThread(self)
        self.worker = ExportWorker(
            jobs=jobs,
            class_names=self.scan_result["class_names"],
            output_dir=output_dir,
            export_png=export_png,
            export_jpeg=export_jpeg,
            export_pdf=export_pdf,
            jpeg_quality=self.jpeg_quality_spin.value(),
            padding_percent=self.padding_spin.value(),
            seed=self.seed_spin.value(),
            pdf_filename=pdf_filename,
        )

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @Slot(int, int, str)
    def on_progress(self, current, total, class_name):
        percent = int(current * 100 / total) if total else 0
        self.progress.setValue(percent)
        self.status_label.setText(f"Generating {current}/{total} — {class_name}")

    @Slot(int, str)
    def on_finished(self, count, message):
        self.progress.setValue(100)
        self.status_label.setText(message)
        self.generate_button.setEnabled(True)
        QMessageBox.information(
            self,
            "Finished",
            f"{message}\n\nObjects exported: {count}"
        )
        self.thread = None
        self.worker = None

    @Slot(str)
    def on_failed(self, message):
        self.status_label.setText("Export failed.")
        self.generate_button.setEnabled(True)
        QMessageBox.critical(self, "Export error", message)
        self.thread = None
        self.worker = None


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
