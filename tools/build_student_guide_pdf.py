"""Build the student visual guide PDF from the captured application snapshots."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "docs" / "assets" / "student-guide"
OUTPUT = ROOT / "output" / "pdf" / "student-visual-guide.pdf"
PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
NAVY = colors.HexColor("#17304d")
TEAL = colors.HexColor("#087f75")
INK = colors.HexColor("#172b4d")
MUTED = colors.HexColor("#61748c")
PANEL = colors.HexColor("#f3f7fb")
LINE = colors.HexColor("#c7d8df")
AMBER = colors.HexColor("#fff6e5")
RED = colors.HexColor("#fff0ed")


def image_for(filename: str, width: float = 7.1 * inch) -> Image:
    image = Image(str(SNAPSHOTS / filename), width=width, height=width * 900 / 1440)
    image.hAlign = "LEFT"
    return image


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def instruction_block(lines: list[str], styles: dict[str, ParagraphStyle]) -> list[object]:
    story: list[object] = []
    for index, line in enumerate(lines, start=1):
        story.append(paragraph(f"{index}. {line}", styles["body"]))
        story.append(Spacer(1, 0.08 * inch))
    return story


def page_footer(canvas: object, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(0.45 * inch, 0.35 * inch, PAGE_WIDTH - 0.45 * inch, 0.35 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.45 * inch, 0.18 * inch, "AI Engineering Assessment Studio | Student visual guide")
    canvas.drawRightString(PAGE_WIDTH - 0.45 * inch, 0.18 * inch, f"Page {document.page}")
    canvas.restoreState()


def build() -> Path:
    required = [SNAPSHOTS / f"{index:02d}-{name}.png" for index, name in (
        (1, "home"), (2, "dataset"), (3, "training"), (4, "vision"),
        (5, "calibration"), (6, "robot"), (7, "ide"), (8, "ai-assistant"),
        (9, "results"), (10, "submission"),
    )]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing guide snapshots: {missing}")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="guideTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=28, leading=32, textColor=NAVY, spaceAfter=8, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="sectionTitle", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=20, leading=24, textColor=NAVY, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="sectionIntro", parent=styles["Normal"], fontName="Helvetica",
        fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="body", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=13, textColor=INK, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="small", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=11, textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        name="callout", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=10, leading=14, textColor=TEAL,
    ))

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=PAGE_SIZE,
        rightMargin=0.45 * inch, leftMargin=0.45 * inch,
        topMargin=0.42 * inch, bottomMargin=0.52 * inch,
        title="AI Engineering Assessment Studio - Student Visual Guide",
        author="AI Engineering Assessment Studio",
    )
    story: list[object] = []

    story.append(paragraph("Student Visual Guide", styles["guideTitle"]))
    story.append(paragraph(
        "A practical walkthrough of the AI Engineering Assessment Studio desktop application.",
        styles["sectionIntro"],
    ))
    cover_table = Table([
        [
            image_for("01-home.png", width=6.25 * inch),
            [
                paragraph("The main workflow", styles["sectionTitle"]),
                paragraph("Prepare the dataset, train and select a model, configure Vision and Calibration, practice with Robot Studio, then use the IDE and AI Assistant for candidate-owned work.", styles["body"]),
                Spacer(1, 0.12 * inch),
                paragraph("Start in Dataset and move forward only when the previous result is ready.", styles["callout"]),
                Spacer(1, 0.12 * inch),
                paragraph("Windows launch", styles["sectionTitle"]),
                Paragraph("conda activate ai-assessment-studio<br/>python -m ai_assessment.app", styles["body"]),
            ],
        ],
    ], colWidths=[6.45 * inch, 4.0 * inch])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (1, 0), (1, 0), PANEL),
        ("BOX", (0, 0), (-1, -1), 0.7, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(paragraph("The teal buttons are primary actions, white outlined buttons are normal actions, red outlined buttons stop or caution, amber panels are hints, and green panels report status.", styles["body"]))

    sections = [
        ("Dataset Studio", "02-dataset.png", "Inspect the source dataset, select exactly four classes, and export a workspace-owned subset.", [
            "Enter the folder containing data.yaml and the train, validation, and test folders.",
            "Press Scan dataset and resolve diagnostics before exporting.",
            "Select exactly four classes and use Move up and Move down to set their order.",
            "Review image overlays, then press Export working subset.",
            "Use Print exports only when physical specimen cards are required.",
        ]),
        ("Training Studio", "03-training.png", "Run recorded local YOLO experiments without downloading weights or freezing the interface.", [
            "Choose the exported four-class subset and an existing local .pt base-weight file.",
            "Configure one experiment slot, including seed, device, and epochs.",
            "Press Start training and follow the live output.",
            "Use Cancel training when needed; cancelled and failed runs remain recorded but are not completed experiments.",
            "Repeat with four distinct configurations when the workflow requires four experiments.",
        ]),
        ("Vision Studio", "04-vision.png", "Preview the camera, draw the inference ROI, adjust processing, and inspect full-frame detections.", [
            "Set Camera index, optionally load a local .pt or .onnx model, and press Start camera.",
            "Drag on the raw image to draw an inference ROI; the processed image shows the YOLO input and annotations.",
            "Adjust confidence, brightness, contrast, and supported camera properties.",
            "Read full-frame boxes, centers, class IDs, confidence, and frame IDs in the detection table.",
            "Save camera profile or snapshot when the setup is ready. Camera clicks never move the robot.",
        ]),
        ("Calibration", "05-calibration.png", "Fit and verify the pixel-to-robot transform for the current camera geometry.", [
            "Start the camera or open a saved image, then freeze the frame.",
            "Select four corresponding camera and robot points; use the magnifier for precise clicks.",
            "Choose affine or homography and press Calculate matrix.",
            "Click the image to test a predicted robot X/Y, then review the matrix and convention.",
            "Press Save to workspace only after confirming the camera geometry is stable.",
        ]),
        ("Robot Studio", "06-robot.png", "Practice verified movement and supervised pick-and-place behavior in the local simulator.", [
            "Start with Local simulator, then press Check health and Read positions.",
            "Use high poses for travel and low poses only at a source or destination.",
            "Use suction controls deliberately and map each class to a destination.",
            "Run supervised trial to exercise the complete clearance route.",
            "Physical MaxArm mode starts disarmed and requires explicit operator enabling.",
        ]),
        ("IDE And Terminal", "07-ide.png", "Edit, save, run, and inspect candidate-owned Python code inside the workspace.", [
            "Double-click a file in the candidate workspace tree or press New Python file.",
            "Use Save or Save As; paths outside the workspace are rejected.",
            "Press Run Python for the saved file and watch stdout, stderr, and tracebacks in Run output.",
            "Use Stop only for the IDE process. It does not stop Training.",
            "Start the genuine pipe-based Terminal when interactive commands are needed.",
        ]),
        ("AI Assistant", "08-ai-assistant.png", "Ask the organizer gateway for help using context that you explicitly review and send.", [
            "Enter the gateway URL and session credential, then press Connect / refresh.",
            "Choose a context type, inspect the exact preview, and write a question.",
            "Press Send deliberately; the client never calls OpenAI directly.",
            "Use Send to IDE on a specific fenced code block in the response.",
            "Choose New file, Insert at cursor, or Replace selection. Generated code stays unsaved and is never run automatically.",
        ]),
        ("Results Studio", "09-results.png", "Compare candidate-visible validation results and select the production checkpoint.", [
            "Review recorded runs and the metrics that are actually available.",
            "Treat unavailable values as unavailable; do not infer hidden scores.",
            "Select a completed checkpoint with Select production model.",
            "The selection records the run identity and exact checkpoint hash for later Vision use.",
        ]),
        ("Submission", "10-submission.png", "Understand the boundary of the current local application.", [
            "The Submission page is reserved for the later organizer-controlled assessment flow.",
            "Local events, candidate-visible metrics, and simulator results are not a server receipt.",
            "Follow the organizer instructions when the official assessment integration is enabled.",
        ]),
    ]

    for title, image_name, intro, instructions in sections:
        story.append(PageBreak())
        story.append(paragraph(title, styles["sectionTitle"]))
        story.append(paragraph(intro, styles["sectionIntro"]))
        text_column = [paragraph("How to use it", styles["sectionTitle"])]
        text_column.extend(instruction_block(instructions, styles))
        layout = Table([[image_for(image_name), text_column]], colWidths=[7.35 * inch, 3.05 * inch])
        layout.setStyle(TableStyle([
            ("BACKGROUND", (1, 0), (1, 0), PANEL),
            ("BOX", (0, 0), (-1, -1), 0.7, LINE),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(layout)

    story.append(PageBreak())
    story.append(paragraph("Safety And Working Rules", styles["sectionTitle"]))
    story.append(paragraph("Keep these boundaries in mind while practicing:", styles["sectionIntro"]))
    rules = [
        "The original dataset is read-only. Exported subsets and generated artifacts belong under the candidate workspace.",
        "The AI gateway owns the provider key. Never put a provider key in the desktop application or candidate scripts.",
        "Vision and calibration never send robot commands automatically.",
        "Physical robot motion remains supervised and starts disarmed. Keep the physical emergency stop available.",
        "Candidate code runs locally and is not an operating-system security sandbox.",
        "The local event log is evidence capture, not an organizer server receipt.",
    ]
    story.append(Table([[paragraph("- " + rule, styles["body"])] for rule in rules], colWidths=[10.25 * inch], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AMBER),
        ("BOX", (0, 0), (-1, -1), 0.7, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ])))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
