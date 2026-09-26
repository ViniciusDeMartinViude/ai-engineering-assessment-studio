# Prototype inventory

| Source | Current evidence | Migration action |
|---|---|---|
| `MaxArm_API(2).md` | Stored API v0.5.1, positions/suction/health | Import canonical copy to `legacy/robot_api/`; contract-test live firmware |
| `MaxArm_Visual_Simulator.zip` | Stored archive | Unpack unchanged and audit endpoints/physics before reuse |
| `MaxArm_PySide6_Controller.zip` | Stored archive | Import unchanged; extract reusable HTTP client |
| `RobotPositions.xlsx` | Stored workbook | Verify values/units against `/positions` and live arm |
| `date_yolo_class_detector_http_high.py` | Stored Tkinter script | Preserve original; extract camera/inference/robot command logic |
| `date_dataset_image_extractor_maxfit_pdf.py` | Stored PySide6 script | Preserve; extract dataset annotation and printable export logic |
| `AI_Engineering_5_Hour_Assessment_Scheme.pdf` | Stored proposal | Use versioned criteria; pilot thresholds and stage times |
| Calibration application | Recent project discussion; current code not verified in starter | Obtain exact latest file and preserve unchanged |
| Dataset manager and original date dataset | Discussed, but exact current payload not verified in starter | Obtain version/export, license/provenance, manifest and hashes |
| MaxArm firmware with XYZ move | Requested after API v0.5.1; not verified here | Obtain current firmware and test protocol before exposing capability |

`legacy/` is empty by design in this starter. Record source filename, origin, date, SHA-256, runtime/dependencies and license when importing each item. Keep archives out of Git if they embed large datasets or generated models.
