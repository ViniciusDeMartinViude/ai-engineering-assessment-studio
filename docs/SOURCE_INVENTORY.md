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
| `legacy/calibration/camera_robot_calibrator.py`, `calibration_math.py`, `requirements.txt`, `README.md` | Reference implementation supplied for M2; SHA-256 recorded 26 September 2026: `590B6E4EBA39CB433D2A920DCA84DE5A1700CE94CEA1B24EAA74A3548C3DD3C6`, `F1B033D0FDC5C4524E8477CFE2F2545EFA5E87DA0A72D82631634FB1483B1B9B`, `5D15BF16536A871E1AD0562D214BB1BFD8DF6F341AA8667A920329D1A24D63EB`, `4F93CBBD811AD7A758DEC87BCFA38DDC82F7875E8ABCF06E103CB456C367BFBC` respectively; PySide6, OpenCV and NumPy runtime | Preserve unchanged; extract math and schema compatibility into `src/ai_assessment/services/` and integrate the UI into the M1 shell |
| Dataset manager and original date dataset | Discussed, but exact current payload not verified in starter | Obtain version/export, license/provenance, manifest and hashes |
| MaxArm firmware with XYZ move | Requested after API v0.5.1; not verified here | Obtain current firmware and test protocol before exposing capability |

The supplied calibration reference is preserved unchanged under `legacy/calibration/`. Record source filename, origin, date, SHA-256, runtime/dependencies and license when importing each item. Keep archives out of Git if they embed large datasets or generated models.
