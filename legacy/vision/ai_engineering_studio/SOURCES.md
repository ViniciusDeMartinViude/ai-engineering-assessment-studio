This project integrates artifacts produced earlier in the AI Engineering project:

- `MaxArm_Visual_Simulator.zip`: `maxarm_sim` physics, canvas, JSON positions and API. `model.py`/`api.py` add XYZ movement.
- `Camera_Robot_Calibrator.zip`: `calibration` UI and homography/affine mathematics. The integrated UI uses shared dashboard frames.
- `date_dataset_image_extractor_maxfit_pdf.py`: `dataset/exporter.py` for PNG/JPEG/A4 outputs.
- `date_yolo_class_detector_http_high.py` and `camera_settings.json`: camera setting semantics, pre-inference brightness/contrast, detection metadata and robot HTTP payloads.
- `MaxArm_API(2).md` and `RobotPositions.xlsx`: HTTP endpoints and ten calibrated position coordinates.

The original simulator's kinematics assumptions and reference URLs are described in the earlier `MaxArm_Visual_Simulator.zip` `SOURCES.md`.
