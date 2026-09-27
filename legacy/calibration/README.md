# Camera → Robot Coordinate Calibrator

An interactive PySide6 application for mapping image pixels to X/Y positions on a **flat robot working plane**. Use four corresponding point pairs to fit either a 3 × 3 perspective homography or a 2 × 3 affine transform.

## Install and launch

On Windows, open a terminal in this folder and run:

```powershell
conda create -n camera-robot python=3.11 -y
conda activate camera-robot
python -m pip install -r requirements.txt
python camera_robot_calibrator.py
```

You may also use a regular Python environment and `python -m pip install -r requirements.txt`.

## Calibrate

1. Fix the camera above the work surface. Keep camera height, focus, resolution, and the robot plane fixed while using the calibration.
2. Select the camera number with **− / +**, then start the camera and **Freeze frame**, or open a saved image. A first click on live video freezes it automatically. Stop the camera before switching camera numbers.
3. Place four markers spread across the usable work area. Click each marker and enter that same marker's robot X and Y in its row. Robot coordinates can be in millimetres or any consistent unit.
   Hover over the camera image to open the **5× magnifier**. Its crosshair and pixel coordinate help you click precise corners; it moves to the opposite side of the image to avoid covering the cursor.
4. Select **3 × 3 Perspective** when the camera views the work plane at an angle, or **2 × 3 Affine** when perspective is negligible. The two visible choices are mutually exclusive. The affine transform uses all four point pairs in a least squares fit.
5. Press **Calculate matrix**. Every subsequent image click shows the predicted robot X/Y in large, rounded integers. To edit a calibration point, select its numbered row, click again, and recalculate.
6. Save a JSON calibration file with the full-precision matrix, point pairs, and source image. Load it later to inspect or revise it. The app also opens files saved by the previous 3 × 3 version.

Keep the camera, image size, and work surface fixed while using the saved calibration. Try positions across the work area before commanding a real robot.

Camera clicks are recorded as whole pixel coordinates, and the robot coordinate fields accept one decimal place. Robot X/Y predictions are displayed as large whole numbers. Older saved calibrations retain their original full-precision point values and matrix until you edit and recalculate them.

The **matrix on screen is rounded to two decimal places for readability**; small but important coefficients may appear as `0.00`. Calculations and the saved JSON matrix always retain full floating-point precision.

## Applying the matrix elsewhere

With the new saved JSON, map any pixel coordinate `(u, v)` with:

```python
import json
import numpy as np

data = json.load(open("camera_robot_calibration.json", encoding="utf-8"))
H = np.asarray(data["matrix"], dtype=float)
p = H @ np.array([u, v, 1.0])
X, Y = (p[0] / p[2], p[1] / p[2]) if H.shape == (3, 3) else (p[0], p[1])
```

The division by `p[2]` is required for the 3 × 3 perspective mapping. The 2 × 3 affine mapping applies without a division. For YOLO detections, choose the object's actual contact point on the work surface if available; a bounding box centre may differ from that point. Use the same full-frame pixel coordinate system and orientation as during calibration. If inference is performed on a cropped or resized region, convert the detected coordinate back to the original full image first.

This maps **2D points on one plane**. It cannot infer the robot Z coordinate, compensate for object height, or map points outside the working plane reliably. Test predicted targets without the robot moving before enabling physical motion.
