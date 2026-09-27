# AI Engineering Studio

Windows desktop application that combines the camera/YOLO detector, MaxArm visual simulator, HTTP command terminal, camera-to-robot calibration, and the date image/print exporter created in this project.

## Install and start

Use Python 3.10 or 3.11 on Windows. In Anaconda Prompt or PowerShell:

```powershell
conda create -n ai-engineering-studio python=3.11 -y
conda activate ai-engineering-studio
python -m pip install -r requirements.txt
python main.py
```

You can subsequently double-click `run_windows.bat` **from an environment where the project dependencies are installed**. An NVIDIA GPU is optional for using the simulator; install a matching PyTorch/CUDA build in your environment if you want YOLO inference on your GPU. YOLO weights are not bundled. Select your own `best.pt` or `best.onnx` in the dashboard; with no model selected the camera preview still works. The app does not download weights automatically.

If port 8080 is occupied, start with `python main.py --sim-port 8081`.

## Dashboard

- **Raw camera** and **processed YOLO view** appear together. Drag on the raw image to set the inference ROI. Detection center coordinates remain in full-frame pixels.
- **Camera settings**: load an existing `camera_settings.json` to set exposure, autofocus, focus, white balance and temperature when opening the camera. Brightness/contrast controls apply in software before inference. Restart the camera after changing controls.
- **Simulator**: rotate with a drag, zoom with the mouse wheel. The local API starts at `http://127.0.0.1:8080`, with interactive documentation at `/docs`. Default binding is local to this PC.
- **Controls**: select simulator or physical robot, enter the robot base URL, use P1–P5 and high/low, choose a movement time, and operate suction. The simulator view represents the local simulator only; sending to a physical robot does not animate its actual pose.
- **Terminal**: type `help`, `move P2 high 1500`, `suction on`, `health`, `positions`, `state`, `xyz -3 -130 89 1500`, or a JSON object such as `{"command":"move","location":"P1","height":"high"}`. All command requests use `POST /command`. `state` is a simulator-only GET endpoint.

The simulator positions were transcribed from `RobotPositions.xlsx`:

| Location | Low XYZ (mm) | High XYZ (mm) |
| --- | --- | --- |
| P1 | (-3, -130, 59) | (-3, -130, 89) |
| P2 | (-124, -161, 59) | (-124, -161, 89) |
| P3 | (-124, -98, 59) | (-124, -98, 89) |
| P4 | (-124, -35, 59) | (-124, -35, 89) |
| P5 | (-124, 28, 59) | (-124, 28, 89) |

The local simulator additionally accepts `{"command":"move","x":-3,"y":-130,"z":89,"duration_ms":1500}`. This extends the older MaxArm API v0.5.1 document. Ensure the physical controller firmware implements the same XYZ payload before sending it to that device.

## Calibration and automatic sorting

Open **Camera calibration**, freeze a frame from the dashboard camera (or load an image), click four spaced reference points and enter the corresponding robot X/Y coordinates. Choose affine 2×3 or perspective 3×3, then calculate the matrix. Click a new point to display its robot X/Y in large numbers. You can move the **simulator** to the tested X/Y with a chosen Z and save/load a JSON calibration session. A 5× magnifier follows the cursor to make clicking corners easier.

The matrix estimates planar X/Y only. Set Z separately for the object and working plane. Keep the camera fixed and use the same resolution and crop convention as calibration. The application rejects using a calibration image whose dimensions differ from the current camera frame when performing live coordinate-based picking.

Automatic sorting is **off by default**. Select a class name (initially `Ajwa`) or numeric class ID and a destination P2–P5. When the selected class appears, a five-second countdown starts and inference pauses. The sequence picks at P1 by default and performs: pickup high, pickup low, suction on, pickup high, destination high, destination low, suction off, destination high. The delay between steps is configurable. To pick at the detected object center, enable **Pick at detected X/Y** after calibration and set the high/low Z values. This sends XYZ move commands. Automatic physical movement requires an additional explicit enable step in the UI. A stationary object needs to disappear for 0.75 s before it triggers again.

**Cancel pending sequence** prevents later commands but cannot stop a movement already accepted by the physical controller. Movement starts are acknowledged immediately; the sequence allows at least the requested movement duration before advancing.

## Dataset prints

The **Dataset prints** tab contains the earlier date exporter. Select a YOLO `test/images` and `test/labels` dataset, choose classes and counts, and export transparent PNGs, white-background JPEGs, and A4 PDF sheets with 6 × 4 cm date cutouts, cut marks and names. Bounding-box-only datasets yield rectangular approximations; segmentation polygons allow shape-following cuts.

## Project structure

- `main.py`: integrated PySide6 dashboard, camera worker, command terminal, sorting sequence and calibration wiring.
- `calibration/`: previously developed four-point calibration UI and numeric transformation functions.
- `maxarm_sim/`: previously developed MaxArm animation, kinematics, positions and FastAPI server, extended for XYZ.
- `dataset/exporter.py`: previously developed printable date image exporter.

This distribution includes Python source and configuration only. A webcam, physical MaxArm and trained date model were unavailable in the build environment; test camera controls, inference speed and physical motion with the actual devices before use.
