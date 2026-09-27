# AI Engineering Assessment Studio

AI Engineering date-fruit training and assessment workstation. M1 provides a PySide6 desktop shell with navigation across the planned application pages, a candidate workspace, append-only local event logs, and idempotent server-receipt acknowledgement storage. M2 adds an embedded four-point camera-to-robot calibration workspace. M3 adds Dataset Studio for read-only YOLO dataset inspection, four-class subset export, and printable object exports. M4 adds an embedded Vision Studio for camera preview, ROI processing, local YOLO inference, and full-frame detection centers.

The M1-M5 shell does not connect to an organizer server yet. Local events remain pending until a future organizer integration records a server acknowledgement. Calibration is a 2D planar mapping tool only, Vision Studio never sends robot commands or performs automatic sorting, Dataset Studio does not train YOLO, and Robot Studio does not sort live detections.

Read [Architecture v1](docs/ARCHITECTURE_V1.md) and [AGENTS.md](AGENTS.md) for the project boundaries and acceptance gates.

## Windows setup

Use Anaconda Prompt or another shell where `conda` is initialized. The project baseline targets Python 3.12:

```bat
cd /d C:\Projects\ai-engineering-assessment-studio
conda create -n ai-assessment-studio python=3.12 -y
conda activate ai-assessment-studio
python -m pip install -e ".[gui]"
```

If the repository is in a different folder, replace the `cd` path. The editable install makes the `ai_assessment` package available to the commands below.

## Run the shell and tests

Create or reopen the default `C014` workspace and launch the desktop shell:

```bat
python -m ai_assessment --workspace candidate_workspaces\C014
```

The candidate ID is only needed when creating a new workspace. When reopening an existing workspace, omit `--candidate-id` or provide the ID already stored in `session.json`. A different explicit ID is rejected.

Run the non-GUI workspace check:

```bat
python -m ai_assessment --headless-check --workspace candidate_workspaces\C014
```

Run the full M1-M4 tests:

```bat
python -m unittest discover -s tests -v
```

Events are stored in `candidate_workspaces\C014\logs\events.jsonl`; server receipts are stored in `candidate_workspaces\C014\logs\event_acks.jsonl`.

## M2 calibration

Open the **Calibration** page from the application sidebar. Start a camera or open a saved image, freeze the full-frame image, select four image points, enter their matching robot X/Y coordinates, and choose exactly one transform:

- `2 x 3 Affine` for a plane where perspective is negligible.
- `3 x 3 Perspective` for a plane viewed at an angle.

After calculating the matrix, click the frozen image to see a rounded integer robot X/Y prediction. The 5x cursor magnifier shows a pixel-level view. The matrix display is rounded to two decimals, while the workspace file retains full precision.

Use **Save to workspace** and **Load workspace calibration** to persist the active calibration at `candidate_workspaces\C014\calibration\calibration.json`. The service reads legacy schema versions 1 and 2. It records the point pairs, transform, image geometry, coordinate convention, camera metadata, optional ROI metadata, and an image snapshot. A saved calibration cannot be applied to an image with a different width or height.

The sample package has no live robot credentials, API keys, copyrighted dataset payload, or candidate data.

## M3 Dataset Studio

Open **Dataset** in the sidebar and choose a local YOLO export whose root contains `data.yaml` plus image and label folders for `train`, `valid` or `val`, and `test`. Scanning is read-only and runs away from the Qt interface. The page reports class IDs and names, image/object counts by split, missing or malformed files, and previews boxes and segmentation polygons.

Select exactly four classes and arrange their order. **Export working subset** writes an atomic, workspace-owned subset below `candidate_workspaces\C014\dataset\`. The selected original IDs are remapped to new IDs `0` through `3`; split membership and annotation coordinates are preserved. Images containing both selected and unselected classes are excluded and reported rather than silently losing objects. Test images remain in the test split. Each export includes `data.yaml`, a versioned manifest, source/version metadata when available, counts, exclusions, and SHA-256 hashes.

The **Print exports** tab adapts the preserved legacy utility. It can write transparent PNGs, white-background JPEGs, and A4 PDFs with 6 x 4 cm cards, cut marks, shape cut lines, fit rotation, and class names under `candidate_workspaces\C014\exports\print\`. Box-only annotations are clearly reported as rectangular cutouts. The original dataset and legacy source are never modified. Dataset selection and successful exports are recorded in the local event log; no organizer signature or lock is claimed in M3.

The GUI extra installs the required M3 dependencies, including PyYAML and ReportLab:

```bat
python -m pip install -e ".[gui]"
```

The full M1-M5 test suite remains:

```bat
python -m unittest discover -s tests -v
```

## M4 Vision Studio

Open **Vision** in the sidebar. Start the camera before selecting a model for a live raw preview; camera capture, model loading, and inference run outside the Qt interface. Select an existing local `.pt` or supported `.onnx` file with **Browse model...**. The app never downloads weights. A local `yolo26n.pt` is used as a convenience default only when it already exists in the current folder or the workspace `models` folder.

Vision Studio shows the untouched full-frame camera image beside the corrected ROI sent to YOLO. Drag an ROI on the raw image, tune confidence, software brightness and contrast, and request exposure, focus, or white-balance values. The camera panel shows requested values and driver read-back values because camera drivers may ignore or quantize settings. The actual inference device is shown after model load.

Detection rows contain class ID/name, confidence, full-frame bounding box, full-frame center pixels, frame ID, and optional robot X/Y. A saved M2 calibration is used only when its full-frame geometry matches the current frame; a mismatch is shown as a warning. A class ID is never treated as a robot position. **Save snapshot** writes the raw frame, annotated processed image, and metadata under `candidate_workspaces\C014\exports\vision\`. The active camera profile is persisted at `candidate_workspaces\C014\camera\vision_profile.json`.

Calibration and Vision share a camera ownership guard, so starting one module while the other has the webcam reports which module must be stopped first. Vision logs camera start/stop, model load, ROI changes, and snapshots without logging every frame.

## M5 Robot Studio

Open **Robot** in the sidebar. The page starts the preserved MaxArm simulator on a loopback port and uses the same HTTP contract as a physical MaxArm: `GET /health`, `/health/deep`, `/positions`, `/suction`, and `POST /command`. The simulator is also reachable by external Python clients while the application is open:

```bat
python -c "import requests; print(requests.post('http://127.0.0.1:PORT/command', json={'command':'suction','state':'on'}).json())"
```

Replace `PORT` with the port shown in the Robot page base URL. The default bind is local-only. **Expose simulator on LAN** is an explicit opt-in and restarts the simulator on `0.0.0.0`; use it only on a trusted network.

Robot Studio provides non-overlapping manual P1-P5 high/low moves, duration, suction, health, and positions controls. It displays the simulator current and target XYZ pose, animated progress, suction state, virtual objects, and command history. Simulator-only XYZ moves are bounds-checked. Physical MaxArm mode exposes only the documented position and suction contract, starts disarmed, requires a health/positions check and explicit session enable, and does not expose XYZ or software stop controls. The physical emergency-stop and power-isolation procedure remains outside the application.

The supervised trial uses the clearance route `source high -> source low -> suction on -> source high -> destination high -> destination low -> suction off -> destination high`. It serializes commands, waits for motion and release settling, handles busy/HTTP/timeout failures without blind retries, and reports an accepted-but-unconfirmed command as uncertain. Cancellation prevents later commands but never claims to stop a move already accepted by the controller. Four class names are loaded from the latest M3 subset manifest and mapped to destinations for trial selection; live Vision detections are not connected to robot movement in M5.

Robot commands, responses, state changes, errors, and trials are written to the candidate event log. Simulator state is available at `/sim/state`; the preserved prototype reset endpoints remain simulator-only. M5 tests include the external `requests.post(.../command)` path, position parity, busy handling, sequence order, timeout uncertainty, cancellation, and simulator-only XYZ capability. A physical MaxArm and supervised Windows hardware trial are still required before enabling real motion in production.
