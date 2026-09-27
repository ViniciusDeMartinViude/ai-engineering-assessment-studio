from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

Point3D: TypeAlias = tuple[float, float, float]
Positions: TypeAlias = dict[str, dict[str, Point3D]]

PACKAGE_DIR = Path(__file__).resolve().parent


def load_positions(path: Path | None = None) -> Positions:
    source = path or PACKAGE_DIR / "positions.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    positions: Positions = {}
    for name, heights in raw.items():
        positions[name] = {}
        for height, coordinates in heights.items():
            if height not in {"high", "low"} or len(coordinates) != 3:
                raise ValueError(f"Invalid position entry: {name}/{height}")
            positions[name][height] = tuple(float(value) for value in coordinates)
    return positions


# The official documentation explains the MaxArm linkage and IK model but does
# not publish every mechanical dimension. These values reproduce its roughly
# 280 mm working reach and remain intentionally easy to tune.
SHOULDER_HEIGHT_MM = 72.0
UPPER_ARM_MM = 145.0
FOREARM_MM = 145.0
NOZZLE_LENGTH_MM = 36.0
BASE_RADIUS_MM = 42.0

OBJECT_RADIUS_MM = 13.0
OBJECT_HEIGHT_MM = 18.0
PICK_TOLERANCE_MM = 17.0
PLACE_RADIUS_MM = 46.0

