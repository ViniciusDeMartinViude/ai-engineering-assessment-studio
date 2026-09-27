from __future__ import annotations

import math
from dataclasses import dataclass

from .config import (
    FOREARM_MM,
    NOZZLE_LENGTH_MM,
    Point3D,
    SHOULDER_HEIGHT_MM,
    UPPER_ARM_MM,
)


@dataclass(frozen=True)
class JointSolution:
    reachable: bool
    base_degrees: float
    shoulder_degrees: float
    elbow_degrees: float
    shoulder: Point3D
    elbow: Point3D
    wrist: Point3D


def _world_point(radial: float, z: float, yaw: float) -> Point3D:
    """Convert MaxArm radial plane coordinates to the documented XYZ frame."""
    return radial * math.sin(yaw), -radial * math.cos(yaw), z


def solve_ik(target: Point3D) -> JointSolution:
    """Solve a two-link projection of MaxArm's base + parallelogram linkage.

    Hiwonder describes the arm as a special three-link arm. The parallelogram
    mechanically preserves the nozzle orientation, leaving two planar angles
    plus the base yaw for XYZ placement. This function models that effective
    geometry while the API continues to use the exact calibrated XYZ target.
    """

    x, y, z = target
    wrist_z = z + NOZZLE_LENGTH_MM
    radial = math.hypot(x, y)
    dz = wrist_z - SHOULDER_HEIGHT_MM
    yaw = math.atan2(x, -y)
    distance_sq = radial * radial + dz * dz
    min_reach = abs(UPPER_ARM_MM - FOREARM_MM)
    max_reach = UPPER_ARM_MM + FOREARM_MM
    distance = math.sqrt(distance_sq)
    reachable = min_reach - 1e-6 <= distance <= max_reach + 1e-6

    denominator = 2.0 * UPPER_ARM_MM * FOREARM_MM
    cosine_elbow = (distance_sq - UPPER_ARM_MM**2 - FOREARM_MM**2) / denominator
    cosine_elbow = max(-1.0, min(1.0, cosine_elbow))
    elbow_angle = -math.acos(cosine_elbow)
    shoulder_angle = math.atan2(dz, radial) - math.atan2(
        FOREARM_MM * math.sin(elbow_angle),
        UPPER_ARM_MM + FOREARM_MM * math.cos(elbow_angle),
    )

    shoulder = (0.0, 0.0, SHOULDER_HEIGHT_MM)
    elbow_radial = UPPER_ARM_MM * math.cos(shoulder_angle)
    elbow_z = SHOULDER_HEIGHT_MM + UPPER_ARM_MM * math.sin(shoulder_angle)
    elbow = _world_point(elbow_radial, elbow_z, yaw)

    return JointSolution(
        reachable=reachable,
        base_degrees=math.degrees(yaw),
        shoulder_degrees=math.degrees(shoulder_angle),
        elbow_degrees=math.degrees(elbow_angle),
        shoulder=shoulder,
        elbow=elbow,
        wrist=(x, y, wrist_z),
    )


def smoothstep(progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return progress * progress * (3.0 - 2.0 * progress)


def interpolate(start: Point3D, end: Point3D, progress: float) -> Point3D:
    amount = smoothstep(progress)
    return tuple(a + (b - a) * amount for a, b in zip(start, end))  # type: ignore[return-value]
