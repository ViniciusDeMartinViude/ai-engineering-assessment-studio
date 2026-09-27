"""Four-point planar camera-to-robot calibration (NumPy only)."""

from __future__ import annotations

import numpy as np


def _points(values, name):
    points = np.asarray(values, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"{name} must contain four finite (x, y) points.")
    span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    if span <= 1e-10:
        raise ValueError(f"{name} points are too close together.")
    for a in range(4):
        for b in range(a + 1, 4):
            for c in range(b + 1, 4):
                ab = points[b] - points[a]
                ac = points[c] - points[a]
                twice_area = abs(ab[0] * ac[1] - ab[1] * ac[0])
                if twice_area <= 1e-8 * span * span:
                    raise ValueError(f"{name} has three nearly collinear points.")
    return points


def _normalize(points):
    center = points.mean(axis=0)
    shifted = points - center
    rms_distance = np.sqrt(np.mean(np.sum(shifted * shifted, axis=1)))
    scale = np.sqrt(2.0) / rms_distance
    transform = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1],
    ])
    normalized = shifted * scale
    return normalized, transform


def fit_homography(camera_points, robot_points):
    """Return a 3x3 matrix mapping pixel (u, v) to planar robot (X, Y).

    Uses normalized DLT. The resulting homogeneous vector must be divided by
    its third component. Four point pairs give an exact fit, not an accuracy
    estimate at other positions.
    """
    src = _points(camera_points, "Camera")
    dst = _points(robot_points, "Robot")
    src_n, ts = _normalize(src)
    dst_n, td = _normalize(dst)
    rows = []
    for (u, v), (x, y) in zip(src_n, dst_n):
        rows.append([-u, -v, -1, 0, 0, 0, x * u, x * v, x])
        rows.append([0, 0, 0, -u, -v, -1, y * u, y * v, y])
    system = np.asarray(rows, dtype=np.float64)
    _, singular, vh = np.linalg.svd(system, full_matrices=True)
    if singular[-1] / singular[0] < 1e-10:
        raise ValueError("The four pairs cannot define a stable mapping. Spread the points out.")
    h = np.linalg.inv(td) @ vh[-1].reshape(3, 3) @ ts
    h /= h[2, 2] if abs(h[2, 2]) > 1e-12 else np.linalg.norm(h)
    if not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
        raise ValueError("The calculated mapping is singular. Check point order and coordinates.")
    for p, target in zip(src, dst):
        mapped = transform_point(h, *p)
        if np.linalg.norm(np.asarray(mapped) - target) > 1e-6 * max(1.0, np.ptp(dst, axis=0).max()):
            raise ValueError("The point pairs cannot be mapped consistently.")
    return h


def fit_affine(camera_points, robot_points):
    """Fit a 2x3 affine map to all four pairs using least squares."""
    src = _points(camera_points, "Camera")
    dst = _points(robot_points, "Robot")
    src_n, ts = _normalize(src)
    dst_n, td = _normalize(dst)
    design = np.column_stack((src_n, np.ones(4)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, dst_n, rcond=None)
    if rank != 3:
        raise ValueError("The camera points do not define a stable affine mapping.")
    normalized_h = np.vstack((coefficients.T, [0.0, 0.0, 1.0]))
    h = np.linalg.inv(td) @ normalized_h @ ts
    h /= h[2, 2]
    if not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
        raise ValueError("The calculated affine mapping is singular.")
    return h[:2, :]


def transform_point(h, u, v):
    h = np.asarray(h, dtype=np.float64)
    if h.shape == (2, 3):
        vector = h @ np.array([float(u), float(v), 1.0])
        if not np.isfinite(vector).all():
            raise ValueError("The transformed point is invalid.")
        return float(vector[0]), float(vector[1])
    if h.shape != (3, 3):
        raise ValueError("The transform must be a 2×3 or 3×3 matrix.")
    vector = h @ np.array([float(u), float(v), 1.0])
    if not np.isfinite(vector).all() or abs(vector[2]) <= 1e-12 * max(1.0, np.linalg.norm(vector)):
        raise ValueError("This pixel is at or near the transform's horizon.")
    return float(vector[0] / vector[2]), float(vector[1] / vector[2])
