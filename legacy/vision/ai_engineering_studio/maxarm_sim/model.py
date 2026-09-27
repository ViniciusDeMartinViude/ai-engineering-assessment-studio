from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from .config import (
    OBJECT_HEIGHT_MM,
    PICK_TOLERANCE_MM,
    PLACE_RADIUS_MM,
    Point3D,
    Positions,
)
from .kinematics import interpolate, solve_ik


@dataclass
class SimObject:
    object_id: str
    color: str
    position: Point3D
    pickup_point: Point3D
    attached: bool = False
    placed_at: str | None = None


class RobotBusyError(RuntimeError):
    pass


class RobotModel:
    API_VERSION = "0.5.1"

    def __init__(self, positions: Positions):
        self.positions = positions
        self.started_at = time.monotonic()
        self._lock = threading.RLock()
        self._current_xyz: Point3D = positions["P1"]["high"]
        self._target_xyz: Point3D = self._current_xyz
        self._location = "P1"
        self._height = "high"
        self._moving = False
        self._motion_progress = 1.0
        self._duration_ms = 0
        self._suction_on = False
        self._release_active = False
        self._attached_object_id: str | None = None
        self._request_count = 0
        self._error_count = 0
        self._last_command: dict[str, Any] | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=250)
        self._objects: dict[str, SimObject] = {}
        self.reset_objects()
        self.log("Simulator ready")

    def log(self, message: str, level: str = "info") -> None:
        with self._lock:
            self._events.append(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "level": level,
                    "message": message,
                }
            )

    def count_request(self) -> None:
        with self._lock:
            self._request_count += 1

    def count_error(self, message: str) -> None:
        with self._lock:
            self._error_count += 1
        self.log(message, "error")

    def reset_objects(self) -> None:
        colors = ["#C97A45", "#A24B39", "#D1A12B", "#8C5A3C", "#B56D3C"]
        objects: dict[str, SimObject] = {}
        for index, (name, heights) in enumerate(sorted(self.positions.items())):
            x, y, z = heights["low"]
            center = (x, y, z - OBJECT_HEIGHT_MM / 2.0)
            objects[f"date-{index + 1}"] = SimObject(
                object_id=f"date-{index + 1}",
                color=colors[index % len(colors)],
                position=center,
                pickup_point=(x, y, z),
                placed_at=name,
            )
        with self._lock:
            self._objects = objects
            self._attached_object_id = None

    def reset(self) -> None:
        with self._lock:
            if self._moving:
                raise RobotBusyError("Robot is busy")
            self._current_xyz = self.positions["P1"]["high"]
            self._target_xyz = self._current_xyz
            self._location = "P1"
            self._height = "high"
            self._motion_progress = 1.0
            self._duration_ms = 0
            self._suction_on = False
            self._release_active = False
            self._last_command = None
        self.reset_objects()
        self.log("Simulation reset")

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._moving

    def start_move(self, location: str, height: str, duration_ms: int) -> Point3D:
        target = self.positions[location][height]
        solution = solve_ik(target)
        if not solution.reachable:
            raise ValueError("Target is outside the configured simulator workspace")

        with self._lock:
            if self._moving:
                raise RobotBusyError("Robot is busy")
            start = self._current_xyz
            self._target_xyz = target
            self._location = location
            self._height = height
            self._moving = True
            self._motion_progress = 0.0
            self._duration_ms = duration_ms
            self._last_command = {
                "command": "move",
                "location": location,
                "height": height,
                "duration_ms": duration_ms,
            }

        self.log(f"Move accepted: {location}/{height} ({duration_ms} ms)")
        thread = threading.Thread(
            target=self._animate_move,
            args=(start, target, duration_ms),
            name="maxarm-motion",
            daemon=True,
        )
        thread.start()
        return target

    def start_move_xyz(self, target: Point3D, duration_ms: int) -> Point3D:
        """Move to an explicit, reachable XYZ point in millimetres."""
        solution = solve_ik(target)
        if not solution.reachable:
            raise ValueError("Target is outside the configured simulator workspace")
        with self._lock:
            if self._moving:
                raise RobotBusyError("Robot is busy")
            start = self._current_xyz
            self._target_xyz = target
            self._location = "custom"
            self._height = "custom"
            self._moving = True
            self._motion_progress = 0.0
            self._duration_ms = duration_ms
            self._last_command = {
                "command": "move", "x": target[0], "y": target[1],
                "z": target[2], "duration_ms": duration_ms,
            }
        self.log(f"Move accepted: XYZ {target} ({duration_ms} ms)")
        threading.Thread(target=self._animate_move, args=(start, target, duration_ms),
                         name="maxarm-xyz-motion", daemon=True).start()
        return target

    def _animate_move(self, start: Point3D, target: Point3D, duration_ms: int) -> None:
        started = time.monotonic()
        duration_s = duration_ms / 1000.0
        while True:
            elapsed = time.monotonic() - started
            progress = min(1.0, elapsed / duration_s)
            position = interpolate(start, target, progress)
            with self._lock:
                self._current_xyz = position
                self._motion_progress = progress
                self._update_attached_object_locked()
            if progress >= 1.0:
                break
            time.sleep(1.0 / 60.0)

        with self._lock:
            self._current_xyz = target
            self._motion_progress = 1.0
            self._moving = False
            self._update_attached_object_locked()
        self.log(f"Move completed: {self._location}/{self._height}")

    def set_suction(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self._last_command = {
                "command": "suction",
                "state": "on" if enabled else "off",
            }
            if enabled:
                self._suction_on = True
                self._release_active = False
                attached = self._try_pick_locked()
                self.log(
                    f"Suction on" + (f"; picked {attached}" if attached else "; no object in range")
                )
                return {
                    "ok": True,
                    "command": "suction",
                    "state": "on",
                    "suction_on": True,
                }

            self._suction_on = False
            self._release_active = True
            released = self._attached_object_id

        thread = threading.Thread(
            target=self._finish_release,
            args=(released,),
            name="maxarm-release",
            daemon=True,
        )
        thread.start()
        self.log("Suction off; asynchronous release started")
        return {
            "ok": True,
            "command": "suction",
            "state": "off",
            "release_started": True,
        }

    def _try_pick_locked(self) -> str | None:
        if self._attached_object_id is not None:
            return self._attached_object_id
        ex, ey, ez = self._current_xyz
        nearest: tuple[float, SimObject] | None = None
        for obj in self._objects.values():
            if obj.attached:
                continue
            ox, oy, oz = obj.pickup_point
            distance = math.dist((ex, ey, ez), (ox, oy, oz))
            if distance <= PICK_TOLERANCE_MM and (nearest is None or distance < nearest[0]):
                nearest = (distance, obj)
        if nearest is None:
            return None
        obj = nearest[1]
        obj.attached = True
        obj.placed_at = None
        self._attached_object_id = obj.object_id
        self._update_attached_object_locked()
        return obj.object_id

    def _update_attached_object_locked(self) -> None:
        if self._attached_object_id is None:
            return
        obj = self._objects[self._attached_object_id]
        x, y, z = self._current_xyz
        obj.position = (x, y, z - OBJECT_HEIGHT_MM / 2.0)
        obj.pickup_point = (x, y, z)

    def _finish_release(self, object_id: str | None) -> None:
        time.sleep(1.0)
        with self._lock:
            if object_id and object_id in self._objects:
                obj = self._objects[object_id]
                x, y, _z = self._current_xyz
                nearest_name: str | None = None
                nearest_distance = float("inf")
                for name, heights in self.positions.items():
                    px, py, pz = heights["low"]
                    distance = math.hypot(x - px, y - py)
                    if distance < nearest_distance:
                        nearest_name = name
                        nearest_distance = distance
                if nearest_name and nearest_distance <= PLACE_RADIUS_MM:
                    px, py, pz = self.positions[nearest_name]["low"]
                    obj.position = (px, py, pz - OBJECT_HEIGHT_MM / 2.0)
                    obj.pickup_point = (px, py, pz)
                    obj.placed_at = nearest_name
                else:
                    obj.position = (x, y, OBJECT_HEIGHT_MM / 2.0)
                    obj.pickup_point = (x, y, OBJECT_HEIGHT_MM)
                    obj.placed_at = None
                obj.attached = False
            self._attached_object_id = None
            self._release_active = False
        self.log(f"Release completed" + (f": {object_id}" if object_id else ""))

    def health(self, deep: bool = False) -> dict[str, Any]:
        snapshot = self.snapshot()
        result: dict[str, Any] = {
            "status": "ok",
            "robot": "Hiwonder MaxArm Simulator",
            "simulator": True,
            "api": {
                "version": self.API_VERSION,
                "port": snapshot["api_port"],
                "uptime_ms": int((time.monotonic() - self.started_at) * 1000),
                "request_count": snapshot["request_count"],
                "error_count": snapshot["error_count"],
            },
            "system": {
                "implementation": "cpython-simulator",
                "micropython_version": "simulated MicroPython 1.12 compatibility",
                "cpu_frequency_hz": 160_000_000,
                "flash_size_bytes": 4_194_304,
            },
            "wifi": {"connected": True, "ip": snapshot["api_host"], "rssi_dbm": -42},
            "robot_state": {
                "arm_available": True,
                "bus_servo_available": True,
                "nozzle_available": True,
                "busy": snapshot["moving"],
            },
        }
        if deep:
            angles = snapshot["joint_angles_degrees"]
            result["servo_diagnostics"] = {
                "available": True,
                "all_servos_ok": True,
                "simulated": True,
                "servos": [
                    {
                        "id": index,
                        "communication_ok": True,
                        "position_raw": max(0, min(1000, round(500 + angle * 1000 / 240))),
                        "position_ok": True,
                        "vin_raw": 12_100,
                        "vin_volts_estimate": 12.1,
                        "vin_ok": True,
                    }
                    for index, angle in enumerate(
                        [angles["base"], angles["shoulder"], angles["elbow"]], start=1
                    )
                ],
            }
        return result

    def set_api_address(self, host: str, port: int) -> None:
        with self._lock:
            self._api_host = host
            self._api_port = port

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            current = self._current_xyz
            solution = solve_ik(current)
            return {
                "current_xyz": list(current),
                "target_xyz": list(self._target_xyz),
                "location": self._location,
                "height": self._height,
                "moving": self._moving,
                "motion_progress": self._motion_progress,
                "duration_ms": self._duration_ms,
                "suction_on": self._suction_on,
                "release_active": self._release_active,
                "attached_object_id": self._attached_object_id,
                "request_count": self._request_count,
                "error_count": self._error_count,
                "last_command": self._last_command,
                "api_host": getattr(self, "_api_host", "127.0.0.1"),
                "api_port": getattr(self, "_api_port", 8080),
                "joint_angles_degrees": {
                    "base": solution.base_degrees,
                    "shoulder": solution.shoulder_degrees,
                    "elbow": solution.elbow_degrees,
                },
                "joints": {
                    "shoulder": list(solution.shoulder),
                    "elbow": list(solution.elbow),
                    "wrist": list(solution.wrist),
                    "reachable": solution.reachable,
                },
                "objects": [asdict(obj) for obj in self._objects.values()],
                "events": list(self._events),
            }
