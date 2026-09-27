"""Robot HTTP contracts, adapters, safety state, and supervised sequences."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import requests

from ..core.workspace import CandidateWorkspace


POSITIONS: dict[str, dict[str, tuple[float, float, float]]] = {
    "P1": {"low": (-3.0, -130.0, 59.0), "high": (-3.0, -130.0, 89.0)},
    "P2": {"low": (-124.0, -161.0, 59.0), "high": (-124.0, -161.0, 89.0)},
    "P3": {"low": (-124.0, -98.0, 59.0), "high": (-124.0, -98.0, 89.0)},
    "P4": {"low": (-124.0, -35.0, 59.0), "high": (-124.0, -35.0, 89.0)},
    "P5": {"low": (-124.0, 28.0, 59.0), "high": (-124.0, 28.0, 89.0)},
}


class RobotError(RuntimeError):
    """Base class for expected robot transport and contract failures."""


class RobotTransportError(RobotError):
    pass


class RobotTimeoutError(RobotTransportError):
    pass


class RobotResponseError(RobotError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RobotBusyError(RobotResponseError):
    pass


class RobotUncertainError(RobotError):
    """A command may have reached the controller, but completion is unknown."""


class RobotSafetyError(RobotError):
    pass


@dataclass(frozen=True)
class RobotCapabilities:
    simulator: bool = False
    cartesian_move: bool = False
    software_stop: bool = False


@dataclass(frozen=True)
class RobotResponse:
    method: str
    path: str
    status_code: int
    data: dict[str, Any]
    elapsed_ms: int


class RobotAdapter(Protocol):
    base_url: str
    capabilities: RobotCapabilities

    def health(self, deep: bool = False) -> RobotResponse: ...
    def positions(self) -> RobotResponse: ...
    def suction_state(self) -> RobotResponse: ...
    def move(self, location: str, height: str, duration_ms: int) -> RobotResponse: ...
    def suction(self, state: str) -> RobotResponse: ...
    def move_xyz(self, x: float, y: float, z: float, duration_ms: int) -> RobotResponse: ...


class RequestsRobotAdapter:
    """HTTP adapter shared by the local simulator and physical MaxArm."""

    def __init__(
        self,
        base_url: str,
        *,
        capabilities: RobotCapabilities,
        timeout_s: float = 5.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.capabilities = capabilities
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> RobotResponse:
        started = time.monotonic()
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout_s,
                headers={"Accept": "application/json"},
            )
        except requests.Timeout as error:
            raise RobotTimeoutError(f"Timed out calling {method} {path}") from error
        except requests.RequestException as error:
            raise RobotTransportError(f"Could not call {method} {path}: {error}") from error

        elapsed_ms = int((time.monotonic() - started) * 1000)
        try:
            data = response.json()
        except ValueError as error:
            raise RobotResponseError(
                f"Invalid JSON from {method} {path} (HTTP {response.status_code})",
                response.status_code,
            ) from error
        if not isinstance(data, dict):
            raise RobotResponseError("Robot response JSON must be an object", response.status_code)
        if response.status_code == 409:
            raise RobotBusyError(str(data.get("error", "Robot is busy")), 409)
        if not 200 <= response.status_code < 300:
            raise RobotResponseError(
                str(data.get("error", f"HTTP {response.status_code}")), response.status_code
            )
        if data.get("ok") is False:
            raise RobotResponseError(str(data.get("error", "Robot rejected command")), response.status_code)
        return RobotResponse(method, path, response.status_code, data, elapsed_ms)

    def health(self, deep: bool = False) -> RobotResponse:
        return self._request("GET", "/health/deep" if deep else "/health")

    def positions(self) -> RobotResponse:
        return self._request("GET", "/positions")

    def suction_state(self) -> RobotResponse:
        return self._request("GET", "/suction")

    def move(self, location: str, height: str, duration_ms: int) -> RobotResponse:
        return self._request(
            "POST",
            "/command",
            {"command": "move", "location": location, "height": height, "duration_ms": duration_ms},
        )

    def suction(self, state: str) -> RobotResponse:
        return self._request("POST", "/command", {"command": "suction", "state": state})

    def move_xyz(self, x: float, y: float, z: float, duration_ms: int) -> RobotResponse:
        if not self.capabilities.cartesian_move:
            raise RobotSafetyError("Cartesian XYZ movement is not verified for this target")
        return self._request(
            "POST",
            "/command",
            {"command": "move", "x": x, "y": y, "z": z, "duration_ms": duration_ms},
        )

    def simulator_state(self) -> RobotResponse:
        if not self.capabilities.simulator:
            raise RobotSafetyError("Simulator state is not available for the physical target")
        return self._request("GET", "/sim/state")


class SimulatorRobotAdapter(RequestsRobotAdapter):
    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__(
            base_url,
            capabilities=RobotCapabilities(simulator=True, cartesian_move=True),
            **kwargs,
        )


class PhysicalMaxArmAdapter(RequestsRobotAdapter):
    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__(
            base_url,
            capabilities=RobotCapabilities(),
            **kwargs,
        )


@dataclass(frozen=True)
class TrialStep:
    command: str
    location: str | None = None
    height: str | None = None
    state: str | None = None


@dataclass(frozen=True)
class TrialResult:
    status: str
    completed_steps: tuple[str, ...] = ()
    message: str = ""


@dataclass
class RobotService:
    workspace: CandidateWorkspace
    adapter: RobotAdapter
    sleep_fn: Callable[[float], None] = time.sleep
    _command_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _physical_armed: bool = field(default=False, init=False)
    _sequence_active: bool = field(default=False, init=False)

    @property
    def capabilities(self) -> RobotCapabilities:
        return self.adapter.capabilities

    @property
    def physical_armed(self) -> bool:
        return self._physical_armed

    @property
    def sequence_active(self) -> bool:
        return self._sequence_active

    def _require_motion_authorized(self) -> None:
        if not self.capabilities.simulator and not self._physical_armed:
            raise RobotSafetyError("Physical robot is disarmed; enable it for this session first")

    def _record(self, event_type: str, payload: dict[str, Any], outcome: str = "recorded") -> None:
        self.workspace.record_event(event_type, payload, outcome=outcome)

    def health(self, deep: bool = False) -> RobotResponse:
        response = self.adapter.health(deep)
        self._record("robot.health.checked", {"deep": deep, "status_code": response.status_code})
        return response

    def positions(self) -> RobotResponse:
        response = self.adapter.positions()
        self._record("robot.positions.read", {"status_code": response.status_code})
        return response

    def suction_state(self) -> RobotResponse:
        return self.adapter.suction_state()

    def connect_ready(self, *, deep: bool = False) -> tuple[RobotResponse, RobotResponse]:
        health = self.health(deep)
        positions = self.positions()
        self._record("robot.connection.ready", {"deep": deep})
        return health, positions

    def arm_physical(self) -> None:
        if self.capabilities.simulator:
            return
        self.connect_ready()
        self._physical_armed = True
        self._record("robot.physical.armed", {"base_url": self.adapter.base_url})

    def disarm(self) -> None:
        if self._physical_armed:
            self._physical_armed = False
            self._record("robot.physical.disarmed", {})

    def move(self, location: str, height: str, duration_ms: int) -> RobotResponse:
        if location not in POSITIONS or height not in {"high", "low"}:
            raise RobotSafetyError("Move must use a P1-P5 location and high or low height")
        if not 100 <= duration_ms <= 10_000:
            raise RobotSafetyError("Move duration must be between 100 and 10000 ms")
        self._require_motion_authorized()
        with self._command_lock:
            self._record("robot.command.started", {"command": "move", "location": location, "height": height, "duration_ms": duration_ms})
            try:
                response = self.adapter.move(location, height, duration_ms)
            except RobotTimeoutError as error:
                self._record("robot.command.uncertain", {"command": "move", "location": location, "height": height, "error": str(error)}, "uncertain")
                raise RobotUncertainError(str(error)) from error
            except RobotError as error:
                self._record("robot.command.error", {"command": "move", "location": location, "height": height, "error": str(error)}, "error")
                raise
            self._record("robot.response.received", {"command": "move", "location": location, "height": height, "status_code": response.status_code, "response": response.data})
            self._record("robot.command.completed", {"command": "move", "location": location, "height": height, "duration_ms": duration_ms, "status_code": response.status_code})
            return response

    def move_xyz(self, x: float, y: float, z: float, duration_ms: int) -> RobotResponse:
        if not self.capabilities.simulator or not self.capabilities.cartesian_move:
            raise RobotSafetyError("XYZ movement is simulator-only until a physical firmware contract is verified")
        if not 100 <= duration_ms <= 10_000:
            raise RobotSafetyError("Move duration must be between 100 and 10000 ms")
        with self._command_lock:
            self._record("robot.command.started", {"command": "move_xyz", "x": x, "y": y, "z": z, "duration_ms": duration_ms})
            try:
                response = self.adapter.move_xyz(x, y, z, duration_ms)
            except RobotTimeoutError as error:
                self._record("robot.command.uncertain", {"command": "move_xyz", "error": str(error)}, "uncertain")
                raise RobotUncertainError(str(error)) from error
            except RobotError as error:
                self._record("robot.command.error", {"command": "move_xyz", "error": str(error)}, "error")
                raise
            self._record("robot.response.received", {"command": "move_xyz", "status_code": response.status_code, "response": response.data})
            self._record("robot.command.completed", {"command": "move_xyz", "status_code": response.status_code})
            return response

    def suction(self, state: str) -> RobotResponse:
        if state not in {"on", "off"}:
            raise RobotSafetyError("Suction state must be on or off")
        self._require_motion_authorized()
        with self._command_lock:
            self._record("robot.command.started", {"command": "suction", "state": state})
            try:
                response = self.adapter.suction(state)
            except RobotTimeoutError as error:
                self._record("robot.command.uncertain", {"command": "suction", "state": state, "error": str(error)}, "uncertain")
                raise RobotUncertainError(str(error)) from error
            except RobotError as error:
                self._record("robot.command.error", {"command": "suction", "state": state, "error": str(error)}, "error")
                raise
            self._record("robot.response.received", {"command": "suction", "state": state, "status_code": response.status_code, "response": response.data})
            self._record("robot.command.completed", {"command": "suction", "state": state, "status_code": response.status_code})
            return response

    def run_pick_place(
        self,
        source: str,
        destination: str,
        duration_ms: int,
        *,
        cancel_event: threading.Event | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> TrialResult:
        if source not in POSITIONS or destination not in POSITIONS:
            raise RobotSafetyError("Pick/place source and destination must be P1-P5")
        if source == destination:
            raise RobotSafetyError("Pick/place destination must differ from source")
        cancel_event = cancel_event or threading.Event()
        steps = (
            TrialStep("move", source, "high"),
            TrialStep("move", source, "low"),
            TrialStep("suction", state="on"),
            TrialStep("move", source, "high"),
            TrialStep("move", destination, "high"),
            TrialStep("move", destination, "low"),
            TrialStep("suction", state="off"),
            TrialStep("move", destination, "high"),
        )
        completed: list[str] = []
        with self._command_lock:
            self._require_motion_authorized()
            if self._sequence_active:
                raise RobotBusyError("A supervised trial is already running")
            self._sequence_active = True
            self._record("robot.trial.started", {"source": source, "destination": destination, "duration_ms": duration_ms})
            try:
                for index, step in enumerate(steps, start=1):
                    if cancel_event.is_set():
                        message = "Cancelled before the next command; an accepted move was not stopped."
                        self._record("robot.trial.cancelled", {"completed_steps": completed}, "cancelled")
                        return TrialResult("cancelled", tuple(completed), message)
                    label = f"{step.command}:{step.location or step.state}"
                    if progress:
                        progress(index - 1, len(steps), label)
                    try:
                        if step.command == "move":
                            self.move(step.location or source, step.height or "high", duration_ms)
                            wait_s = duration_ms / 1000.0 + 0.15
                        else:
                            self.suction(step.state or "off")
                            wait_s = 1.1 if step.state == "off" else 0.25
                    except RobotUncertainError as error:
                        self._record("robot.trial.uncertain", {"step": label, "error": str(error)}, "uncertain")
                        return TrialResult("uncertain", tuple(completed), str(error))
                    except RobotError as error:
                        self._record("robot.trial.error", {"step": label, "error": str(error)}, "error")
                        return TrialResult("error", tuple(completed), str(error))
                    completed.append(label)
                    self.sleep_fn(wait_s)
                    if cancel_event.is_set():
                        message = "Cancelled after an accepted command; the controller was not stopped."
                        self._record("robot.trial.cancelled", {"completed_steps": completed}, "cancelled_uncertain")
                        return TrialResult("cancelled_uncertain", tuple(completed), message)
                if progress:
                    progress(len(steps), len(steps), "complete")
                self._record("robot.trial.completed", {"completed_steps": completed}, "completed")
                return TrialResult("completed", tuple(completed))
            finally:
                self._sequence_active = False

