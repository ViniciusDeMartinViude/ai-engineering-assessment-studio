from __future__ import annotations

import threading
import math
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .model import RobotBusyError, RobotModel


def error_response(message: str, status_code: int, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "error": message, **extra},
    )


def create_app(model: RobotModel) -> FastAPI:
    app = FastAPI(
        title="MaxArm Simulator API",
        version=RobotModel.API_VERSION,
        docs_url="/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_counter(request: Request, call_next):
        model.count_request()
        try:
            response = await call_next(request)
        except Exception as exc:
            model.count_error(f"Unhandled API error: {exc}")
            return error_response("Unexpected controller/API error", 500)
        return response

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "MaxArm Visual Simulator",
            "api_version": RobotModel.API_VERSION,
            "simulator": True,
            "health": "/health",
            "interactive_docs": "/docs",
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return model.health(deep=False)

    @app.get("/health/deep")
    async def deep_health() -> dict[str, Any]:
        return model.health(deep=True)

    @app.get("/positions")
    async def positions() -> dict[str, Any]:
        return {
            "ok": True,
            "positions": {
                name: {
                    height: list(coordinates)
                    for height, coordinates in heights.items()
                }
                for name, heights in model.positions.items()
            },
        }

    @app.get("/suction")
    async def suction_state() -> dict[str, Any]:
        state = model.snapshot()
        return {"available": True, "on": state["suction_on"]}

    @app.post("/command")
    async def command(request: Request):
        try:
            payload = await request.json()
        except Exception:
            model.count_error("Invalid JSON body")
            return error_response("Invalid JSON", 400)
        if not isinstance(payload, dict):
            model.count_error("JSON body must be an object")
            return error_response("JSON body must be an object", 400)

        command_name = payload.get("command")
        if command_name == "move":
            return handle_move(payload)
        if command_name == "suction":
            return handle_suction(payload)
        model.count_error(f"Unknown command: {command_name!r}")
        return error_response(
            "Unknown command",
            400,
            supported_commands=["move", "suction"],
        )

    def handle_move(payload: dict[str, Any]):
        location = payload.get("location")
        height = payload.get("height")
        duration_ms = payload.get("duration_ms", 1000)

        xyz_keys = {"x", "y", "z"}
        has_xyz = bool(xyz_keys.intersection(payload))
        if has_xyz and (location is not None or height is not None or not xyz_keys.issubset(payload)):
            return error_response("Provide either location/height or all of x/y/z", 400)

        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
            return error_response("duration_ms must be an integer", 400)
        if duration_ms < 100:
            return error_response("duration_ms is too small", 400, minimum_ms=100)
        if duration_ms > 10_000:
            return error_response("duration_ms is too large", 400, maximum_ms=10_000)

        if has_xyz:
            coords = [payload[key] for key in ("x", "y", "z")]
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in coords):
                return error_response("x/y/z must be finite numbers", 400)
            target = tuple(float(value) for value in coords)
            try:
                model.start_move_xyz(target, duration_ms)
            except RobotBusyError:
                return error_response("Robot is busy", 409)
            except ValueError as exc:
                return error_response(str(exc), 400)
            return {"ok": True, "command": "move", "position":
                    dict(zip(("x", "y", "z"), target)), "duration_ms": duration_ms}

        if location is None:
            return error_response("Missing location", 400)
        if location not in model.positions:
            return error_response(
                "location must be P1 to P5",
                400,
                supported_locations=sorted(model.positions),
            )
        if height not in {"high", "low"}:
            return error_response("height must be high or low", 400)
        try:
            target = model.start_move(location, height, duration_ms)
        except RobotBusyError:
            return error_response("Robot is busy", 409)
        except ValueError as exc:
            model.count_error(str(exc))
            return error_response(str(exc), 400)

        return {
            "ok": True,
            "command": "move",
            "location": location,
            "height": height,
            "position": {"x": target[0], "y": target[1], "z": target[2]},
            "duration_ms": duration_ms,
        }

    def handle_suction(payload: dict[str, Any]):
        state = payload.get("state")
        if state not in {"on", "off"}:
            return error_response("state must be on or off", 400)
        return model.set_suction(state == "on")

    # Simulator-only endpoints. They do not change the robot-compatible API.
    @app.get("/sim/state")
    async def simulator_state() -> dict[str, Any]:
        return {"ok": True, **model.snapshot()}

    @app.post("/sim/reset")
    async def simulator_reset():
        try:
            model.reset()
        except RobotBusyError:
            return error_response("Robot is busy", 409)
        return {"ok": True, "reset": True}

    @app.post("/sim/objects/reset")
    async def simulator_objects_reset() -> dict[str, Any]:
        model.reset_objects()
        model.log("Virtual objects reset")
        return {"ok": True, "objects_reset": True}

    return app


class ApiServerThread(threading.Thread):
    def __init__(self, model: RobotModel, host: str, port: int):
        super().__init__(name="maxarm-api", daemon=True)
        self.model = model
        self.host = host
        self.port = port
        self.server: uvicorn.Server | None = None
        self.startup_error: str | None = None

    def run(self) -> None:
        self.model.set_api_address(self.host, self.port)
        app = create_app(self.model)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        try:
            self.model.log(f"HTTP API listening on http://{self.host}:{self.port}")
            self.server.run()
        except BaseException as exc:
            self.startup_error = str(exc)
            self.model.count_error(f"API server stopped: {exc}")

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
