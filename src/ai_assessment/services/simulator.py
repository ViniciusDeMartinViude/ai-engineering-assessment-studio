"""Embedded HTTP simulator wrapper for the preserved MaxArm prototype."""

from __future__ import annotations

import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class EmbeddedSimulator:
    """Own the preserved simulator model and its externally reachable API."""

    host: str = "127.0.0.1"
    port: int | None = None

    def __post_init__(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        try:
            from legacy.vision.ai_engineering_studio.maxarm_sim.api import ApiServerThread
            from legacy.vision.ai_engineering_studio.maxarm_sim.config import load_positions
            from legacy.vision.ai_engineering_studio.maxarm_sim.model import RobotModel
        except ImportError as error:
            raise RuntimeError(
                "The preserved MaxArm simulator is unavailable. Run the application from the project root."
            ) from error
        self.port = self.port or find_free_port()
        self.model = RobotModel(load_positions())
        self._server = ApiServerThread(self.model, self.host, self.port)
        self._started = False

    @property
    def base_url(self) -> str:
        connect_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{connect_host}:{self.port}"

    @property
    def external_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout_s: float = 5.0) -> None:
        if self._started:
            return
        self._server.start()
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=0.25)
                if response.ok:
                    self._started = True
                    return
            except requests.RequestException as error:
                last_error = error
            time.sleep(0.05)
        self._server.stop()
        self._server.join(timeout=1.0)
        raise RuntimeError(f"Embedded simulator did not start: {last_error or self._server.startup_error}")

    def stop(self) -> None:
        if not self._started and not self._server.is_alive():
            return
        self._server.stop()
        self._server.join(timeout=3.0)
        self._started = False

