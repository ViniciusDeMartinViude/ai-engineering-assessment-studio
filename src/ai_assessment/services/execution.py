from __future__ import annotations

import hashlib
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from ..core.workspace import CandidateWorkspace


class ExecutionError(RuntimeError):
    """A candidate process could not be started or used safely."""


class ExecutionBusyError(ExecutionError):
    """The workspace already has an IDE Python run."""


class ProcessLike(Protocol):
    pid: int
    stdout: object
    stderr: object
    stdin: object

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    status: str
    exit_code: int | None
    duration_ms: int
    path: str
    log_path: str
    started_at: str
    ended_at: str
    error: str | None = None


@dataclass
class _ActiveProcess:
    run_id: str
    process: ProcessLike
    stop_requested: bool = False


OutputCallback = Callable[[str, str], None]
CompleteCallback = Callable[[ExecutionResult], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments(value: str) -> list[str]:
    """Parse the optional UI argument field without constructing a shell command."""
    if not value.strip():
        return []
    return shlex.split(value, posix=os.name != "nt")


class _WorkspaceProcessService:
    def __init__(self, workspace: CandidateWorkspace) -> None:
        self.workspace = workspace
        self._lock = threading.RLock()

    def _raw_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.workspace.root / candidate

    def _reject_reparse_components(self, raw: Path) -> None:
        root = self.workspace.root
        absolute = Path(os.path.abspath(raw))
        try:
            relative = absolute.relative_to(root)
        except ValueError as exc:
            raise ExecutionError(f"Path escapes candidate workspace: {path_display(raw)}") from exc

        current = root
        for component in relative.parts:
            current /= component
            if current.is_symlink() or os.path.islink(current):
                raise ExecutionError(f"Symlink or junction is not allowed: {path_display(current)}")
            try:
                attrs = os.stat(current, follow_symlinks=False).st_file_attributes
            except (AttributeError, FileNotFoundError, OSError):
                attrs = 0
            if attrs & getattr(stat_module(), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ExecutionError(f"Symlink or junction is not allowed: {path_display(current)}")

    def resolve_path(self, path: str | Path, *, editable: bool = False) -> Path:
        raw = self._raw_path(path)
        self._reject_reparse_components(raw)
        try:
            resolved = self.workspace.resolve_inside(raw)
        except ValueError as exc:
            raise ExecutionError(str(exc)) from exc
        if resolved == self.workspace.root:
            raise ExecutionError("A file path is required")
        relative = resolved.relative_to(self.workspace.root)
        if relative.parts and relative.parts[0].lower() == "dataset":
            if editable:
                raise ExecutionError("The original dataset is read-only")
        if editable:
            if not relative.parts or relative.parts[0].lower() != "src":
                raise ExecutionError("Candidate code must be stored under workspace/src")
            if (self.workspace.root / ".git").exists():
                raise ExecutionError("Repository source cannot be edited from the candidate IDE")
        return resolved

    def relative(self, path: Path) -> str:
        return path.relative_to(self.workspace.root).as_posix()


def path_display(path: Path) -> str:
    return str(path)


def stat_module():
    import stat

    return stat


class ExecutionService(_WorkspaceProcessService):
    """Run one saved candidate Python file at a time, outside the Qt thread."""

    def __init__(
        self,
        workspace: CandidateWorkspace,
        *,
        python_executable: str | None = None,
        process_factory: Callable[..., ProcessLike] = subprocess.Popen,
        max_ui_chars: int = 100_000,
    ) -> None:
        super().__init__(workspace)
        self.python_executable = python_executable or sys.executable
        self.process_factory = process_factory
        self.max_ui_chars = max_ui_chars
        self._active: _ActiveProcess | None = None

    @property
    def active_run_id(self) -> str | None:
        with self._lock:
            return self._active.run_id if self._active else None

    def save_file(self, path: str | Path, text: str) -> tuple[Path, str]:
        target = self.resolve_path(path, editable=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="")
        digest = _sha256(target)
        self.workspace.record_event(
            "ide.file_saved",
            {"path": self.relative(target), "bytes": target.stat().st_size},
            artifact_hashes={self.relative(target): digest},
        )
        return target, digest

    def load_file(self, path: str | Path) -> tuple[Path, str]:
        target = self.resolve_path(path)
        if not target.is_file():
            raise ExecutionError(f"File does not exist: {target}")
        return target, target.read_text(encoding="utf-8")

    def run_script(
        self,
        path: str | Path,
        args: list[str] | tuple[str, ...] = (),
        *,
        on_output: OutputCallback | None = None,
        on_complete: CompleteCallback | None = None,
    ) -> str:
        script = self.resolve_path(path, editable=True)
        if not script.is_file():
            raise ExecutionError(f"File does not exist: {script}")
        with self._lock:
            if self._active is not None:
                raise ExecutionBusyError("An IDE Python run is already active")
            run_id = f"exec-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
            log_path = self.workspace.resolve_inside("logs", "execution", f"{run_id}.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = self._child_environment()
            argv = [self.python_executable, str(script), *[str(value) for value in args]]
            started = _now()
            try:
                process = self.process_factory(
                    argv,
                    cwd=str(self.workspace.root),
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                    start_new_session=os.name != "nt",
                )
            except (OSError, ValueError) as exc:
                self.workspace.record_event(
                    "ide.execution_failed",
                    {"path": self.relative(script), "error": str(exc)},
                    outcome="failed",
                )
                raise ExecutionError(f"Could not launch Python: {exc}") from exc

            self._active = _ActiveProcess(run_id, process)
            self.workspace.record_event(
                "ide.execution_started",
                {"run_id": run_id, "path": self.relative(script), "args": list(args)},
            )
            thread = threading.Thread(
                target=self._monitor,
                args=(run_id, process, script, log_path, started, on_output, on_complete),
                name=f"ide-execution-{run_id}",
                daemon=True,
            )
            thread.start()
            return run_id

    def _child_environment(self) -> dict[str, str]:
        blocked_fragments = ("API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD", "ASSESSMENT_HIDDEN")
        environment = {
            key: value
            for key, value in os.environ.items()
            if not any(fragment in key.upper() for fragment in blocked_fragments)
        }
        source = str(self.workspace.root / "src")
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = source + (os.pathsep + existing if existing else "")
        return environment

    def _monitor(
        self,
        run_id: str,
        process: ProcessLike,
        script: Path,
        log_path: Path,
        started: str,
        on_output: OutputCallback | None,
        on_complete: CompleteCallback | None,
    ) -> None:
        write_lock = threading.Lock()
        log = log_path.open("w", encoding="utf-8", newline="")

        def read_stream(stream: object, name: str) -> None:
            if stream is None:
                return
            for line in iter(stream.readline, ""):
                if line == "":
                    break
                text = str(line)
                with write_lock:
                    log.write(f"[{name}] {text}")
                    log.flush()
                if on_output is not None:
                    on_output(name, text)

        readers = [
            threading.Thread(target=read_stream, args=(process.stdout, "stdout"), daemon=True),
            threading.Thread(target=read_stream, args=(process.stderr, "stderr"), daemon=True),
        ]
        for reader in readers:
            reader.start()
        try:
            exit_code = process.wait()
        except Exception as exc:  # a broken adapter still gets a durable result
            exit_code = None
            with write_lock:
                log.write(f"[service] wait failed: {exc}\n")
        for reader in readers:
            reader.join(timeout=2.0)
        log.close()

        with self._lock:
            active = self._active
            stopped = active is not None and active.run_id == run_id and active.stop_requested
        ended = _now()
        duration_ms = max(0, int((time.time() - _iso_to_epoch(started)) * 1000))
        status = "stopped" if stopped else ("completed" if exit_code == 0 else "failed")
        result = ExecutionResult(
            run_id=run_id,
            status=status,
            exit_code=exit_code,
            duration_ms=duration_ms,
            path=self.relative(script),
            log_path=self.relative(log_path),
            started_at=started,
            ended_at=ended,
        )
        self.workspace.record_event(
            f"ide.execution_{status}",
            {
                "run_id": run_id,
                "path": result.path,
                "exit_code": exit_code,
                "duration_ms": duration_ms,
                "log_path": result.log_path,
            },
            outcome=status,
            artifact_hashes={result.log_path: _sha256(log_path)},
        )
        if on_complete is not None:
            on_complete(result)
        with self._lock:
            if self._active is not None and self._active.run_id == run_id:
                self._active = None

    def stop(self, run_id: str | None = None) -> bool:
        with self._lock:
            active = self._active
            if active is None or (run_id is not None and active.run_id != run_id):
                return False
            active.stop_requested = True
            process = active.process
        _terminate_process_tree(process)
        return True

    def shutdown(self) -> None:
        self.stop()
        deadline = time.monotonic() + 3.0
        while self.active_run_id is not None and time.monotonic() < deadline:
            time.sleep(0.03)


class TerminalService(_WorkspaceProcessService):
    """A real stdin/stdout process, deliberately separate from Python execution."""

    def __init__(
        self,
        workspace: CandidateWorkspace,
        *,
        process_factory: Callable[..., ProcessLike] = subprocess.Popen,
        shell_command: list[str] | None = None,
    ) -> None:
        super().__init__(workspace)
        self.process_factory = process_factory
        self.shell_command = shell_command
        self._active: _ActiveProcess | None = None
        self._on_output: OutputCallback | None = None
        self._on_exit: Callable[[], None] | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active is not None

    def start(
        self,
        *,
        on_output: OutputCallback | None = None,
        on_exit: Callable[[], None] | None = None,
    ) -> None:
        with self._lock:
            if self._active is not None:
                return
            command = self.shell_command or ([os.environ.get("COMSPEC", "cmd.exe")] if os.name == "nt" else ["/bin/sh"])
            try:
                process = self.process_factory(
                    command,
                    cwd=str(self.workspace.root),
                    env=ExecutionService(self.workspace)._child_environment(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                    start_new_session=os.name != "nt",
                )
            except (OSError, ValueError) as exc:
                self.workspace.record_event("ide.terminal_failed", {"error": str(exc)}, outcome="failed")
                raise ExecutionError(f"Could not launch terminal: {exc}") from exc
            self._active = _ActiveProcess(f"terminal-{uuid.uuid4().hex[:8]}", process)
            self._on_output = on_output
            self._on_exit = on_exit
            self.workspace.record_event("ide.terminal_started", {"cwd": "."})
            threading.Thread(target=self._monitor, args=(process,), daemon=True, name="ide-terminal").start()

    def write(self, command: str) -> None:
        with self._lock:
            active = self._active
            if active is None or active.process.stdin is None:
                raise ExecutionError("Terminal is not running")
            active.process.stdin.write(command + "\n")
            active.process.stdin.flush()

    def _monitor(self, process: ProcessLike) -> None:
        def read_stream(stream: object, name: str) -> None:
            if stream is None:
                return
            for line in iter(stream.readline, ""):
                if line == "":
                    break
                if self._on_output is not None:
                    self._on_output(name, str(line))

        readers = [
            threading.Thread(target=read_stream, args=(process.stdout, "stdout"), daemon=True),
            threading.Thread(target=read_stream, args=(process.stderr, "stderr"), daemon=True),
        ]
        for reader in readers:
            reader.start()
        try:
            process.wait()
        finally:
            for reader in readers:
                reader.join(timeout=1.0)
            with self._lock:
                if self._active is not None and self._active.process is process:
                    self._active = None
            self.workspace.record_event("ide.terminal_stopped", {})
            if self._on_exit is not None:
                self._on_exit()

    def stop(self) -> None:
        with self._lock:
            active = self._active
            if active is None:
                return
            active.stop_requested = True
        _terminate_process_tree(active.process)

    def restart(self, *, on_output: OutputCallback | None = None, on_exit: Callable[[], None] | None = None) -> None:
        self.stop()
        deadline = time.monotonic() + 2.0
        while self.active and time.monotonic() < deadline:
            time.sleep(0.03)
        self.start(on_output=on_output, on_exit=on_exit)

    def shutdown(self) -> None:
        self.stop()
        deadline = time.monotonic() + 3.0
        while self.active and time.monotonic() < deadline:
            time.sleep(0.03)


def _terminate_process_tree(process: ProcessLike) -> None:
    pid = getattr(process, "pid", None)
    if pid and os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    elif pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    try:
        process.terminate()
        process.wait(timeout=1.5)
    except (OSError, subprocess.TimeoutExpired, TimeoutError):
        try:
            process.kill()
            process.wait(timeout=1.0)
        except (OSError, subprocess.SubprocessError, TimeoutError):
            pass


def _iso_to_epoch(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()
