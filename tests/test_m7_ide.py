from __future__ import annotations

import io
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ai_assessment.core.workspace import CandidateWorkspace
from ai_assessment.services.execution import (
    ExecutionBusyError,
    ExecutionError,
    ExecutionService,
)


class FakeProcess:
    _next_pid = 50000

    def __init__(self, *, return_code: int = 0, block: bool = False) -> None:
        FakeProcess._next_pid += 1
        self.pid = FakeProcess._next_pid
        self.stdout = io.StringIO("hello \N{SNOWMAN}\n")
        self.stderr = io.StringIO("traceback line\n")
        self.stdin = io.StringIO()
        self.return_code = return_code
        self.terminated = False
        self.release = threading.Event() if block else None

    def poll(self) -> int | None:
        if self.release is not None and not self.release.is_set():
            return None
        return 1 if self.terminated else self.return_code

    def wait(self, timeout: float | None = None) -> int:
        if self.release is not None:
            if not self.release.wait(timeout):
                raise TimeoutError("still running")
        return 1 if self.terminated else self.return_code

    def terminate(self) -> None:
        self.terminated = True
        if self.release is not None:
            self.release.set()

    def kill(self) -> None:
        self.terminate()


class M7ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = CandidateWorkspace.create(Path(self.temp_dir.name) / "C014", candidate_id="C014")
        (self.workspace.root / "src" / "main.py").write_text("print('test')\n", encoding="utf-8")
        self.processes: list[FakeProcess] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def factory(self, argv, **_kwargs) -> FakeProcess:
        process = FakeProcess()
        process.argv = argv
        self.processes.append(process)
        return process

    def wait_for_run(self, service: ExecutionService) -> None:
        deadline = time.monotonic() + 3
        while service.active_run_id is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNone(service.active_run_id)

    def test_save_reopen_and_workspace_path_safety(self) -> None:
        service = ExecutionService(self.workspace, process_factory=self.factory)
        path, digest = service.save_file("src/main.py", "print('héllo')\n")
        loaded_path, text = service.load_file("src/main.py")
        self.assertEqual(path, loaded_path)
        self.assertEqual(text, "print('héllo')\n")
        self.assertEqual(len(digest), 64)
        with self.assertRaises(ExecutionError):
            service.save_file("src/../dataset/labels.txt", "bad")
        with self.assertRaises(ExecutionError):
            service.save_file(Path("..") / "outside.py", "bad")

    def test_symlink_is_rejected_when_platform_allows_creating_one(self) -> None:
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        link = self.workspace.root / "src" / "link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is unavailable on this Windows account")
        with self.assertRaises(ExecutionError):
            ExecutionService(self.workspace).save_file(link / "file.py", "bad")

    def test_streams_exit_code_and_complete_log_are_recorded(self) -> None:
        service = ExecutionService(self.workspace, process_factory=self.factory)
        received: list[tuple[str, str]] = []
        completed = []
        run_id = service.run_script("src/main.py", on_output=lambda stream, text: received.append((stream, text)), on_complete=completed.append)
        self.wait_for_run(service)
        self.assertEqual(completed[0].status, "completed")
        self.assertEqual(completed[0].exit_code, 0)
        self.assertEqual({stream for stream, _ in received}, {"stdout", "stderr"})
        log = self.workspace.root / completed[0].log_path
        self.assertIn("[stdout] hello", log.read_text(encoding="utf-8"))
        self.assertTrue(any(event.event_type == "ide.execution_completed" for event in self.workspace.events.read_events()))
        self.assertTrue(run_id.startswith("exec-"))

    def test_launch_failure_is_reported_and_not_left_active(self) -> None:
        def failing_factory(*_args, **_kwargs):
            raise OSError("interpreter missing")

        service = ExecutionService(self.workspace, process_factory=failing_factory)
        with self.assertRaises(ExecutionError):
            service.run_script("src/main.py")
        self.assertIsNone(service.active_run_id)
        event = self.workspace.events.read_events()[-1]
        self.assertEqual(event.event_type, "ide.execution_failed")
        self.assertEqual(event.outcome, "failed")

    def test_only_one_python_run_and_stop_does_not_touch_other_service(self) -> None:
        def blocking_factory(argv, **_kwargs):
            process = FakeProcess(block=True)
            process.argv = argv
            self.processes.append(process)
            return process

        service = ExecutionService(self.workspace, process_factory=blocking_factory)
        run_id = service.run_script("src/main.py")
        with self.assertRaises(ExecutionBusyError):
            service.run_script("src/main.py")
        self.assertTrue(service.stop(run_id))
        self.wait_for_run(service)
        self.assertEqual(self.processes[0].terminated, True)
        self.assertEqual(self.workspace.events.read_events()[-1].event_type, "ide.execution_stopped")

    def test_file_save_hash_event_is_relative_to_workspace(self) -> None:
        service = ExecutionService(self.workspace, process_factory=self.factory)
        service.save_file("src/example.py", "# example\n")
        event = self.workspace.events.read_events()[-1]
        self.assertEqual(event.event_type, "ide.file_saved")
        self.assertEqual(event.payload["path"], "src/example.py")
        self.assertNotIn(str(self.workspace.root), str(event.payload))


if __name__ == "__main__":
    unittest.main()
