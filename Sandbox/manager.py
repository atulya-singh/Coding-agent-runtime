from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from Execution.tools.base import ToolTimeoutError

from . import docker_backend as docker
from .config import SandboxConfig

logger = logging.getLogger("sandbox")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

TASK_ROOT = "/task"
REPO_DIR = f"{TASK_ROOT}/repository"
WORKSPACE_DIR = f"{TASK_ROOT}/workspace"
LOGS_DIR = f"{TASK_ROOT}/logs"

MAX_LOGGED_OUTPUT_CHARS = 4000


class SandboxState(str, Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    COLLECTED = "collected"
    DESTROYED = "destroyed"
    FAILED = "failed"


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    success: bool
    duration_ms: float
    timed_out: bool = False


def _event(task_id: str, event: str, metadata: dict) -> str:
    return json.dumps(
        {
            "task_id": task_id,
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
    )


class Sandbox:
    """One isolated task environment: /task/{repository,workspace,logs} inside a
    resource-limited container. Use as a context manager so destroy() always runs.
    """

    def __init__(self, task_id: Optional[str] = None, config: Optional[SandboxConfig] = None):
        self.task_id = task_id or f"task-{uuid.uuid4().hex[:12]}"
        self.config = config or SandboxConfig()
        self.container_name = f"sandbox-{self.task_id}"
        self.container_id: Optional[str] = None
        self.state = SandboxState.CREATED
        self._local_logs_dir = Path(tempfile.mkdtemp(prefix=f"{self.container_name}-logs-"))
        self._run_count = 0

    # ---- lifecycle: CREATE -------------------------------------------------

    def create(self) -> "Sandbox":
        logger.info(_event(self.task_id, "sandbox_create", {"image": self.config.image}))
        if not docker.image_exists(self.config.image):
            docker.pull_image(self.config.image)
        self.container_id = docker.create_container(self.container_name, self.config, workdir=TASK_ROOT)
        docker.start_container(self.container_id)
        self._exec_or_raise(f"mkdir -p {REPO_DIR} {WORKSPACE_DIR} {LOGS_DIR}")
        self.state = SandboxState.CREATED
        return self

    # ---- lifecycle: INITIALIZE ----------------------------------------------

    def initialize(self, repository_path: str) -> None:
        """Copy a local repository checkout into /task/repository."""
        assert self.container_id, "sandbox not created"
        docker.copy_into_container(self.container_id, repository_path.rstrip("/") + "/.", REPO_DIR)
        self.state = SandboxState.INITIALIZED
        logger.info(_event(self.task_id, "sandbox_initialize", {"repository_path": repository_path}))

    # ---- lifecycle: RUN COMMAND / RUN TESTS -------------------------------

    def run_command(self, command: str, cwd: str = REPO_DIR, timeout: Optional[float] = None) -> SandboxResult:
        return self._run(command, cwd, timeout)

    def run_tests(self, command: str = "pytest -q", cwd: str = REPO_DIR, timeout: Optional[float] = None) -> SandboxResult:
        return self._run(command, cwd, timeout)

    # ---- lifecycle: COLLECT RESULTS ----------------------------------------

    def collect_results(self, dest_dir: str) -> Path:
        """Copy /task/repository and /task/logs (plus host-captured run logs) out."""
        assert self.container_id, "sandbox not created"
        out = Path(dest_dir)
        out.mkdir(parents=True, exist_ok=True)

        docker.copy_from_container(self.container_id, REPO_DIR, str(out / "repository"))
        docker.copy_from_container(self.container_id, LOGS_DIR, str(out / "logs"))

        runs_log = self._local_logs_dir / "runs.jsonl"
        if runs_log.exists():
            (out / "logs").mkdir(parents=True, exist_ok=True)
            shutil.copy2(runs_log, out / "logs" / "runs.jsonl")

        self.state = SandboxState.COLLECTED
        logger.info(_event(self.task_id, "sandbox_collect", {"dest_dir": str(out)}))
        return out

    # ---- lifecycle: DESTROY -------------------------------------------------

    def destroy(self) -> None:
        if self.container_id:
            docker.stop_container(self.container_id, timeout=10.0)
            docker.remove_container(self.container_id, force=True)
            self.container_id = None
        shutil.rmtree(self._local_logs_dir, ignore_errors=True)
        self.state = SandboxState.DESTROYED
        logger.info(_event(self.task_id, "sandbox_destroy", {}))

    # ---- internals -----------------------------------------------------------

    def _run(self, command: str, cwd: str, timeout: Optional[float]) -> SandboxResult:
        assert self.container_id, "sandbox not created"
        effective_timeout = timeout or self.config.timeout_seconds
        self.state = SandboxState.RUNNING
        self._run_count += 1
        start = time.monotonic()

        try:
            result = docker.exec_in_container(self.container_id, command, cwd=cwd, timeout=effective_timeout)
        except ToolTimeoutError as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(_event(self.task_id, "sandbox_run_timeout", {"command": command, "timeout": effective_timeout}))
            sandbox_result = SandboxResult(exit_code=-1, stdout="", stderr=str(exc), success=False, duration_ms=duration_ms, timed_out=True)
            self._record_run(command, cwd, sandbox_result)
            # A killed `docker exec` client does not kill the process it started
            # inside the container -- tear the whole container down so a hung
            # command can't keep burning its CPU/pids quota. Must happen after
            # _record_run: destroy() deletes the local logs dir it writes into.
            self.destroy()
            self.state = SandboxState.FAILED
            return sandbox_result

        duration_ms = (time.monotonic() - start) * 1000
        sandbox_result = SandboxResult(
            exit_code=result["exit_code"],
            stdout=result["stdout"],
            stderr=result["stderr"],
            success=result["success"],
            duration_ms=duration_ms,
        )
        logger.info(
            _event(
                self.task_id,
                "sandbox_run",
                {"command": command, "exit_code": sandbox_result.exit_code, "duration_ms": round(duration_ms, 2)},
            )
        )
        self._record_run(command, cwd, sandbox_result)
        return sandbox_result

    def _record_run(self, command: str, cwd: str, result: SandboxResult) -> None:
        entry = {
            "run_index": self._run_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "cwd": cwd,
            "exit_code": result.exit_code,
            "duration_ms": round(result.duration_ms, 2),
            "success": result.success,
            "timed_out": result.timed_out,
            "stdout": result.stdout[:MAX_LOGGED_OUTPUT_CHARS],
            "stderr": result.stderr[:MAX_LOGGED_OUTPUT_CHARS],
        }
        with open(self._local_logs_dir / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _exec_or_raise(self, command: str) -> None:
        result = docker.exec_in_container(self.container_id, command, timeout=30.0)
        if not result["success"]:
            raise RuntimeError(f"sandbox init command failed: {command}: {result['stderr']}")

    # ---- context manager -----------------------------------------------------

    def __enter__(self) -> "Sandbox":
        self.create()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()
