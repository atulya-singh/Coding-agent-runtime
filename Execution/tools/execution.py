from __future__ import annotations

import subprocess
from typing import Optional

from .base import ToolTimeoutError, tool

DEFAULT_COMMAND_TIMEOUT = 30.0
DEFAULT_TEST_TIMEOUT = 300.0
MAX_OUTPUT_CHARS = 20_000


def truncate_output(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n...[truncated {len(text) - MAX_OUTPUT_CHARS} chars]"


def _run(command: str, cwd: Optional[str], timeout: float, env: Optional[dict]) -> dict:
    try:
        proc = subprocess.run(
            command, cwd=cwd, env=env, shell=True, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeoutError(f"command timed out after {timeout}s: {command}") from exc
    return {
        "command": command,
        "exit_code": proc.returncode,
        "stdout": truncate_output(proc.stdout),
        "stderr": truncate_output(proc.stderr),
        "success": proc.returncode == 0,
    }


@tool("run_command")
def run_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
    env: Optional[dict] = None,
):
    # Host-side only. Agent-issued commands go through
    # Execution.sandbox_tools.SandboxToolset.run_command instead.
    result = _run(command, cwd, timeout, env)
    return result, {"cwd": cwd}


@tool("run_tests")
def run_tests(
    command: str = "pytest -q",
    cwd: Optional[str] = None,
    timeout: float = DEFAULT_TEST_TIMEOUT,
    env: Optional[dict] = None,
):
    result = _run(command, cwd, timeout, env)
    return result, {"cwd": cwd, "command": command}
