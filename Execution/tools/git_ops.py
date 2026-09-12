from __future__ import annotations

import subprocess
from typing import Optional

from .base import ToolTimeoutError, tool

DEFAULT_GIT_TIMEOUT = 15.0


@tool("git_diff")
def git_diff(
    cwd: Optional[str] = None,
    staged: bool = False,
    path: Optional[str] = None,
    timeout: float = DEFAULT_GIT_TIMEOUT,
):
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")
    if path:
        cmd += ["--", path]

    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeoutError(f"git_diff timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed or not on PATH") from exc

    if proc.returncode != 0:
        raise RuntimeError(f"git diff failed (exit {proc.returncode}): {proc.stderr.strip()}")

    return proc.stdout, {"staged": staged, "path": path, "has_changes": bool(proc.stdout.strip())}
