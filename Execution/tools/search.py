from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .base import ToolTimeoutError, tool


def _resolve_search_binary() -> list[str]:
    if shutil.which("rg"):
        return ["rg", "--line-number", "--no-heading", "--color=never"]
    if shutil.which("grep"):
        return ["grep", "-rn", "--color=never"]
    raise RuntimeError("Neither 'rg' nor 'grep' is available on this system")


@tool("search_code")
def search_code(
    query: str,
    path: str = ".",
    file_pattern: Optional[str] = None,
    case_sensitive: bool = False,
    max_results: int = 200,
    timeout: float = 20.0,
):
    if not Path(path).exists():
        raise FileNotFoundError(f"Search path does not exist: {path}")

    cmd = _resolve_search_binary()
    if not case_sensitive:
        cmd.append("-i")
    if file_pattern:
        cmd += (["--glob", file_pattern] if cmd[0] == "rg" else ["--include", file_pattern])
    cmd += [query, path]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeoutError(f"search_code timed out after {timeout}s") from exc

    # rg/grep exit code 1 means "no matches", not a failure.
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"search failed (exit {proc.returncode}): {proc.stderr.strip()}")

    matches = []
    for line in proc.stdout.splitlines()[:max_results]:
        file_part, _, rest = line.partition(":")
        line_no, _, content = rest.partition(":")
        matches.append(
            {
                "file": file_part,
                "line_number": int(line_no) if line_no.isdigit() else None,
                "line": content,
            }
        )

    return matches, {"query": query, "path": path, "match_count": len(matches)}
