"""Shared plumbing for anything that runs a task in a sandbox.

Two consumers: `Tasks.verify`, which proves a task is real, and `Evaluation`,
which grades an agent's patch against it. They need the same three things -- a
local clone of the upstream repository, a pristine export of a commit, and a way
to turn a Benchmark into a number -- and those must behave identically in both,
or a task could verify under one set of rules and be graded under another.

All git work happens on the host and produces plain directories. Nothing here
executes repository code; the container is where that happens, always.
"""
from __future__ import annotations

import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .schema import Benchmark

DEFAULT_REPO_CACHE = Path(tempfile.gettempdir()) / "agent-runtime-task-repos"


def git(repo: Path, *args: str, binary: bool = False):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    return result.stdout if binary else result.stdout.decode("utf-8", "replace")


def ensure_repo(
    repository: str,
    cache_dir: Path = DEFAULT_REPO_CACHE,
    commits: Sequence[str] = (),
    quiet: bool = False,
) -> Path:
    """Clone the upstream repo once and reuse it across tasks and re-runs."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / re.sub(r"[^A-Za-z0-9_.-]", "_", repository)

    if not (path / ".git").exists():
        if not quiet:
            print(f"  cloning {repository} (one time, into {path})")
        subprocess.run(["git", "clone", "--quiet", repository, str(path)], check=True)

    missing = [
        commit
        for commit in commits
        if subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
        ).returncode
        != 0
    ]
    if missing:
        subprocess.run(["git", "-C", str(path), "fetch", "--quiet", "--all", "--tags"], check=True)
    return path


def export_commit(repo: Path, commit: str, dest: Path) -> Path:
    """Materialise a commit as a plain directory -- no .git, nothing to leak."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = git(repo, "archive", commit, binary=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    return dest


def file_at_commit(repo: Path, commit: str, path: str) -> bytes:
    return git(repo, "show", f"{commit}:{path}", binary=True)


def tail(result, limit: int = 400) -> str:
    """The last of a command's combined output, on one line -- what a person reads
    first when a stage fails."""
    text = (result.stdout or "") + (result.stderr or "")
    return text.strip()[-limit:].replace("\n", " | ")


def measure_benchmark(
    sandbox, benchmark: Benchmark, timeout: float
) -> Tuple[Optional[float], str]:
    """Median of benchmark.runs, so one scheduling hiccup can't decide a verdict."""
    values: List[float] = []
    for _ in range(benchmark.runs):
        result = sandbox.run_command(benchmark.command, timeout=timeout)
        if not result.success:
            return None, tail(result)
        try:
            values.append(float(result.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            return None, f"benchmark printed no float: {result.stdout.strip()[-200:]!r}"
    return statistics.median(values), ""
