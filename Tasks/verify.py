"""Verify every task in the dataset against a real sandbox.

`python -m Tasks` only checks that task.yaml is well-formed. This checks the
thing that actually matters: that each task is a real, solvable, objectively
graded problem. For every task it proves the grading criterion separates the
pre-change state from the real upstream fix -- so a task can never quietly
become one that is already satisfied before the agent has done anything, and a
hidden test can never quietly become one that nothing can pass.

Each category is checked on its own terms, matching CATEGORY_CONTRACTS:

  bug_fix / feature / refactor / api_change / dependency_migration
      hidden_tests must FAIL on the base commit and PASS with the reference
      solution applied.
  refactor / api_change / performance
      regression_tests must PASS in both states -- otherwise "delete the module"
      would satisfy a structural or timing criterion.
  performance
      the benchmark ratio must sit above max_ratio on the base and at or below
      it with the reference applied.
  testing
      the pre-existing tests must SURVIVE every mutation (a mutation the old
      tests already catch proves nothing about the tests the agent writes), and
      the reference tests must pass on clean source and FAIL against every one.

Nothing here runs on the host: the repository is exported from git host-side,
but every command runs inside a resource-limited container, which is always
destroyed. Reference solutions are copied in file by file rather than checked
out in-container, so the task image is not required to ship git.

    python -m Tasks.verify                                  # every task
    python -m Tasks.verify requests-007-proxy-bypass-short-circuit
    python -m Tasks.verify --quiet --keep-logs /tmp/verify
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from Sandbox import Sandbox
from Sandbox.manager import REPO_DIR

from .loader import discover_task_dirs, load_task
from .schema import Benchmark, Task, TaskCategory

DATASET_DIR = Path(__file__).parent / "dataset"
DEFAULT_REPO_CACHE = Path(tempfile.gettempdir()) / "agent-runtime-task-repos"
SETUP_TIMEOUT = 900.0

# public_tests exist to give the agent a signal it can trust while working, so
# they should already pass on the pre-change state. dependency_migration is the
# one category where they cannot: the whole point of the task is that the
# environment is broken until the migration lands, so nothing even imports.
PUBLIC_TESTS_PASS_ON_BASE = {category: True for category in TaskCategory}
PUBLIC_TESTS_PASS_ON_BASE[TaskCategory.DEPENDENCY_MIGRATION] = False


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TaskReport:
    task_id: str
    category: str
    checks: List[Check] = field(default_factory=list)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]


# ---- host-side git helpers -------------------------------------------------


def _git(repo: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True
    )
    return result.stdout if binary else result.stdout.decode("utf-8", "replace")


def ensure_repo(repository: str, cache_dir: Path, commits: List[str]) -> Path:
    """Clone the upstream repo once and reuse it across tasks and re-runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / re.sub(r"[^A-Za-z0-9_.-]", "_", repository)
    if not (path / ".git").exists():
        print(f"  cloning {repository} (one time, into {path})")
        subprocess.run(["git", "clone", "--quiet", repository, str(path)], check=True)
    missing = [c for c in commits if subprocess.run(["git", "-C", str(path), "cat-file", "-e", f"{c}^{{commit}}"], capture_output=True).returncode != 0]
    if missing:
        subprocess.run(["git", "-C", str(path), "fetch", "--quiet", "--all", "--tags"], check=True)
    return path


def export_commit(repo: Path, commit: str, dest: Path) -> None:
    """Materialise a commit as a plain directory -- no .git, nothing to leak."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = _git(repo, "archive", commit, binary=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)


def file_at_commit(repo: Path, commit: str, path: str) -> bytes:
    return _git(repo, "show", f"{commit}:{path}", binary=True)


# ---- sandbox helpers -------------------------------------------------------


def _tail(result, limit: int = 400) -> str:
    text = (result.stdout or "") + (result.stderr or "")
    return text.strip()[-limit:].replace("\n", " | ")


def put_bytes(sandbox: Sandbox, content: bytes, repo_rel: str, staging: Path) -> None:
    local = staging / repo_rel
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(content)
    sandbox.put_file(str(local), f"{REPO_DIR}/{repo_rel}")


def put_local(sandbox: Sandbox, local: Path, repo_rel: str) -> None:
    sandbox.put_file(str(local), f"{REPO_DIR}/{repo_rel}")


def measure(sandbox: Sandbox, benchmark: Benchmark, timeout: float) -> Tuple[Optional[float], str]:
    """Median of benchmark.runs, so one scheduling hiccup can't decide a verdict."""
    values = []
    for _ in range(benchmark.runs):
        result = sandbox.run_command(benchmark.command, timeout=timeout)
        if not result.success:
            return None, _tail(result)
        try:
            values.append(float(result.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            return None, f"benchmark printed no float: {result.stdout.strip()[-200:]!r}"
    return statistics.median(values), ""


# ---- the per-task procedure ------------------------------------------------


def verify_task(task: Task, task_dir: Path, repo: Path, keep_logs: Optional[Path]) -> TaskReport:
    report = TaskReport(task_id=task.task_id, category=task.category.value)
    add = lambda name, passed, detail="": report.checks.append(Check(name, passed, detail))

    workspace = Path(tempfile.mkdtemp(prefix=f"verify-{task.task_id}-"))
    base_dir = workspace / "base"
    staging = workspace / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    timeout = float(task.resource_limit.timeout_seconds)

    try:
        export_commit(repo, task.commit, base_dir)
        config = task.resource_limit.to_sandbox_config()

        with Sandbox(task_id=f"verify-{task.task_id}", config=config) as sandbox:
            sandbox.initialize(str(base_dir))

            for step in task.setup:
                result = sandbox.run_command(step, timeout=SETUP_TIMEOUT)
                if not result.success:
                    add("setup", False, f"`{step}` exited {result.exit_code}: {_tail(result)}")
                    return report
            add("setup", True)

            # ---- the pre-change state -------------------------------------
            if task.public_tests:
                result = sandbox.run_tests(task.public_tests, timeout=timeout)
                expected = PUBLIC_TESTS_PASS_ON_BASE[task.category]
                add(
                    f"public_tests {'pass' if expected else 'fail'} on base",
                    result.success is expected,
                    _tail(result),
                )

            if task.regression_tests:
                result = sandbox.run_tests(task.regression_tests, timeout=timeout)
                add("regression_tests pass on base", result.success, _tail(result))

            for rel_path in task.hidden_test_files:
                put_local(sandbox, task_dir / "hidden" / rel_path, rel_path)

            if task.hidden_tests:
                result = sandbox.run_tests(task.hidden_tests, timeout=timeout)
                add(
                    "hidden_tests FAIL on base",
                    not result.success,
                    "hidden tests already pass before the change" if result.success else _tail(result),
                )

            if task.benchmark:
                ratio, error = measure(sandbox, task.benchmark, timeout)
                add(
                    f"benchmark on base is above max_ratio ({task.benchmark.max_ratio})",
                    ratio is not None and ratio > task.benchmark.max_ratio,
                    error or f"measured {ratio}",
                )

            # A mutation is only evidence about the agent's tests if the tests
            # that already exist do not catch it.
            for mutation in task.mutations:
                for rel_path in mutation.files:
                    put_local(sandbox, task_dir / "hidden" / "mutations" / mutation.name / rel_path, rel_path)
                result = sandbox.run_tests(task.agent_tests, timeout=timeout)
                add(
                    f"mutation '{mutation.name}' survives the pre-existing tests",
                    result.success,
                    "existing tests already catch it, so it grades nothing" if not result.success else "",
                )
                for rel_path in mutation.files:
                    put_bytes(sandbox, (base_dir / rel_path).read_bytes(), rel_path, staging)

            # ---- apply the real upstream solution --------------------------
            for rel_path in task.reference_paths:
                put_bytes(sandbox, file_at_commit(repo, task.reference_commit, rel_path), rel_path, staging)

            if task.public_tests:
                result = sandbox.run_tests(task.public_tests, timeout=timeout)
                add("public_tests pass with reference", result.success, _tail(result))

            if task.hidden_tests:
                result = sandbox.run_tests(task.hidden_tests, timeout=timeout)
                add("hidden_tests PASS with reference", result.success, _tail(result))

            if task.regression_tests:
                result = sandbox.run_tests(task.regression_tests, timeout=timeout)
                add("regression_tests pass with reference", result.success, _tail(result))

            if task.benchmark:
                ratio, error = measure(sandbox, task.benchmark, timeout)
                add(
                    f"benchmark with reference is at or below max_ratio ({task.benchmark.max_ratio})",
                    ratio is not None and ratio <= task.benchmark.max_ratio,
                    error or f"measured {ratio}",
                )

            if task.mutations:
                result = sandbox.run_tests(task.agent_tests, timeout=timeout)
                add("reference tests pass on clean source", result.success, _tail(result))
                for mutation in task.mutations:
                    for rel_path in mutation.files:
                        put_local(sandbox, task_dir / "hidden" / "mutations" / mutation.name / rel_path, rel_path)
                    result = sandbox.run_tests(task.agent_tests, timeout=timeout)
                    add(
                        f"reference tests KILL mutation '{mutation.name}'",
                        not result.success,
                        "mutation went undetected" if result.success else "",
                    )
                    for rel_path in mutation.files:
                        put_bytes(sandbox, (base_dir / rel_path).read_bytes(), rel_path, staging)

            if keep_logs:
                sandbox.collect_results(str(Path(keep_logs) / task.task_id))
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    return report


# ---- entry point -----------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m Tasks.verify", description=__doc__.splitlines()[0])
    parser.add_argument("task_ids", nargs="*", help="verify only these task_ids (default: all)")
    parser.add_argument("--dataset", default=str(DATASET_DIR), help="dataset directory")
    parser.add_argument("--repo-cache", default=str(DEFAULT_REPO_CACHE), help="where upstream clones are kept")
    parser.add_argument("--keep-logs", metavar="DIR", help="collect each sandbox's results into DIR")
    parser.add_argument("--quiet", action="store_true", help="silence sandbox JSON log lines")
    args = parser.parse_args(argv)

    if args.quiet:
        for name in ("sandbox", "sandbox.docker", "agent.tools"):
            logging.getLogger(name).setLevel(logging.CRITICAL)

    tasks = []
    for task_dir in discover_task_dirs(Path(args.dataset)):
        task = load_task(task_dir)
        if not args.task_ids or task.task_id in args.task_ids:
            tasks.append((task, task_dir))

    unknown = set(args.task_ids) - {t.task_id for t, _ in tasks}
    if unknown:
        print(f"no such task_id: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    if not tasks:
        print(f"no tasks found in {args.dataset}", file=sys.stderr)
        return 2

    repos = {}
    reports = []
    for task, task_dir in tasks:
        print(f"\n=== {task.task_id}  [{task.category.value}]")
        if task.repository not in repos:
            commits = [t.commit for t, _ in tasks if t.repository == task.repository]
            commits += [t.reference_commit for t, _ in tasks if t.repository == task.repository]
            repos[task.repository] = ensure_repo(task.repository, Path(args.repo_cache), commits)

        report = verify_task(task, task_dir, repos[task.repository], args.keep_logs)
        reports.append(report)
        for check in report.checks:
            marker = "PASS" if check.passed else "FAIL"
            print(f"  {marker}  {check.name}")
            if not check.passed and check.detail:
                print(f"        {check.detail}")

    total = sum(len(r.checks) for r in reports)
    failed = sum(len(r.failures) for r in reports)
    print(f"\n{total - failed}/{total} checks passed across {len(reports)} tasks")
    if failed:
        for report in reports:
            if report.failures:
                print(f"  {report.task_id}: " + ", ".join(c.name for c in report.failures))
        return 1
    print("every task is solvable, and every grading criterion separates base from fix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
