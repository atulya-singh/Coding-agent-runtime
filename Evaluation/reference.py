"""Reconstruct the known-good patch for a task, from the real upstream commit.

This is the fixed point the evaluator is calibrated against. A grading pipeline
with no known-correct input cannot be trusted: if the reference patch does not
grade as SOLVED, the bug is in the pipeline or the task, not in some future
agent. It is also what makes a per-task upper bound available for free -- a task
whose own upstream fix does not pass its own criteria should never be shipped.

Never given to an agent, and never consulted during grading.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from Tasks.harness import export_commit, file_at_commit
from Tasks.schema import Task

from .patch import Patch, diff_trees


def reference_patch(task: Task, repo: Path) -> Patch:
    """The diff from the task's base commit to the real upstream solution.

    Built by exporting the base twice and overlaying only `reference_paths` on
    one copy, rather than diffing the two upstream commits directly: the real
    commit may also touch files the task deliberately leaves out (its own test
    file, a changelog entry), and those must not appear in the patch under test.
    """
    if not task.reference_commit or not task.reference_paths:
        raise ValueError(
            f"{task.task_id} has no reference solution recorded; "
            f"reference_commit/reference_paths are required"
        )

    workspace = Path(tempfile.mkdtemp(prefix=f"reference-{task.task_id}-"))
    try:
        base = export_commit(repo, task.commit, workspace / "base")
        candidate = export_commit(repo, task.commit, workspace / "candidate")

        for rel_path in task.reference_paths:
            target = candidate / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file_at_commit(repo, task.reference_commit, rel_path))

        return diff_trees(base, candidate)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
