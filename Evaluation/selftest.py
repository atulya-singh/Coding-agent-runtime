"""Prove the evaluation pipeline itself is correct, before trusting a verdict from it.

A grader is only as good as its calibration, so this checks it from both ends
against inputs whose correct grade is already known:

  the reference patch -- the real upstream fix -- must grade SOLVED on every task
  the empty patch     -- an agent that did nothing -- must grade unsolved on every
                         task, with no partial credit

If the first fails, the pipeline (or the task) is broken and no agent could ever
pass. If the second fails, the pipeline hands out marks for nothing, and every
number it produces is inflated. Both run through exactly the code path that
grades a real agent -- same containers, same stages, same thresholds.

The offline half needs no Docker and takes under a second: it pins down patch
construction, pytest-output parsing, the static checks, the outcome rules and the
metric arithmetic, which is where a silent error would be hardest to notice.

    python -m Evaluation.selftest              # offline checks, then all tasks
    python -m Evaluation.selftest --offline    # no Docker, no network
    python -m Evaluation.selftest requests-006-lookupdict-attribute-tests
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from Tasks.harness import DEFAULT_REPO_CACHE, ensure_repo, export_commit, file_at_commit
from Tasks.loader import discover_task_dirs, load_task
from Tasks.schema import TaskCategory

from . import static_checks as checks
from .evaluator import EvaluationConfig, Evaluator
from .metrics import aggregate, format_report
from .patch import Patch, PatchError, apply_patch, collect_patch, diff_trees
from .pytest_report import parse
from .reference import reference_patch
from .result import Outcome, Stage, StageResult, StageStatus, credit_for, decide_outcome
from .result import EvaluationResult

DATASET_DIR = Path(__file__).parent.parent / "Tasks" / "dataset"


class Checks:
    """Collects pass/fail lines so one failure doesn't hide the rest."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        self.results.append((name, bool(condition), detail))
        marker = "PASS" if condition else "FAIL"
        print(f"  {marker}  {name}")
        if not condition and detail:
            print(f"        {detail}")
        return bool(condition)

    def equal(self, name: str, actual, expected) -> bool:
        return self.check(name, actual == expected, f"expected {expected!r}, got {actual!r}")

    def raises(self, name: str, exception, call: Callable) -> bool:
        try:
            call()
        except exception as exc:
            return self.check(name, True, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self.check(name, False, f"raised {type(exc).__name__} instead: {exc}")
        return self.check(name, False, f"did not raise {exception.__name__}")

    @property
    def failures(self) -> List[Tuple[str, bool, str]]:
        return [entry for entry in self.results if not entry[1]]


# ---- offline: the parts that must be right before any container runs -------


def check_pytest_parsing(c: Checks) -> None:
    print("\n--- pytest output parsing")

    quiet = parse("..F\n1 failed, 2 passed in 0.05s")
    c.check("reads a -q summary line", quiet is not None)
    if quiet:
        c.equal("  counts failures", quiet.failed, 1)
        c.equal("  counts passes", quiet.passed, 2)
        c.equal("  grades only decided tests", quiet.graded, 3)
        c.check("  pass rate", abs((quiet.pass_rate or 0) - 2 / 3) < 1e-9)

    verbose = parse("=========== 1 failed, 24 passed, 2 warnings in 1.13s ============")
    c.check("reads a padded summary line", verbose is not None and verbose.passed == 24)

    selected = parse("1 passed, 40 deselected in 0.31s")
    c.check(
        "excludes deselected tests from the denominator",
        selected is not None and selected.graded == 1 and selected.deselected == 40,
    )

    skipped = parse("3 passed, 2 skipped in 0.10s")
    c.check(
        "excludes skips from the denominator, so skipping cannot raise a score",
        skipped is not None and skipped.graded == 3 and (skipped.pass_rate or 0) == 1.0,
    )

    empty = parse("no tests ran in 0.01s")
    c.check("recognises a run that matched nothing", empty is not None and empty.collected_nothing)

    errored = parse("!!!! Interrupted: 1 error during collection !!!!\n1 error in 0.05s")
    c.check("counts collection errors", errored is not None and errored.errors == 1)

    c.check("returns None when there is no summary", parse("Traceback ...") is None)
    c.check(
        "ignores a traceback line that merely says 'passed'",
        parse("assert 'passed' in body\nTraceback (most recent call last):") is None,
    )

    twice = parse("1 failed, 1 passed in 0.1s\n\n5 passed in 0.2s")
    c.check("takes the last summary when a command runs pytest twice",
            twice is not None and twice.passed == 5 and twice.failed == 0)


def check_patch_roundtrip(c: Checks) -> None:
    print("\n--- patch construction and application")

    root = Path(tempfile.mkdtemp(prefix="selftest-patch-"))
    try:
        base = root / "base"
        (base / "src").mkdir(parents=True)
        (base / "src" / "app.py").write_text("def add(a, b):\n    return a - b\n")
        (base / "src" / "gone.py").write_text("obsolete = True\n")
        (base / "README.md").write_text("hello\n")

        candidate = root / "candidate"
        shutil.copytree(base, candidate)
        (candidate / "src" / "app.py").write_text("def add(a, b):\n    return a + b\n")
        (candidate / "src" / "gone.py").unlink()
        (candidate / "src" / "new.py").write_text("VALUE = 1\n")
        # Build debris that setup and pytest leave behind, which must not appear.
        (candidate / "src" / "__pycache__").mkdir()
        (candidate / "src" / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\x00\x01")
        (candidate / "pkg.egg-info").mkdir()
        (candidate / "pkg.egg-info" / "PKG-INFO").write_text("Name: pkg\n")

        patch = diff_trees(base, candidate)
        c.equal(
            "diff lists exactly the files that changed, and nothing else",
            patch.files,
            ["src/app.py", "src/gone.py", "src/new.py"],
        )
        c.check("modification is in the diff", "return a + b" in patch.text)
        c.check("deletion is in the diff", "deleted file" in patch.text)
        c.check("addition is in the diff", "new file" in patch.text)
        c.check("build debris is excluded", "__pycache__" not in patch.text and "egg-info" not in patch.text)
        c.check("README was unchanged, so it carries no hunk", "diff --git a/README.md" not in patch.text)
        c.equal("counts added lines", patch.lines_added, 2)

        replay = root / "replay"
        shutil.copytree(base, replay)
        apply_patch(patch, replay)
        c.equal(
            "applying to a clean base reproduces the candidate tree",
            (replay / "src" / "app.py").read_text(),
            "def add(a, b):\n    return a + b\n",
        )
        c.check("applying replays the deletion", not (replay / "src" / "gone.py").exists())
        c.check("applying replays the addition", (replay / "src" / "new.py").exists())

        identical = root / "identical"
        shutil.copytree(base, identical)
        c.check("an unchanged tree produces an empty patch", diff_trees(base, identical).is_empty)

        escape = Patch.from_text(
            "diff --git a/../../etc/passwd b/../../etc/passwd\n"
            "--- a/../../etc/passwd\n+++ b/../../etc/passwd\n"
            "@@ -0,0 +1 @@\n+root::0:0::/:/bin/sh\n"
        )
        c.raises(
            "a patch writing outside the repository is refused",
            PatchError,
            lambda: apply_patch(escape, replay),
        )

        gitdir = Patch.from_text(
            "diff --git a/.git/hooks/pre-commit b/.git/hooks/pre-commit\n"
            "--- /dev/null\n+++ b/.git/hooks/pre-commit\n@@ -0,0 +1 @@\n+#!/bin/sh\n"
        )
        c.raises(
            "a patch writing into .git is refused",
            PatchError,
            lambda: apply_patch(gitdir, replay),
        )

        stale = Patch.from_text(
            "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n"
            "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a * b\n+    return a + b\n"
        )
        fresh = root / "fresh"
        shutil.copytree(base, fresh)
        c.raises(
            "a patch against the wrong base does not apply silently",
            PatchError,
            lambda: apply_patch(stale, fresh),
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_static_checks(c: Checks) -> None:
    print("\n--- static checks")

    patch = Patch.from_text(
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -10,3 +10,7 @@ def handler():\n"
        " context = build()\n"
        "+import pdb; pdb.set_trace()\n"
        "+<<<<<<< HEAD\n"
        "+value = 1\n"
        " return context\n"
    )
    findings = checks.inspect_patch(patch)
    kinds = {finding.check for finding in findings}
    c.check("flags a left-behind debugger", "debugger_statement" in kinds)
    c.check("flags an unresolved conflict marker", "conflict_marker" in kinds)

    debugger = next(f for f in findings if f.check == "debugger_statement")
    c.equal("reports the file", debugger.path, "src/app.py")
    c.equal("reports the line number in the new file", debugger.line, 11)

    clean = Patch.from_text(
        "diff --git a/docs/guide.md b/docs/guide.md\n"
        "--- a/docs/guide.md\n"
        "+++ b/docs/guide.md\n"
        "@@ -1,2 +1,4 @@\n"
        " Title\n"
        "+=======\n"
        "+Set a breakpoint in your editor to inspect this.\n"
        "+Call `pdb` from the shell if you prefer.\n"
    )
    c.equal(
        "does not flag prose that merely mentions debuggers, or a '=======' rule",
        checks.errors(checks.inspect_patch(clean)),
        [],
    )

    overwritten = checks.inspect_patch(
        Patch.from_text(
            "diff --git a/tests/test_utils.py b/tests/test_utils.py\n"
            "--- a/tests/test_utils.py\n+++ b/tests/test_utils.py\n"
            "@@ -1 +1 @@\n-assert broken()\n+assert True\n"
        ),
        protected_paths=["tests/test_utils.py"],
    )
    c.check(
        "notes when the agent edited a test file that grading restores",
        any(f.check == "modified_graded_test_file" for f in overwritten),
    )
    c.equal("but does not fail the stage for it", checks.errors(overwritten), [])

    c.check(
        "notes an empty patch",
        any(f.check == "empty_patch" for f in checks.inspect_patch(Patch.empty())),
    )


def check_partial_credit(c: Checks) -> None:
    print("\n--- partial credit against the baseline")

    c.equal("with nothing already passing, credit is the raw rate", credit_for(0.6, 0.0), 0.6)
    c.equal("matching the base earns nothing", credit_for(8 / 13, 8 / 13), 0.0)
    c.equal("passing everything is full credit whatever the base scored", credit_for(1.0, 8 / 13), 1.0)
    c.check(
        "closing half the remaining gap earns half the credit",
        abs(credit_for(0.5, 0.0) - 0.5) < 1e-9
        and abs(credit_for(11 / 13, 9 / 13) - 0.5) < 1e-9,
    )
    c.equal("scoring below the base cannot go negative", credit_for(0.2, 0.5), 0.0)
    c.equal(
        "a sliver of credit from the baseline's own rounding is not progress",
        credit_for(1 / 12, 0.0833),
        0.0,
    )
    c.check(
        "but the credit of one real test out of a suite still counts",
        credit_for(2 / 12, 0.0833) > 0.05,
        f"got {credit_for(2 / 12, 0.0833)}",
    )


def check_outcome_rules(c: Checks) -> None:
    print("\n--- outcome rules")

    def stage(name: Stage, status: StageStatus, score: float = 0.0) -> StageResult:
        return StageResult(stage=name, status=status, score=score)

    passed = StageStatus.PASSED
    failed = StageStatus.FAILED

    solved = [
        stage(Stage.PATCH, passed, 1.0),
        stage(Stage.SETUP, passed, 1.0),
        stage(Stage.BUILD, passed, 1.0),
        stage(Stage.PUBLIC_TESTS, passed, 1.0),
        stage(Stage.HIDDEN_TESTS, passed, 1.0),
        stage(Stage.STATIC_CHECKS, passed, 1.0),
    ]
    c.equal("everything green is solved", decide_outcome(TaskCategory.BUG_FIX, solved, False), Outcome.SOLVED)

    broke_regression = solved + [stage(Stage.REGRESSION_TESTS, failed)]
    c.equal(
        "passing the hidden tests while breaking the regression suite is not solved",
        decide_outcome(TaskCategory.API_CHANGE, broke_regression, False),
        Outcome.PARTIAL,
    )

    debris = [s for s in solved if s.stage is not Stage.STATIC_CHECKS] + [
        stage(Stage.STATIC_CHECKS, failed)
    ]
    c.equal(
        "a static-check failure blocks a solve",
        decide_outcome(TaskCategory.BUG_FIX, debris, False),
        Outcome.PARTIAL,
    )

    half = [s for s in solved if s.stage is not Stage.HIDDEN_TESTS] + [
        stage(Stage.HIDDEN_TESTS, failed, 0.6)
    ]
    c.equal(
        "some hidden tests passing earns partial, not success",
        decide_outcome(TaskCategory.BUG_FIX, half, False),
        Outcome.PARTIAL,
    )

    none = [s for s in solved if s.stage is not Stage.HIDDEN_TESTS] + [
        stage(Stage.HIDDEN_TESTS, failed, 0.0)
    ]
    c.equal("no hidden tests passing is a failure",
            decide_outcome(TaskCategory.BUG_FIX, none, False), Outcome.FAILED)

    # The reason regression tests are gates and not achievements: without that
    # split, doing nothing to a refactor task would score 50%.
    idle_refactor = [
        stage(Stage.PATCH, passed, 1.0),
        stage(Stage.SETUP, passed, 1.0),
        stage(Stage.BUILD, passed, 1.0),
        stage(Stage.HIDDEN_TESTS, failed, 0.0),
        stage(Stage.REGRESSION_TESTS, passed, 1.0),
        stage(Stage.STATIC_CHECKS, passed, 1.0),
    ]
    c.equal(
        "an untouched refactor task earns nothing for its regression suite still passing",
        decide_outcome(TaskCategory.REFACTOR, idle_refactor, True),
        Outcome.FAILED,
    )

    c.equal(
        "a patch that will not apply is reported as such",
        decide_outcome(TaskCategory.BUG_FIX, [stage(Stage.PATCH, failed)], False),
        Outcome.PATCH_FAILED,
    )
    c.equal(
        "a patch that breaks the install is a build failure",
        decide_outcome(
            TaskCategory.BUG_FIX,
            [stage(Stage.PATCH, passed, 1.0), stage(Stage.SETUP, failed)],
            False,
        ),
        Outcome.BUILD_FAILED,
    )
    c.equal(
        "the same failure with no patch is the environment's fault, not the agent's",
        decide_outcome(
            TaskCategory.BUG_FIX,
            [stage(Stage.PATCH, passed, 1.0), stage(Stage.SETUP, failed)],
            True,
        ),
        Outcome.INFRASTRUCTURE_ERROR,
    )
    c.equal(
        "a category with nothing to prove is never solved by default",
        decide_outcome(TaskCategory.PERFORMANCE, solved, False),
        Outcome.FAILED,
    )


def check_metrics(c: Checks) -> None:
    print("\n--- metrics")

    def result(task_id, category, outcome, stages) -> EvaluationResult:
        return EvaluationResult(task_id=task_id, category=category, outcome=outcome, stages=stages)

    from .pytest_report import TestCounts

    good = [
        StageResult(Stage.PATCH, StageStatus.PASSED, score=1.0),
        StageResult(Stage.SETUP, StageStatus.PASSED, score=1.0),
        StageResult(Stage.BUILD, StageStatus.PASSED, score=1.0),
        StageResult(Stage.HIDDEN_TESTS, StageStatus.PASSED, score=1.0, tests=TestCounts(passed=4)),
    ]
    partial = [
        StageResult(Stage.PATCH, StageStatus.PASSED, score=1.0),
        StageResult(Stage.SETUP, StageStatus.PASSED, score=1.0),
        StageResult(Stage.BUILD, StageStatus.PASSED, score=1.0),
        StageResult(
            Stage.HIDDEN_TESTS, StageStatus.FAILED, score=0.5, tests=TestCounts(passed=2, failed=2)
        ),
    ]
    broken = [
        StageResult(Stage.PATCH, StageStatus.PASSED, score=1.0),
        StageResult(Stage.SETUP, StageStatus.ERROR, detail="docker daemon not running"),
    ]

    results = [
        result("a", TaskCategory.BUG_FIX, Outcome.SOLVED, good),
        result("b", TaskCategory.BUG_FIX, Outcome.PARTIAL, partial),
        result("c", TaskCategory.FEATURE, Outcome.INFRASTRUCTURE_ERROR, broken),
    ]
    metrics = aggregate(results)

    c.equal("infrastructure errors leave the denominator", metrics.evaluable, 2)
    c.equal("task success rate", metrics.task_success_rate, 0.5)
    c.check(
        "hidden test pass rate pools every test (6 of 8)",
        abs((metrics.hidden_test_pass_rate or 0) - 0.75) < 1e-9,
    )
    c.check(
        "macro pass rate weights tasks equally instead (mean of 1.0 and 0.5)",
        abs((metrics.hidden_test_pass_rate_macro or 0) - 0.75) < 1e-9,
    )
    c.equal("build success rate", metrics.build_success_rate, 1.0)
    c.check(
        "partial success sits above the binary success rate",
        (metrics.partial_success or 0) == 0.75,
        f"got {metrics.partial_success}",
    )
    c.check(
        "the infrastructure error is still reported, not hidden",
        metrics.infrastructure_errors == 1 and metrics.total == 3,
    )


def run_offline(c: Checks) -> None:
    check_pytest_parsing(c)
    check_patch_roundtrip(c)
    check_static_checks(c)
    check_partial_credit(c)
    check_outcome_rules(c)
    check_metrics(c)


# ---- end to end: the real pipeline, in real containers ---------------------


def check_collect_from_sandbox(c: Checks, task, task_dir: Path, repo: Path, config) -> None:
    """The whole loop, once: work in a container, collect the patch, grade it elsewhere.

    Every other end-to-end check starts from a patch built host-side, which
    leaves `collect_patch` -- the one piece an agent run will actually depend on
    -- unexercised. Here the change is made the way an agent makes it, through
    SandboxToolset inside the sandbox, and the diff is lifted out of the live
    container.
    """
    from Sandbox import Sandbox
    from Execution.sandbox_tools import SandboxToolset

    print(f"\n--- collecting a patch from a live sandbox  [{task.task_id}]")
    workspace = Path(tempfile.mkdtemp(prefix="selftest-collect-"))
    try:
        base = export_commit(repo, task.commit, workspace / "base")
        with Sandbox(
            task_id=f"selftest-collect-{task.task_id}",
            config=task.resource_limit.to_sandbox_config(),
        ) as sandbox:
            sandbox.initialize(str(base))
            tools = SandboxToolset(sandbox)

            for rel_path in task.reference_paths:
                content = file_at_commit(repo, task.reference_commit, rel_path).decode("utf-8")
                written = tools.write_file(rel_path, content)
                c.check(f"agent tool wrote {rel_path}", written.success, str(written.error))

            # Byte-compiling leaves __pycache__ behind, exactly as a real run
            # would; none of it may end up in the graded diff.
            sandbox.run_command("python -m compileall -q src || true", timeout=120)
            patch = collect_patch(sandbox, base)

        c.equal(
            "the collected patch names only the files the agent edited",
            patch.files,
            sorted(task.reference_paths),  # Patch.files is sorted, task order is not
        )
        c.check(
            "caches and byte-code the run left behind are not in the patch",
            "__pycache__" not in patch.text and ".pyc" not in patch.text,
        )

        graded = Evaluator(task, task_dir, repo=repo, config=config).evaluate(patch)
        c.check(
            "a patch collected from a live sandbox grades SOLVED in a fresh one",
            graded.outcome is Outcome.SOLVED,
            _explain(graded),
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def run_end_to_end(
    c: Checks,
    task_ids: List[str],
    dataset: Path,
    repo_cache: Path,
    include_empty: bool,
) -> List[EvaluationResult]:
    tasks = [
        (load_task(task_dir), task_dir)
        for task_dir in discover_task_dirs(dataset)
    ]
    if task_ids:
        tasks = [(task, path) for task, path in tasks if task.task_id in task_ids]
        unknown = set(task_ids) - {task.task_id for task, _ in tasks}
        if unknown:
            c.check(f"unknown task_id: {', '.join(sorted(unknown))}", False)

    config = EvaluationConfig(repo_cache=repo_cache, quiet=True)
    repos: dict = {}
    collected: List[EvaluationResult] = []
    checked_collection = False

    for task, task_dir in tasks:
        print(f"\n--- {task.task_id}  [{task.category.value}]")
        if task.repository not in repos:
            commits = [t.commit for t, _ in tasks] + [t.reference_commit for t, _ in tasks]
            repos[task.repository] = ensure_repo(task.repository, repo_cache, commits, quiet=True)
        repo = repos[task.repository]
        evaluator = Evaluator(task, task_dir, repo=repo, config=config)

        # Once is enough: this proves the collection path, not the task.
        if not checked_collection:
            check_collect_from_sandbox(c, task, task_dir, repo, config)
            checked_collection = True

        solution = reference_patch(task, repo)
        c.check(
            f"{task.task_id}: the reference solution is a real diff",
            not solution.is_empty,
            "reference_paths produced no change against the base commit",
        )
        graded = evaluator.evaluate(solution)
        collected.append(graded)
        c.check(
            f"{task.task_id}: the real upstream fix grades SOLVED",
            graded.outcome is Outcome.SOLVED,
            _explain(graded),
        )

        if include_empty:
            nothing = evaluator.evaluate(Patch.empty())
            c.check(
                f"{task.task_id}: changing nothing does not grade SOLVED",
                nothing.outcome is not Outcome.SOLVED,
                _explain(nothing),
            )
            c.check(
                f"{task.task_id}: changing nothing earns no partial credit",
                nothing.partial_success == 0.0,
                f"scored {nothing.partial_success:.0%}: {_explain(nothing)}",
            )

    return collected


def _explain(result: EvaluationResult) -> str:
    lines = [f"outcome={result.outcome.value}"]
    for stage in result.stages:
        if stage.status in (StageStatus.FAILED, StageStatus.ERROR):
            lines.append(f"{stage.stage.value}: {stage.detail[:200]}")
    return " | ".join(lines)


# ---- entry point -----------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m Evaluation.selftest", description=__doc__.splitlines()[0]
    )
    parser.add_argument("task_ids", nargs="*", help="end-to-end check only these tasks")
    parser.add_argument("--offline", action="store_true", help="skip everything needing Docker")
    parser.add_argument(
        "--no-empty",
        action="store_true",
        help="skip the empty-patch floor (halves the container time)",
    )
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--repo-cache", default=str(DEFAULT_REPO_CACHE))
    args = parser.parse_args(argv)

    for name in ("sandbox", "sandbox.docker", "agent.tools", "evaluation"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    c = Checks()
    print("=== offline checks (no Docker)")
    run_offline(c)

    results: List[EvaluationResult] = []
    if not args.offline:
        print("\n=== end-to-end checks (one container per evaluation)")
        results = run_end_to_end(
            c,
            args.task_ids,
            Path(args.dataset),
            Path(args.repo_cache),
            include_empty=not args.no_empty,
        )

    if results:
        print(format_report(results))

    total = len(c.results)
    failed = len(c.failures)
    print(f"\n{total - failed}/{total} checks passed")
    if failed:
        for name, _, detail in c.failures:
            print(f"  FAIL {name}" + (f"\n       {detail}" if detail else ""))
        return 1
    print("the evaluation pipeline grades the known-good solution as solved and no-change as not")
    return 0


if __name__ == "__main__":
    sys.exit(main())
