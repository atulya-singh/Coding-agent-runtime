"""Phase 4: grade one patch against one task, objectively.

The pipeline is the one in plan.md, with the stages the Phase 3 grading contracts
require folded in at the point they belong:

    apply patch to a clean checkout
      -> setup            (for a Python repo, `pip install -e .` is the build)
      -> build            (does the patched tree run at all?)
      -> public tests     (the signal the agent was allowed to see)
      -> hidden tests     (overlaid from the task, over whatever the agent wrote)
      -> regression tests (what the change was not allowed to break)
      -> benchmark        (performance tasks)
      -> mutation tests   (testing tasks)
      -> static checks
      -> result

Three properties this is built to hold:

*Nothing is graded in the container the agent worked in.* That container has
whatever the agent installed, deleted or cached in it. Grading happens in a fresh
container built from a pristine export of the base commit with only the patch
applied, so the verdict depends on the diff and nothing else.

*Hidden tests are restored, not trusted.* Files listed in `hidden_test_files` are
copied over the agent's copies after the patch is applied. Editing a test file is
therefore not a way to pass; for `testing` tasks, where writing tests IS the job,
grading is by mutation instead and the agent's file is left alone.

*A broken runtime is not a failed agent* (Principle 5). Docker failing, an image
that will not pull, a registry outage during setup: those produce
`INFRASTRUCTURE_ERROR`, which the metrics exclude from the capability rates
rather than counting as something the model got wrong.
"""
from __future__ import annotations

import logging
import shlex
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from Sandbox import Sandbox
from Sandbox.manager import REPO_DIR
from Tasks.harness import (
    DEFAULT_REPO_CACHE,
    ensure_repo,
    export_commit,
    measure_benchmark,
    tail,
)
from Tasks.schema import Task

from . import static_checks as checks
from .patch import Patch, PatchError, apply_patch
from .pytest_report import parse_result
from .result import (
    EVALUATOR_VERSION,
    EvaluationResult,
    Outcome,
    Stage,
    StageResult,
    StageStatus,
    credit_for,
    decide_outcome,
    utc_now,
)

logger = logging.getLogger("evaluation")

DEFAULT_SETUP_TIMEOUT = 900.0
DEFAULT_BUILD_TIMEOUT = 300.0
DEFAULT_STATIC_CHECK_TIMEOUT = 300.0


@dataclass
class EvaluationConfig:
    repo_cache: Path = DEFAULT_REPO_CACHE
    setup_timeout: float = DEFAULT_SETUP_TIMEOUT
    build_timeout: float = DEFAULT_BUILD_TIMEOUT
    #: Where to leave the graded working tree and the sandbox run log. Off by
    #: default: it is a full copy of the repository per evaluation.
    artifacts_dir: Optional[Path] = None
    quiet: bool = False


class Evaluator:
    """Grades patches for one task. Reusable across many patches for that task,
    so a batch run clones and fetches the upstream repository only once."""

    def __init__(
        self,
        task: Task,
        task_dir: Path,
        repo: Optional[Path] = None,
        config: Optional[EvaluationConfig] = None,
    ):
        self.task = task
        self.task_dir = Path(task_dir)
        self.config = config or EvaluationConfig()
        self.repo = Path(repo) if repo else ensure_repo(
            task.repository, self.config.repo_cache, [task.commit], quiet=self.config.quiet
        )

    # ---- public API --------------------------------------------------------

    def evaluate(self, patch: Patch, agent: Optional[dict] = None) -> EvaluationResult:
        task = self.task
        started_at = utc_now()
        start = time.monotonic()
        run = _Run(self, patch)

        workspace = Path(tempfile.mkdtemp(prefix=f"evaluate-{task.task_id}-"))
        clean = workspace / "clean"
        try:
            run.stage_patch(clean)
            if run.aborted:
                return run.finish(started_at, start, agent)

            config = task.resource_limit.to_sandbox_config()
            with Sandbox(task_id=f"eval-{task.task_id}", config=config) as sandbox:
                sandbox.initialize(str(clean))
                run.sandbox = sandbox
                run.clean = clean

                run.stage_setup()
                run.stage_build()
                run.stage_public_tests()
                run.overlay_graded_files()
                run.stage_hidden_tests()
                run.stage_regression_tests()
                run.stage_benchmark()
                run.stage_mutation_tests()
                run.stage_static_checks()

                if self.config.artifacts_dir and sandbox.container_id:
                    sandbox.collect_results(
                        str(Path(self.config.artifacts_dir) / task.task_id)
                    )
        except Exception as exc:  # noqa: BLE001 -- any escape here is infrastructure
            run.record_infrastructure_error(exc)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

        return run.finish(started_at, start, agent)


@dataclass
class _Run:
    """One evaluation in progress. Holds the mutable bits so Evaluator stays
    reusable and thread-safe across concurrent tasks."""

    evaluator: Evaluator
    patch: Patch
    stages: List[StageResult] = field(default_factory=list)
    sandbox: Optional[Sandbox] = None
    clean: Optional[Path] = None
    aborted: bool = False

    @property
    def task(self) -> Task:
        return self.evaluator.task

    @property
    def timeout(self) -> float:
        return float(self.task.resource_limit.timeout_seconds)

    # ---- stage bookkeeping -------------------------------------------------

    def add(
        self,
        stage: Stage,
        status: StageStatus,
        duration_ms: float = 0.0,
        detail: str = "",
        score: Optional[float] = None,
        tests=None,
        **data,
    ) -> StageResult:
        result = StageResult(
            stage=stage,
            status=status,
            duration_ms=duration_ms,
            detail=detail,
            score=1.0 if score is None and status is StageStatus.PASSED else (score or 0.0),
            tests=tests,
            data=data,
        )
        self.stages.append(result)
        return result

    def skip(self, stage: Stage, reason: str) -> None:
        self.add(stage, StageStatus.SKIPPED, detail=reason)

    def record_infrastructure_error(self, exc: Exception) -> None:
        """Attribute the error to the first stage that never got to run, so the
        record says where the pipeline broke rather than where it last succeeded."""
        self.aborted = True
        done = {result.stage for result in self.stages}
        stage = next((candidate for candidate in Stage if candidate not in done), Stage.SETUP)
        self.add(stage, StageStatus.ERROR, detail=f"{type(exc).__name__}: {exc}")
        logger.exception("evaluation of %s failed with an infrastructure error", self.task.task_id)

    def _blocked(self, stage: Stage) -> bool:
        """True once the pipeline can no longer produce a meaningful result.

        Also covers the sandbox tearing itself down after a command timeout,
        which leaves no container to run the remaining stages in.
        """
        if self.aborted:
            self.skip(stage, "an earlier stage stopped the pipeline")
            return True
        if self.sandbox is not None and self.sandbox.container_id is None:
            self.skip(stage, "the sandbox was torn down (a previous command timed out)")
            return True
        return False

    # ---- running commands --------------------------------------------------

    def _timed(self, call: Callable):
        start = time.monotonic()
        result = call()
        return result, (time.monotonic() - start) * 1000

    def _put(self, local: Path, repo_rel: str) -> None:
        self.sandbox.put_file(str(local), f"{REPO_DIR}/{repo_rel}")

    def _run_test_stage(
        self, stage: Stage, command: str, what: str, baseline: float = 0.0
    ) -> Optional[StageResult]:
        """Run a test command and record it, with per-test counts where pytest
        gave us enough output to read them.

        `baseline` is the pass rate the untouched base commit already achieves.
        Credit is awarded for closing the gap to 1.0, not for the raw rate: a
        hidden suite that also covers the cases around the change starts well
        above zero, and paying for those would hand an agent that did nothing
        most of the marks. Raw counts are kept intact on the result -- the Hidden
        Test Pass Rate metric reports what actually passed.
        """
        result, duration = self._timed(
            lambda: self.sandbox.run_tests(command, timeout=self.timeout)
        )
        counts = parse_result(result)

        if result.timed_out:
            self.aborted = True
            return self.add(
                stage,
                StageStatus.FAILED,
                duration,
                detail=f"{what} timed out after {self.timeout}s",
                command=command,
            )

        # A command that matched no tests is never evidence of success: it means
        # the agent deleted or renamed what was being graded.
        collected_nothing = counts is not None and counts.collected_nothing
        passed = result.success and not collected_nothing

        raw = 1.0 if passed else 0.0
        if counts and counts.graded and not collected_nothing:
            raw = counts.pass_rate or 0.0
        score = credit_for(raw, baseline)

        return self.add(
            stage,
            StageStatus.PASSED if passed else StageStatus.FAILED,
            duration,
            detail="the command matched no tests" if collected_nothing else tail(result),
            score=score,
            tests=counts,
            command=command,
            exit_code=result.exit_code,
            raw_pass_rate=raw,
            baseline_pass_rate=baseline,
        )

    # ---- the stages --------------------------------------------------------

    def stage_patch(self, clean: Path) -> None:
        task = self.task
        start = time.monotonic()
        try:
            export_commit(self.evaluator.repo, task.commit, clean)
            apply_patch(self.patch, clean)
        except PatchError as exc:
            # The change was written against a tree that is not the task's base:
            # an agent failure, and a fatal one -- there is nothing to grade.
            self.aborted = True
            self.add(
                Stage.PATCH,
                StageStatus.FAILED,
                (time.monotonic() - start) * 1000,
                detail=str(exc),
                **self.patch.to_dict(),
            )
            return
        except Exception as exc:  # noqa: BLE001 -- git/export problems are ours
            self.aborted = True
            self.add(
                Stage.PATCH,
                StageStatus.ERROR,
                (time.monotonic() - start) * 1000,
                detail=f"could not prepare the clean checkout: {type(exc).__name__}: {exc}",
            )
            return

        self.add(
            Stage.PATCH,
            StageStatus.PASSED,
            (time.monotonic() - start) * 1000,
            detail=f"{len(self.patch.files)} file(s), +{self.patch.lines_added}/-{self.patch.lines_deleted}",
            **self.patch.to_dict(),
        )

    def stage_setup(self) -> None:
        if self._blocked(Stage.SETUP):
            return
        if not self.task.setup:
            self.add(Stage.SETUP, StageStatus.PASSED, detail="no setup steps")
            return

        start = time.monotonic()
        for step in self.task.setup:
            result = self.sandbox.run_command(step, timeout=self.evaluator.config.setup_timeout)
            if not result.success:
                self.aborted = True
                self.add(
                    Stage.SETUP,
                    StageStatus.FAILED,
                    (time.monotonic() - start) * 1000,
                    detail=f"`{step}` exited {result.exit_code}: {tail(result)}",
                    failed_step=step,
                )
                return
        self.add(Stage.SETUP, StageStatus.PASSED, (time.monotonic() - start) * 1000)

    def stage_build(self) -> None:
        if self._blocked(Stage.BUILD):
            return

        command = self.task.build or self._default_build_command()
        if not command:
            self.skip(Stage.BUILD, "nothing to build: no build command and no source files changed")
            return

        result, duration = self._timed(
            lambda: self.sandbox.run_command(command, timeout=self.evaluator.config.build_timeout)
        )
        if result.timed_out:
            self.aborted = True
        self.add(
            Stage.BUILD,
            StageStatus.PASSED if result.success else StageStatus.FAILED,
            duration,
            detail="" if result.success else tail(result),
            command=command,
        )

    def _default_build_command(self) -> str:
        """Byte-compile the Python files the patch touched.

        Weak compared with importing the package, which is why tasks should set
        `build:` explicitly -- but it needs no knowledge of the repository, and it
        does catch the common case of a patch that leaves a syntax error behind.
        """
        if self.task.language != "python":
            return ""
        sources = [path for path in self.patch.files if path.endswith(".py")]
        if not sources:
            return ""
        return "python -m compileall -q " + " ".join(shlex.quote(path) for path in sources)

    def stage_public_tests(self) -> None:
        if self._blocked(Stage.PUBLIC_TESTS):
            return
        if not self.task.public_tests:
            self.skip(Stage.PUBLIC_TESTS, "the task declares no public tests")
            return
        self._run_test_stage(Stage.PUBLIC_TESTS, self.task.public_tests, "public tests")

    def overlay_graded_files(self) -> None:
        """Copy every `hidden_test_files` entry over whatever the agent left behind.

        This is the step that makes "edit the test until it passes" a dead end.
        It runs after the public tests -- which are graded on the agent's own tree,
        so a divergence between the two is real information -- and before every
        stage that is graded, not only the hidden tests: a `performance` task
        delivers its benchmark script this way too, and would otherwise have
        nothing to measure.
        """
        if self.aborted or not self.sandbox or self.sandbox.container_id is None:
            return
        for rel_path in self.task.hidden_test_files:
            self._put(self.task_hidden(rel_path), rel_path)

    def stage_hidden_tests(self) -> None:
        if self._blocked(Stage.HIDDEN_TESTS):
            return
        if not self.task.hidden_tests:
            self.skip(
                Stage.HIDDEN_TESTS,
                "the task is not graded by hidden tests (see CATEGORY_CONTRACTS)",
            )
            return
        self._run_test_stage(
            Stage.HIDDEN_TESTS,
            self.task.hidden_tests,
            "hidden tests",
            baseline=self.task.baseline_hidden_pass_rate,
        )

    def stage_regression_tests(self) -> None:
        if self._blocked(Stage.REGRESSION_TESTS):
            return
        if not self.task.regression_tests:
            self.skip(Stage.REGRESSION_TESTS, "the task declares no regression tests")
            return
        self._run_test_stage(
            Stage.REGRESSION_TESTS, self.task.regression_tests, "regression tests"
        )

    def stage_benchmark(self) -> None:
        if self._blocked(Stage.BENCHMARK):
            return
        benchmark = self.task.benchmark
        if not benchmark:
            self.skip(Stage.BENCHMARK, "the task declares no benchmark")
            return

        (ratio, error), duration = self._timed(
            lambda: measure_benchmark(self.sandbox, benchmark, self.timeout)
        )
        if ratio is None:
            self.add(
                Stage.BENCHMARK,
                StageStatus.FAILED,
                duration,
                detail=f"the benchmark did not produce a measurement: {error}",
                command=benchmark.command,
            )
            return

        met = ratio <= benchmark.max_ratio
        self.add(
            Stage.BENCHMARK,
            StageStatus.PASSED if met else StageStatus.FAILED,
            duration,
            detail=(
                f"median ratio {ratio:.6g} over {benchmark.runs} runs "
                f"(threshold {benchmark.max_ratio}, unfixed base scores ~{benchmark.baseline_ratio})"
            ),
            score=1.0 if met else 0.0,
            command=benchmark.command,
            ratio=ratio,
            max_ratio=benchmark.max_ratio,
            baseline_ratio=benchmark.baseline_ratio,
            runs=benchmark.runs,
        )

    def stage_mutation_tests(self) -> None:
        if self._blocked(Stage.MUTATION_TESTS):
            return
        task = self.task
        if not task.mutations:
            self.skip(Stage.MUTATION_TESTS, "the task is not graded by mutation")
            return

        start = time.monotonic()

        # The agent's tests must pass on the source as the agent left it. Skipping
        # this would let a test file that fails on correct code score points for
        # "detecting" every mutation.
        clean_result = self.sandbox.run_tests(task.agent_tests, timeout=self.timeout)
        clean_counts = parse_result(clean_result)
        if not clean_result.success or (clean_counts and clean_counts.collected_nothing):
            if clean_result.timed_out:
                self.aborted = True
            self.add(
                Stage.MUTATION_TESTS,
                StageStatus.FAILED,
                (time.monotonic() - start) * 1000,
                detail=(
                    "the tests do not pass against unmutated source, so nothing "
                    f"they report is trustworthy: {tail(clean_result)}"
                ),
                score=0.0,
                tests=clean_counts,
                passes_on_clean_source=False,
            )
            return

        killed, survived = [], []
        for mutation in task.mutations:
            for rel_path in mutation.files:
                self._put(
                    self.task_dir / "hidden" / "mutations" / mutation.name / rel_path, rel_path
                )

            result = self.sandbox.run_tests(task.agent_tests, timeout=self.timeout)
            counts = parse_result(result)
            detected = (not result.success) and not (counts and counts.collected_nothing)
            (killed if detected else survived).append(mutation.name)

            if result.timed_out:
                # A hang counts as detection (the tests noticed something), but the
                # container is gone, so stop here and grade what we have.
                self.aborted = True
                break

            for rel_path in mutation.files:
                self._restore(rel_path)

        total = len(killed) + len(survived)
        score = len(killed) / total if total else 0.0
        self.add(
            Stage.MUTATION_TESTS,
            StageStatus.PASSED if total and not survived else StageStatus.FAILED,
            (time.monotonic() - start) * 1000,
            detail=(
                f"{len(killed)}/{total} mutations detected"
                + (f"; undetected: {', '.join(survived)}" if survived else "")
            ),
            score=score,
            passes_on_clean_source=True,
            killed=killed,
            survived=survived,
            command=task.agent_tests,
        )

    def stage_static_checks(self) -> None:
        # Deliberately not gated on `_blocked`: the patch-level checks are
        # host-side, so a torn-down container still gets a real static verdict.
        start = time.monotonic()
        findings = checks.inspect_patch(self.patch, protected_paths=self.task.hidden_test_files)
        commands_run = []

        if self.task.static_checks and not self.aborted and self.sandbox and self.sandbox.container_id:
            for command in self.task.static_checks:
                result = self.sandbox.run_command(command, timeout=DEFAULT_STATIC_CHECK_TIMEOUT)
                commands_run.append({"command": command, "exit_code": result.exit_code})
                if not result.success:
                    findings.append(
                        checks.Finding(
                            "task_static_check",
                            checks.ERROR,
                            f"`{command}` exited {result.exit_code}: {tail(result, 200)}",
                        )
                    )
                if result.timed_out:
                    self.aborted = True
                    break

        failures = checks.errors(findings)
        self.add(
            Stage.STATIC_CHECKS,
            StageStatus.FAILED if failures else StageStatus.PASSED,
            (time.monotonic() - start) * 1000,
            detail="; ".join(str(finding) for finding in failures[:5]),
            findings=[finding.to_dict() for finding in findings],
            commands=commands_run,
        )

    # ---- helpers -----------------------------------------------------------

    @property
    def task_dir(self) -> Path:
        return self.evaluator.task_dir

    def task_hidden(self, rel_path: str) -> Path:
        return self.task_dir / "hidden" / rel_path

    def _restore(self, rel_path: str) -> None:
        """Put a file back the way the agent's patch left it, from the host copy
        of the graded tree -- the container's own copy is the mutated one."""
        source = self.clean / rel_path if self.clean else None
        if source and source.exists():
            self._put(source, rel_path)

    # ---- assembling the result ---------------------------------------------

    def finish(self, started_at: str, start: float, agent: Optional[dict]) -> EvaluationResult:
        task = self.task
        outcome = decide_outcome(task.category, self.stages, self.patch.is_empty)
        limits = task.resource_limit

        return EvaluationResult(
            task_id=task.task_id,
            category=task.category,
            outcome=outcome,
            stages=self.stages,
            started_at=started_at,
            duration_ms=(time.monotonic() - start) * 1000,
            evaluator_version=EVALUATOR_VERSION,
            agent=agent or {},
            provenance={
                "repository": task.repository,
                "commit": task.commit,
                "language": task.language,
                "difficulty": task.difficulty.value,
                "image": limits.to_sandbox_config().image,
                "resource_limit": {
                    "cpus": limits.cpus,
                    "memory_gb": limits.memory_gb,
                    "timeout_seconds": limits.timeout_seconds,
                    "pids_limit": limits.pids_limit,
                    "network_enabled": limits.network_enabled,
                },
                "patch": self.patch.to_dict(),
                "commands": {
                    "setup": task.setup,
                    "build": task.build,
                    "public_tests": task.public_tests,
                    "hidden_tests": task.hidden_tests,
                    "regression_tests": task.regression_tests,
                    "agent_tests": task.agent_tests,
                    "benchmark": task.benchmark.to_dict() if task.benchmark else None,
                    "static_checks": task.static_checks,
                },
            },
        )


def evaluate_patch(
    task: Task,
    task_dir: Path,
    patch: Patch,
    repo: Optional[Path] = None,
    config: Optional[EvaluationConfig] = None,
    agent: Optional[dict] = None,
) -> EvaluationResult:
    """One-shot convenience wrapper around Evaluator."""
    return Evaluator(task, task_dir, repo=repo, config=config).evaluate(patch, agent=agent)


#: Re-exported so callers do not need to know which module raised.
__all__ = [
    "EvaluationConfig",
    "Evaluator",
    "Outcome",
    "evaluate_patch",
]
