"""The record one evaluation produces, and the rules that turn stages into a verdict.

Two ideas do most of the work here.

**Gates versus achievements.** Some stages say the agent accomplished the task
(hidden tests, a benchmark threshold, killed mutations); the rest only say it did
not wreck anything on the way (the patch applies, the build works, the regression
suite and the public suite still pass, the static checks are clean). Only
achievements earn partial credit. Without that split an agent that changed nothing
would score partial credit on a refactor task purely because the regression tests
it never touched still pass.

**Agent failure versus infrastructure failure** (Principle 5). A task the model
could not solve and a task whose container never came up are different events and
must not be averaged together, so `INFRASTRUCTURE_ERROR` is its own outcome and is
excluded from the capability metrics rather than counted as a failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from Tasks.schema import TaskCategory

from .pytest_report import TestCounts

#: Bumped when the meaning of a stage or an outcome changes, so a stored result
#: can never be silently compared against one produced by different rules.
EVALUATOR_VERSION = 1


class Stage(str, Enum):
    """Ordered exactly as plan.md's Phase 4 pipeline runs them."""

    PATCH = "patch"
    SETUP = "setup"
    BUILD = "build"
    PUBLIC_TESTS = "public_tests"
    HIDDEN_TESTS = "hidden_tests"
    REGRESSION_TESTS = "regression_tests"
    BENCHMARK = "benchmark"
    MUTATION_TESTS = "mutation_tests"
    STATIC_CHECKS = "static_checks"


class StageStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    #: Not applicable to this task (no benchmark on a bug_fix), or unreachable
    #: because an earlier stage stopped the pipeline.
    SKIPPED = "skipped"
    #: The runtime broke, not the change under test.
    ERROR = "error"


class Outcome(str, Enum):
    SOLVED = "solved"
    PARTIAL = "partial"
    FAILED = "failed"
    BUILD_FAILED = "build_failed"
    PATCH_FAILED = "patch_failed"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


#: Stages that only license a success; passing them is not itself progress.
GATE_STAGES = (
    Stage.PATCH,
    Stage.SETUP,
    Stage.BUILD,
    Stage.PUBLIC_TESTS,
    Stage.REGRESSION_TESTS,
    Stage.STATIC_CHECKS,
)

#: Stages that constitute solving the task. Mirrors CATEGORY_CONTRACTS in
#: Tasks/schema.py -- that decides what a task must declare to be gradable, this
#: decides what grading it then means.
ACHIEVEMENT_STAGES_BY_CATEGORY: Dict[TaskCategory, tuple] = {
    TaskCategory.BUG_FIX: (Stage.HIDDEN_TESTS,),
    TaskCategory.FEATURE: (Stage.HIDDEN_TESTS,),
    TaskCategory.DEPENDENCY_MIGRATION: (Stage.HIDDEN_TESTS,),
    TaskCategory.API_CHANGE: (Stage.HIDDEN_TESTS,),
    TaskCategory.REFACTOR: (Stage.HIDDEN_TESTS,),
    TaskCategory.PERFORMANCE: (Stage.BENCHMARK,),
    TaskCategory.TESTING: (Stage.MUTATION_TESTS,),
}


@dataclass
class StageResult:
    stage: Stage
    status: StageStatus
    duration_ms: float = 0.0
    #: Short human-readable reason. For a failure this is the tail of the command
    #: output, which is what a person reads first when triaging.
    detail: str = ""
    #: How completely this stage was satisfied, 0.0-1.0. Only meaningful for
    #: achievement stages; a gate is 1.0 or 0.0.
    score: float = 0.0
    tests: Optional[TestCounts] = None
    #: Stage-specific structured facts (measured ratio, mutations killed, the
    #: command that ran) kept out of `detail` so tooling doesn't parse prose.
    data: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is StageStatus.PASSED

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "score": self.score,
            "detail": self.detail,
            "tests": self.tests.to_dict() if self.tests else None,
            "data": self.data,
        }


@dataclass
class EvaluationResult:
    task_id: str
    category: TaskCategory
    outcome: Outcome
    stages: List[StageResult] = field(default_factory=list)
    #: Everything needed to reproduce this run (Principle 3). Filled with what
    #: Phase 4 knows -- repository, commit, image, limits, patch identity. Later
    #: phases add the model, prompt and token/cost fields under `agent`.
    provenance: dict = field(default_factory=dict)
    agent: dict = field(default_factory=dict)
    started_at: str = ""
    duration_ms: float = 0.0
    evaluator_version: int = EVALUATOR_VERSION

    # ---- lookups -----------------------------------------------------------

    def stage(self, stage: Stage) -> Optional[StageResult]:
        for result in self.stages:
            if result.stage is stage:
                return result
        return None

    def _ran(self, stages) -> List[StageResult]:
        return [
            result
            for result in self.stages
            if result.stage in stages and result.status is not StageStatus.SKIPPED
        ]

    # ---- the Phase 4 primary metrics, per task -----------------------------

    @property
    def solved(self) -> bool:
        return self.outcome is Outcome.SOLVED

    @property
    def build_succeeded(self) -> bool:
        """Did the patch produce a runnable tree? Setup counts: for a Python repo
        `pip install -e .` is the build."""
        return all(
            result.passed
            for result in self._ran((Stage.SETUP, Stage.BUILD))
        ) and self.stage(Stage.SETUP) is not None

    @property
    def hidden_test_counts(self) -> Optional[TestCounts]:
        result = self.stage(Stage.HIDDEN_TESTS)
        return result.tests if result else None

    @property
    def hidden_test_pass_rate(self) -> Optional[float]:
        counts = self.hidden_test_counts
        return counts.pass_rate if counts else None

    @property
    def partial_success(self) -> float:
        """Fraction of the task's achievement criteria that were met.

        Useful exactly where plan.md says it is: a task with several independent
        requirements, where "9 of the 10 new hidden tests pass" is a materially
        different result from "none of them do". Measured against what the base
        commit already scored (see `credit_for`), so it starts at 0 for an agent
        that changed nothing and reaches 1.0 only for a full solution.
        """
        achievements = self._ran(ACHIEVEMENT_STAGES_BY_CATEGORY.get(self.category, ()))
        if not achievements:
            return 0.0
        return sum(result.score for result in achievements) / len(achievements)

    @property
    def gates_passed(self) -> bool:
        return all(result.passed for result in self._ran(GATE_STAGES))

    @property
    def failed_stages(self) -> List[Stage]:
        return [
            result.stage
            for result in self.stages
            if result.status in (StageStatus.FAILED, StageStatus.ERROR)
        ]

    def to_dict(self) -> dict:
        return {
            "evaluator_version": self.evaluator_version,
            "task_id": self.task_id,
            "category": self.category.value,
            "outcome": self.outcome.value,
            "solved": self.solved,
            "build_succeeded": self.build_succeeded,
            "hidden_test_pass_rate": self.hidden_test_pass_rate,
            "partial_success": self.partial_success,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 2),
            "stages": [stage.to_dict() for stage in self.stages],
            "provenance": self.provenance,
            "agent": self.agent,
        }

    def summary(self) -> str:
        line = f"{self.task_id} [{self.category.value}]: {self.outcome.value.upper()}"
        counts = self.hidden_test_counts
        if counts and counts.graded:
            line += f"  hidden {counts.passed}/{counts.graded}"
        if 0.0 < self.partial_success < 1.0:
            line += f"  partial {self.partial_success:.0%}"
        if self.failed_stages:
            line += "  failed: " + ", ".join(stage.value for stage in self.failed_stages)
        return line


#: Credit below this is rounding noise rather than progress. `baseline_hidden_
#: pass_rate` is stored to a few decimal places (Tasks.verify accepts a 0.001
#: mismatch), so without a floor a patch that changes literally nothing lands a
#: sliver above zero and reads as PARTIAL. Comfortably under the credit one test
#: is worth in a suite of any size this dataset uses.
CREDIT_EPSILON = 0.005


def credit_for(raw: float, baseline: float) -> float:
    """Credit for a measured pass rate, given what the untouched base already scores.

    Credit is for closing the gap to 1.0, not for the raw rate. A hidden suite
    normally covers the cases *around* the change as well as the new ones -- which
    is what stops a fix that breaks its neighbours -- so its rate on the base is
    often well above zero, and paying for that would award an agent that changed
    nothing most of the marks. Solving the task still scores exactly 1.0.
    """
    if baseline >= 1.0:  # rejected by validate_task; treated as unearnable here
        return 0.0
    gained = raw if baseline <= 0.0 else (raw - baseline) / (1.0 - baseline)
    if gained < CREDIT_EPSILON:
        return 0.0
    return min(1.0, gained)


def decide_outcome(
    category: TaskCategory, stages: List[StageResult], patch_is_empty: bool
) -> Outcome:
    """Collapse the stage results into one verdict.

    Order matters: the earliest thing that went wrong is the most informative
    label, because everything after it was either skipped or measured against a
    tree that was already broken.
    """
    by_stage = {result.stage: result for result in stages}

    def failed(stage: Stage) -> bool:
        result = by_stage.get(stage)
        return result is not None and result.status is StageStatus.FAILED

    def errored(stage: Stage) -> bool:
        result = by_stage.get(stage)
        return result is not None and result.status is StageStatus.ERROR

    if failed(Stage.PATCH):
        return Outcome.PATCH_FAILED
    if errored(Stage.PATCH) or errored(Stage.SETUP):
        return Outcome.INFRASTRUCTURE_ERROR
    if failed(Stage.SETUP):
        # Setup ran clean on this commit when the task was verified, so a failure
        # now is the patch's doing -- unless there is no patch, in which case the
        # environment itself is at fault (a registry outage, a yanked wheel).
        return Outcome.INFRASTRUCTURE_ERROR if patch_is_empty else Outcome.BUILD_FAILED
    if failed(Stage.BUILD):
        return Outcome.BUILD_FAILED
    if any(result.status is StageStatus.ERROR for result in stages):
        return Outcome.INFRASTRUCTURE_ERROR

    achievements = [
        by_stage[stage]
        for stage in ACHIEVEMENT_STAGES_BY_CATEGORY.get(category, ())
        if stage in by_stage and by_stage[stage].status is not StageStatus.SKIPPED
    ]
    if not achievements:
        # Nothing that could demonstrate success ever ran; refusing to call that
        # solved is the whole point of the category contracts.
        return Outcome.FAILED

    score = sum(result.score for result in achievements) / len(achievements)
    gates_ok = all(
        result.passed
        for result in stages
        if result.stage in GATE_STAGES and result.status is not StageStatus.SKIPPED
    )

    if gates_ok and score >= 1.0:
        return Outcome.SOLVED
    return Outcome.PARTIAL if score > 0.0 else Outcome.FAILED


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
