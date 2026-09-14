"""Phase 4: the objective evaluation system.

Grades an agent's work by what its patch does to a clean checkout of the task's
base commit -- never by asking a model whether the solution looks right
(Principle 1). A verdict comes from tests, a build, a measured benchmark, killed
mutations and static checks; `expected_behavior` in a task is documentation, not a
criterion.

    from Evaluation import Evaluator, collect_patch, aggregate

    patch = collect_patch(sandbox, base_dir)          # what the agent changed
    result = Evaluator(task, task_dir).evaluate(patch)
    print(result.summary(), result.outcome, result.hidden_test_pass_rate)

Command line:

    python -m Evaluation <task_id> --patch changes.diff
    python -m Evaluation --all --reference       # grade the known-good solutions
    python -m Evaluation.selftest                # prove the pipeline itself works
"""
from .evaluator import EvaluationConfig, Evaluator, evaluate_patch
from .metrics import Metrics, aggregate, by_category, format_report
from .patch import Patch, PatchError, apply_patch, collect_patch, diff_trees, load_patch
from .pytest_report import TestCounts
from .result import (
    ACHIEVEMENT_STAGES_BY_CATEGORY,
    EVALUATOR_VERSION,
    GATE_STAGES,
    EvaluationResult,
    Outcome,
    Stage,
    StageResult,
    StageStatus,
    credit_for,
)
from .static_checks import Finding, inspect_patch

__all__ = [
    "Evaluator",
    "EvaluationConfig",
    "evaluate_patch",
    "EvaluationResult",
    "Outcome",
    "Stage",
    "StageResult",
    "StageStatus",
    "GATE_STAGES",
    "credit_for",
    "ACHIEVEMENT_STAGES_BY_CATEGORY",
    "EVALUATOR_VERSION",
    "Patch",
    "PatchError",
    "collect_patch",
    "diff_trees",
    "apply_patch",
    "load_patch",
    "TestCounts",
    "Finding",
    "inspect_patch",
    "Metrics",
    "aggregate",
    "by_category",
    "format_report",
]
