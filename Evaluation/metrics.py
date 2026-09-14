"""The Phase 4 primary metrics, aggregated over a set of evaluations.

plan.md names four: Task Success Rate, Hidden Test Pass Rate, Build Success Rate
and Partial Success. Two choices about how they are computed matter more than the
arithmetic:

**Infrastructure errors are excluded from every capability rate, and reported
separately** (Principle 5). A run where Docker died says nothing about whether the
model can fix a netrc bug, and folding it in as a failure would make the runtime's
reliability look like the model's incapability -- exactly the confusion the
principle exists to prevent. `infrastructure_error_rate` is reported on its own so
the number is visible rather than buried.

**Hidden Test Pass Rate is reported both ways.** Micro (every hidden test across
the dataset, pooled) answers "what fraction of the required behaviour works" and
is dominated by tasks with large suites; macro (the mean of the per-task rates)
weights every task equally. They diverge, so quoting only one invites a wrong
conclusion.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .result import EvaluationResult, Outcome


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


@dataclass
class Metrics:
    """Metrics over one group of results (the whole dataset, or one category)."""

    label: str = "all"
    total: int = 0
    #: Runs that actually measured the agent: everything but infrastructure errors.
    evaluable: int = 0
    solved: int = 0
    partial: int = 0
    failed: int = 0
    build_failed: int = 0
    patch_failed: int = 0
    infrastructure_errors: int = 0
    hidden_tests_passed: int = 0
    hidden_tests_graded: int = 0
    builds_attempted: int = 0
    builds_succeeded: int = 0
    partial_scores: List[float] = field(default_factory=list)
    macro_hidden_rates: List[float] = field(default_factory=list)
    outcomes: Dict[str, int] = field(default_factory=dict)

    # ---- the four primary metrics ------------------------------------------

    @property
    def task_success_rate(self) -> Optional[float]:
        """solved / evaluable. None when nothing could be evaluated."""
        return self.solved / self.evaluable if self.evaluable else None

    @property
    def hidden_test_pass_rate(self) -> Optional[float]:
        """Micro-average: every hidden test in the group, pooled."""
        return (
            self.hidden_tests_passed / self.hidden_tests_graded
            if self.hidden_tests_graded
            else None
        )

    @property
    def hidden_test_pass_rate_macro(self) -> Optional[float]:
        """Mean of the per-task pass rates: one vote per task, not per test."""
        return _mean(self.macro_hidden_rates)

    @property
    def build_success_rate(self) -> Optional[float]:
        return self.builds_succeeded / self.builds_attempted if self.builds_attempted else None

    @property
    def partial_success(self) -> Optional[float]:
        """Mean fraction of each task's achievement criteria that were met.

        Always at least the task success rate, and above it exactly when agents
        are getting part-way rather than nowhere -- which is the distinction a
        binary success rate throws away.
        """
        return _mean(self.partial_scores)

    # ---- reliability, kept separate from capability ------------------------

    @property
    def infrastructure_error_rate(self) -> Optional[float]:
        return self.infrastructure_errors / self.total if self.total else None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "total": self.total,
            "evaluable": self.evaluable,
            "task_success_rate": self.task_success_rate,
            "hidden_test_pass_rate": self.hidden_test_pass_rate,
            "hidden_test_pass_rate_macro": self.hidden_test_pass_rate_macro,
            "build_success_rate": self.build_success_rate,
            "partial_success": self.partial_success,
            "infrastructure_error_rate": self.infrastructure_error_rate,
            "counts": {
                "solved": self.solved,
                "partial": self.partial,
                "failed": self.failed,
                "build_failed": self.build_failed,
                "patch_failed": self.patch_failed,
                "infrastructure_errors": self.infrastructure_errors,
                "hidden_tests_passed": self.hidden_tests_passed,
                "hidden_tests_graded": self.hidden_tests_graded,
            },
            "outcomes": self.outcomes,
        }


def _accumulate(metrics: Metrics, results: Iterable[EvaluationResult]) -> Metrics:
    outcome_counts: Counter = Counter()

    for result in results:
        metrics.total += 1
        outcome_counts[result.outcome.value] += 1

        if result.outcome is Outcome.INFRASTRUCTURE_ERROR:
            metrics.infrastructure_errors += 1
            # Deliberately contributes to nothing below: this run measured the
            # runtime, not the agent.
            continue

        metrics.evaluable += 1
        if result.outcome is Outcome.SOLVED:
            metrics.solved += 1
        elif result.outcome is Outcome.PARTIAL:
            metrics.partial += 1
        elif result.outcome is Outcome.BUILD_FAILED:
            metrics.build_failed += 1
        elif result.outcome is Outcome.PATCH_FAILED:
            metrics.patch_failed += 1
        else:
            metrics.failed += 1

        metrics.partial_scores.append(result.partial_success)

        # A patch that never applied never had a build to succeed or fail at, so
        # it is left out of the build rate rather than counted as a build failure.
        if result.outcome is not Outcome.PATCH_FAILED:
            metrics.builds_attempted += 1
            if result.build_succeeded:
                metrics.builds_succeeded += 1

        counts = result.hidden_test_counts
        if counts and counts.graded:
            metrics.hidden_tests_passed += counts.passed
            metrics.hidden_tests_graded += counts.graded
            metrics.macro_hidden_rates.append(counts.pass_rate or 0.0)

    metrics.outcomes = dict(outcome_counts)
    return metrics


def aggregate(results: Iterable[EvaluationResult], label: str = "all") -> Metrics:
    return _accumulate(Metrics(label=label), results)


def by_category(results: Iterable[EvaluationResult]) -> Dict[str, Metrics]:
    grouped: Dict[str, List[EvaluationResult]] = {}
    for result in results:
        grouped.setdefault(result.category.value, []).append(result)
    return {name: aggregate(group, label=name) for name, group in sorted(grouped.items())}


def _percent(value: Optional[float]) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def format_report(results: Sequence[EvaluationResult]) -> str:
    """The human-readable summary printed at the end of a run."""
    overall = aggregate(results)
    lines = [""]

    lines.append(f"{'task':<46} {'outcome':<14} {'hidden':>9} {'partial':>8}")
    lines.append("-" * 80)
    for result in results:
        counts = result.hidden_test_counts
        hidden = f"{counts.passed}/{counts.graded}" if counts and counts.graded else "-"
        lines.append(
            f"{result.task_id:<46} {result.outcome.value:<14} {hidden:>9} "
            f"{result.partial_success:>7.0%}"
        )

    lines.append("")
    lines.append("Primary metrics (plan.md Phase 4)")
    lines.append(f"  Task success rate      {_percent(overall.task_success_rate)}"
                 f"   ({overall.solved}/{overall.evaluable} evaluable tasks solved)")
    lines.append(f"  Hidden test pass rate  {_percent(overall.hidden_test_pass_rate)}"
                 f"   ({overall.hidden_tests_passed}/{overall.hidden_tests_graded} tests,"
                 f" macro {_percent(overall.hidden_test_pass_rate_macro).strip()})")
    lines.append(f"  Build success rate     {_percent(overall.build_success_rate)}"
                 f"   ({overall.builds_succeeded}/{overall.builds_attempted})")
    lines.append(f"  Partial success        {_percent(overall.partial_success)}"
                 f"   (mean achievement criteria met)")

    if overall.infrastructure_errors:
        lines.append(
            f"  Infrastructure errors  {_percent(overall.infrastructure_error_rate)}"
            f"   ({overall.infrastructure_errors}/{overall.total}) -- excluded from the rates above"
        )

    groups = by_category(results)
    if len(groups) > 1:
        lines.append("")
        lines.append(f"{'by category':<24} {'solved':>8} {'success':>9} {'partial':>9}")
        for name, metrics in groups.items():
            lines.append(
                f"  {name:<22} {metrics.solved:>3}/{metrics.evaluable:<4} "
                f"{_percent(metrics.task_success_rate)} {_percent(metrics.partial_success)}"
            )

    return "\n".join(lines)
