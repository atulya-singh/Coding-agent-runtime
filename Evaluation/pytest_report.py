"""Turn pytest's terminal output into counts.

A stage that only records "the command exited non-zero" cannot answer two of the
Phase 4 metrics: Hidden Test Pass Rate needs `passed / total`, and Partial Success
needs to know that 9 of 10 requirements were met rather than 0 of 10.

Counts are read from pytest's own summary line rather than by rewriting the task's
command to add `--junitxml`. A task's test command is an arbitrary shell string --
it may be a pipeline, a wrapper script, or a runner other than pytest -- and
splicing a flag into it would silently corrupt those. Parsing is best-effort by
design: `parse` returns None when it cannot find a summary, and every caller falls
back to the exit code, which is always authoritative for pass/fail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

#: "... in 1.23s" / "... in 0.5 seconds" -- pytest always ends its summary with a
#: duration, which is what distinguishes the real summary line from a traceback
#: line that happens to mention "passed".
_DURATION = re.compile(r"\bin\s+[\d.]+\s*(?:s|seconds?)\b")

_COUNT = re.compile(
    r"(\d+)\s+(passed|failed|xfailed|xpassed|skipped|deselected|errors?|warnings?)\b"
)

#: pytest prints this instead of counts when collection matched nothing. Worth
#: distinguishing from "0 tests, all passed": a hidden-test command that matches
#: no tests is a broken task, not a solved one.
_NO_TESTS = re.compile(r"\bno tests ran\b", re.IGNORECASE)


@dataclass
class TestCounts:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    deselected: int = 0
    collected_nothing: bool = False

    @property
    def graded(self) -> int:
        """Tests that actually returned a verdict.

        Skipped, deselected and xfailed tests are excluded: none of them is
        evidence for or against the agent's change, and counting them would let a
        patch raise its pass rate by making tests skip.
        """
        return self.passed + self.failed + self.errors

    @property
    def pass_rate(self) -> Optional[float]:
        return self.passed / self.graded if self.graded else None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "xfailed": self.xfailed,
            "xpassed": self.xpassed,
            "deselected": self.deselected,
            "graded": self.graded,
            "pass_rate": self.pass_rate,
            "collected_nothing": self.collected_nothing,
        }

    def __str__(self) -> str:
        if self.collected_nothing:
            return "no tests ran"
        parts = [f"{self.passed} passed"]
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.errors:
            parts.append(f"{self.errors} errors")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        return ", ".join(parts)


_FIELD_FOR_WORD = {
    "passed": "passed",
    "failed": "failed",
    "error": "errors",
    "errors": "errors",
    "skipped": "skipped",
    "xfailed": "xfailed",
    "xpassed": "xpassed",
    "deselected": "deselected",
}


def parse(output: str) -> Optional[TestCounts]:
    """Read the last pytest summary line in `output`, or None if there isn't one.

    Scans backwards so that the final summary wins even when the output contains
    several runs (a task whose test command invokes pytest more than once).
    """
    if not output:
        return None

    for line in reversed(output.splitlines()):
        line = line.strip().strip("=").strip()
        if not line:
            continue

        if _NO_TESTS.search(line):
            return TestCounts(collected_nothing=True)

        matches = _COUNT.findall(line)
        if not matches or not _DURATION.search(line):
            continue

        counts = TestCounts()
        recognised = False
        for number, word in matches:
            field_name = _FIELD_FOR_WORD.get(word)
            if field_name:  # "3 warnings" is real but not a verdict; skip it
                setattr(counts, field_name, getattr(counts, field_name) + int(number))
                recognised = True
        if recognised:
            return counts

    return None


def parse_result(result) -> Optional[TestCounts]:
    """Parse a SandboxResult. pytest writes its summary to stdout, but a crashing
    run can put the tail on stderr, so both are considered."""
    return parse(result.stdout) or parse(result.stderr)
