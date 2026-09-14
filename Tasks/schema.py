from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Sandbox import SandboxConfig


class TaskCategory(str, Enum):
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    TESTING = "testing"
    PERFORMANCE = "performance"
    API_CHANGE = "api_change"
    DEPENDENCY_MIGRATION = "dependency_migration"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ResourceLimit:
    cpus: float = 2.0
    memory_gb: float = 4.0
    timeout_seconds: int = 600
    pids_limit: int = 100
    network_enabled: bool = False

    def to_sandbox_config(self, **overrides) -> SandboxConfig:
        kwargs = dict(
            cpus=self.cpus,
            memory_gb=self.memory_gb,
            timeout_seconds=self.timeout_seconds,
            pids_limit=self.pids_limit,
            network_enabled=self.network_enabled,
        )
        kwargs.update(overrides)
        return SandboxConfig(**kwargs)


@dataclass
class Benchmark:
    """Objective pass/fail criterion for a `performance` task.

    `command` runs in the sandbox and must print a single float to stdout: the
    ratio between the hot path under test and an in-process control measured in
    the same run. A ratio rather than wall-clock seconds, so one threshold stays
    meaningful on a laptop, in CI and inside a CPU-limited container.
    """

    command: str
    max_ratio: float
    # What the untouched base commit scores. Documents the size of the gap, and
    # lets validation reject a threshold the unfixed code already meets.
    baseline_ratio: float
    runs: int = 3  # median of N, to damp scheduler noise

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "max_ratio": self.max_ratio,
            "baseline_ratio": self.baseline_ratio,
            "runs": self.runs,
        }


@dataclass
class Mutation:
    """A deliberate break, used to grade a `testing` task.

    Tests that pass prove nothing on their own -- an empty file passes. Each
    mutation overwrites source files with a broken version, and the tests written
    for the task must FAIL against every one of them. Replacement files live in
    `<task_dir>/hidden/mutations/<name>/<path-matching-repo-layout>`; whole files
    rather than diffs, so applying one needs no patch tool in the container.
    """

    name: str
    files: List[str]
    description: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "files": self.files, "description": self.description}


@dataclass
class Task:
    task_id: str
    repository: str
    commit: str
    language: str
    category: TaskCategory
    difficulty: Difficulty
    description: str
    setup: List[str] = field(default_factory=list)
    public_tests: str = ""
    # hidden_tests/hidden_test_files are grading-only (Phase 4) and must never be
    # fed into the agent's context -- kept as separate fields/files for that reason.
    hidden_tests: str = ""
    hidden_test_files: List[str] = field(default_factory=list)
    # Must pass both before and after the change. public_tests are advisory and
    # agent-visible; these are graded, and catch a "solution" that satisfies
    # hidden_tests by breaking everything around it.
    regression_tests: str = ""
    # `testing` tasks only: runs the tests the AGENT was asked to write. Graded
    # against `mutations` rather than by passing on its own.
    agent_tests: str = ""
    benchmark: Optional[Benchmark] = None
    mutations: List[Mutation] = field(default_factory=list)
    # Provenance, and what makes the dataset self-checking. `reference_commit` is
    # the real upstream commit this task was mined from -- `commit` above is its
    # parent, the pre-change state the agent starts from. `reference_paths` are
    # the paths to take from it to reconstruct the intended solution. Used only by
    # Tasks.verify to prove a task is actually solvable; never shown to the agent,
    # never part of grading.
    reference_commit: str = ""
    reference_paths: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    resource_limit: ResourceLimit = field(default_factory=ResourceLimit)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        data = dict(data)
        data["category"] = TaskCategory(data["category"])
        data["difficulty"] = Difficulty(data["difficulty"])
        data["resource_limit"] = ResourceLimit(**(data.get("resource_limit") or {}))
        benchmark = data.get("benchmark")
        data["benchmark"] = Benchmark(**benchmark) if benchmark else None
        data["mutations"] = [Mutation(**m) for m in (data.get("mutations") or [])]
        # YAML infers types from content (e.g. an all-digit commit SHA parses as an
        # int), so force these back to strings regardless of how they were written.
        for str_field in ("task_id", "repository", "commit", "language"):
            if str_field in data:
                data[str_field] = str(data[str_field])
        return cls(**data)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "repository": self.repository,
            "commit": self.commit,
            "language": self.language,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "description": self.description,
            "setup": self.setup,
            "public_tests": self.public_tests,
            "hidden_tests": self.hidden_tests,
            "hidden_test_files": self.hidden_test_files,
            "regression_tests": self.regression_tests,
            "agent_tests": self.agent_tests,
            "benchmark": self.benchmark.to_dict() if self.benchmark else None,
            "mutations": [m.to_dict() for m in self.mutations],
            "reference_commit": self.reference_commit,
            "reference_paths": self.reference_paths,
            "expected_behavior": self.expected_behavior,
            "resource_limit": {
                "cpus": self.resource_limit.cpus,
                "memory_gb": self.resource_limit.memory_gb,
                "timeout_seconds": self.resource_limit.timeout_seconds,
                "pids_limit": self.resource_limit.pids_limit,
                "network_enabled": self.resource_limit.network_enabled,
            },
        }


REQUIRED_FIELDS = ("task_id", "repository", "commit", "language", "category", "difficulty", "description")

# A category is not just a label: it selects the grading contract. Each entry
# names the fields that make a task in that category objectively gradable, and
# validate_task() refuses a task that doesn't declare them. Without this the enum
# is decorative and a `performance` task can ship with no measurement at all.
CATEGORY_CONTRACTS: Dict[TaskCategory, Tuple[str, ...]] = {
    # Something concrete is broken or missing: one command fails before the
    # change and passes after it.
    TaskCategory.BUG_FIX: ("hidden_tests",),
    TaskCategory.FEATURE: ("hidden_tests",),
    # The environment is what changed, so the same suite that fails on the old
    # dependency has to pass on the new one.
    TaskCategory.DEPENDENCY_MIGRATION: ("hidden_tests",),
    # "...while preserving existing behavior" -- the preserved half has to be
    # graded too, or the agent can pass by rewriting whatever it likes around the
    # new API.
    TaskCategory.API_CHANGE: ("hidden_tests", "regression_tests"),
    # A refactor changes no behavior, so hidden_tests must assert the structure
    # really changed; without regression_tests, deleting the module would pass.
    TaskCategory.REFACTOR: ("hidden_tests", "regression_tests"),
    # Nothing fails before a performance fix, so the criterion is a measurement
    # against a threshold, plus proof the speed-up didn't change behavior.
    TaskCategory.PERFORMANCE: ("benchmark", "regression_tests"),
    # Here the agent writes the tests, so passing is not evidence -- an empty file
    # passes. They are graded by mutation instead.
    TaskCategory.TESTING: ("agent_tests", "mutations"),
}


def validate_task(task: Task, task_dir: Optional[Path] = None) -> List[str]:
    errors = []
    for field_name in REQUIRED_FIELDS:
        if not getattr(task, field_name, None):
            errors.append(f"missing required field: {field_name}")

    for field_name in CATEGORY_CONTRACTS.get(task.category, ()):
        if not getattr(task, field_name, None):
            errors.append(
                f"category '{task.category.value}' requires '{field_name}' "
                f"(see CATEGORY_CONTRACTS): a task in this category is not "
                f"objectively gradable without it"
            )

    if len(task.commit) < 7:
        errors.append(f"commit '{task.commit}' does not look like a valid git SHA")

    if not task.reference_commit:
        errors.append("missing reference_commit: the task cannot be verified as solvable")
    elif len(task.reference_commit) < 7:
        errors.append(
            f"reference_commit '{task.reference_commit}' does not look like a valid git SHA"
        )
    if task.reference_commit and not task.reference_paths:
        errors.append("reference_commit is set but reference_paths is empty")
    if task.reference_commit == task.commit:
        errors.append("reference_commit must differ from commit (commit is the pre-change parent)")

    if task.benchmark is not None:
        if task.benchmark.max_ratio <= 0:
            errors.append("benchmark.max_ratio must be > 0")
        elif task.benchmark.max_ratio >= task.benchmark.baseline_ratio:
            errors.append(
                f"benchmark.max_ratio ({task.benchmark.max_ratio}) must be below "
                f"baseline_ratio ({task.benchmark.baseline_ratio}), otherwise the "
                f"unfixed base commit already passes"
            )
        if task.benchmark.runs < 1:
            errors.append("benchmark.runs must be >= 1")

    seen_mutations: set = set()
    for mutation in task.mutations:
        if not mutation.name or not mutation.files:
            errors.append(f"mutation {mutation.name!r} needs a name and at least one file")
        if mutation.name in seen_mutations:
            errors.append(f"duplicate mutation name: {mutation.name}")
        seen_mutations.add(mutation.name)

    if task_dir is not None:
        for rel_path in task.hidden_test_files:
            if not (Path(task_dir) / "hidden" / rel_path).exists():
                errors.append(f"hidden_test_files entry not found on disk: hidden/{rel_path}")
        for mutation in task.mutations:
            for rel_path in mutation.files:
                mutation_file = Path(task_dir) / "hidden" / "mutations" / mutation.name / rel_path
                if not mutation_file.exists():
                    errors.append(
                        f"mutation file not found on disk: "
                        f"hidden/mutations/{mutation.name}/{rel_path}"
                    )

    return errors
