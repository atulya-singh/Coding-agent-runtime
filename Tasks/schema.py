from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

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
    expected_behavior: str = ""
    resource_limit: ResourceLimit = field(default_factory=ResourceLimit)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        data = dict(data)
        data["category"] = TaskCategory(data["category"])
        data["difficulty"] = Difficulty(data["difficulty"])
        data["resource_limit"] = ResourceLimit(**(data.get("resource_limit") or {}))
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


def validate_task(task: Task, task_dir: Optional[Path] = None) -> List[str]:
    errors = []
    for field_name in REQUIRED_FIELDS:
        if not getattr(task, field_name, None):
            errors.append(f"missing required field: {field_name}")

    if not task.public_tests and not task.hidden_tests:
        errors.append("task defines neither public_tests nor hidden_tests")

    if len(task.commit) < 7:
        errors.append(f"commit '{task.commit}' does not look like a valid git SHA")

    if task_dir is not None:
        for rel_path in task.hidden_test_files:
            if not (Path(task_dir) / "hidden" / rel_path).exists():
                errors.append(f"hidden_test_files entry not found on disk: hidden/{rel_path}")

    return errors
