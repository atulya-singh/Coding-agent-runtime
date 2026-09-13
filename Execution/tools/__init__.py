"""Structured, logged, timeout-bound tools for inspecting and modifying a repo.

These run in the HOST process and are for harness-side use only. Agent-driven calls
must go through `Execution.sandbox_tools.SandboxToolset`, which exposes the same
seven tools with the same ToolResult/logging contract but executes them inside a
Sandbox container -- plan.md's Phase 1 rule is that the model never runs anything on
the host. `base.py` is shared by both.
"""
from .base import ToolExecutionError, ToolResult, ToolTimeoutError
from .execution import run_command, run_tests
from .file_ops import edit_file, read_file, write_file
from .git_ops import git_diff
from .search import search_code

__all__ = [
    "ToolResult",
    "ToolTimeoutError",
    "ToolExecutionError",
    "read_file",
    "write_file",
    "edit_file",
    "search_code",
    "run_command",
    "run_tests",
    "git_diff",
]
