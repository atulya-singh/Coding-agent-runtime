"""Structured, logged, timeout-bound tools the agent uses to inspect and modify a repo.

Phase 1 only: run_command/run_tests execute on the host process. Phase 2 wraps all
execution in a sandbox so agent-issued commands never touch the host directly.
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
