"""Sandboxed tools: the Phase 1 toolset executed inside a Phase 2 Sandbox.

`Execution.tools` runs on the host and is for harness code only. Anything an agent
drives goes through `SandboxToolset`, which satisfies plan.md's Phase 1 rule that
the model never executes arbitrary host commands.
"""
from .toolset import (
    PROTOCOL_VERSION,
    TOOLHOST_PATH,
    SandboxToolError,
    SandboxToolset,
    toolset_for,
)

__all__ = [
    "SandboxToolset",
    "SandboxToolError",
    "toolset_for",
    "PROTOCOL_VERSION",
    "TOOLHOST_PATH",
]
