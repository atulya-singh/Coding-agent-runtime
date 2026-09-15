"""The one tool the loop answers itself.

`finish` is a control-flow signal, not an action: the loop intercepts it and
never dispatches it to the SandboxToolset, so it stays outside the seven real
tools rather than being bolted onto Execution's contract. The model needs some
way to say "I'm done" that is distinguishable from "I have nothing more to
say", which is why this exists at all -- an empty assistant turn is ambiguous
between finishing and giving up.
"""
from __future__ import annotations

from typing import List

FINISH_TOOL_NAME = "finish"

FINISH_TOOL_SPEC = {
    "name": FINISH_TOOL_NAME,
    "description": (
        "Call this when the task is complete and you have verified your changes, "
        "or when you are certain you cannot make further progress. This ends the "
        "session -- no further tools can be used afterwards."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "What you changed and how you verified it. If you could not "
                    "complete the task, say what blocked you."
                ),
            }
        },
        "required": ["summary"],
    },
}


def build_tool_specs(toolset) -> List[dict]:
    """The seven sandbox tools plus `finish`, as the model-facing tool list."""
    specs = list(toolset.tool_specs())
    names = {spec["name"] for spec in specs}
    if FINISH_TOOL_NAME in names:
        # Would be intercepted by the loop and silently never executed.
        raise ValueError(
            f"SandboxToolset defines a tool named {FINISH_TOOL_NAME!r}, which collides "
            "with the loop's completion signal"
        )
    specs.append(FINISH_TOOL_SPEC)
    return specs
