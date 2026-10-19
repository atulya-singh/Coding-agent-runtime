"""Strategy D: when the agent is stuck, hand the task to a stronger model.

    haiku -- stuck (agent_loop) --> sonnet -- stuck --> opus -- stuck --> give up

Triggered by StopReason.AGENT_LOOP only: the agent looped, was told so by
Strategy B, and kept looping. That is a capability failure, which a stronger
model can plausibly fix. MODEL_ERROR is deliberately not a trigger -- the API
failing is not something a different model on the same API will fix.

The stronger model does not get the weaker one's transcript. It gets the task,
the files as the weaker model left them (restored from the checkpoint), and a
short summary of what was tried -- enough to avoid repeating a dead end without
being anchored to it.

The ladder is configuration, not code: which model counts as "stronger" is a
judgement the person running the experiment makes, and price is part of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

#: Weakest first. Every model here needs a `pricing` entry in Agent/config.yaml,
#: or a run that escalates into it reports an unknown cost.
DEFAULT_LADDER = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EscalationPolicy:
    """Which models to climb through, in order. Configuration, not code."""

    #: Off gives the control arm: a stuck run stays stuck and is recorded so.
    enabled: bool = True
    ladder: List[str] = field(default_factory=lambda: list(DEFAULT_LADDER))

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "ladder": list(self.ladder),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "EscalationPolicy":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            ladder=[str(model) for model in data.get("ladder", DEFAULT_LADDER)],
        )


def validate_escalation_policy(policy: EscalationPolicy) -> List[str]:
    """Human-readable errors, mirroring Agent.config.validate_config's shape."""
    errors: List[str] = []
    if any(not model for model in policy.ladder):
        errors.append("ladder entries must be non-empty model ids")
    if len(set(policy.ladder)) != len(policy.ladder):
        # A repeated model would let a run "escalate" to the model that just failed.
        errors.append(f"ladder must not repeat a model, got {policy.ladder}")
    if policy.enabled and len(policy.ladder) < 2:
        errors.append("ladder needs at least two models to escalate between")
    return errors


def next_model(ladder: List[str], current: str) -> Optional[str]:
    """The next model up the ladder, or None if there is nowhere left to go.

    A model not on the ladder also gives None: there is no way to know where
    it ranks, and guessing would be escalating to something possibly weaker.
    """
    if current not in ladder:
        return None
    position = ladder.index(current)
    if position + 1 >= len(ladder):
        return None
    return ladder[position + 1]


def summarize_attempt(tool_calls: List[dict], interventions: List[dict]) -> str:
    """A short note for the stronger model: what was tried, and what failed.

    Built from the checkpoint's records, not by asking a model -- free, cannot
    fail, and the same run always produces the same summary.
    """
    if not tool_calls and not interventions:
        return "A previous attempt with a weaker model got stuck before calling any tools."

    # Only edits that actually happened: a refused call (dispatched=False) or a
    # failed write left the file as it was, and claiming otherwise would send
    # the stronger model looking for changes that are not there.
    edited = sorted(
        {
            str(call["arguments"].get("path", "?"))
            for call in tool_calls
            if call.get("tool") in EDIT_TOOLS
            and call.get("success")
            and call.get("dispatched", True)
            and isinstance(call.get("arguments"), dict)
        }
    )
    test_runs = [call for call in tool_calls if call.get("tool") == "run_tests"]
    failed_tests = [call for call in test_runs if not call.get("success")]
    last_error = next(
        (call.get("error") for call in reversed(tool_calls) if call.get("error")), ""
    )
    reasons = sorted({entry.get("reason", "?") for entry in interventions})

    lines = [
        "A previous attempt with a weaker model got stuck and was stopped.",
        f"- Tool calls made: {len(tool_calls)}",
    ]
    if edited:
        lines.append(f"- Files it changed (still changed now): {', '.join(edited)}")
    else:
        lines.append("- It changed no files.")
    if test_runs:
        lines.append(f"- Tests run {len(test_runs)} time(s), failing {len(failed_tests)}")
    if last_error:
        lines.append(f"- Last error: {_clip(last_error)}")
    if reasons:
        lines.append(f"- The harness warned it about: {', '.join(reasons)}")
    lines.append(
        "Check its changes before building on them, and do not repeat an "
        "approach that already failed without changing something first."
    )
    return "\n".join(lines)


#: Tools whose success means a file on disk is now different.
EDIT_TOOLS = ("write_file", "edit_file")
#: Long enough for an assertion message, short enough that one huge traceback
#: does not become most of the summary.
MAX_ERROR_CHARS = 200


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_ERROR_CHARS else text[: MAX_ERROR_CHARS - 3] + "..."


@dataclass
class EscalationStep:
    """One handoff, as it will appear on the run record."""

    from_model: str
    to_model: str
    #: The stop reason that triggered it -- today always "agent_loop".
    reason: str
    #: The turn the weaker model was on when it was stopped.
    at_turn: int
    #: What the weaker model's stretch cost on its own rates, so the record
    #: never prices one model's tokens at another's.
    cost_usd: Optional[float] = None
    at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "from_model": self.from_model,
            "to_model": self.to_model,
            "reason": self.reason,
            "at_turn": self.at_turn,
            "cost_usd": self.cost_usd,
            "at": self.at,
        }
