"""Strategy B: retry with context -- tell the agent what went wrong.

Strategy A absorbs failures the agent could do nothing about. This is the
opposite case: failures the agent caused and is the only thing that can fix, so
the recovery is not to repeat the call but to say something about it.

Three of them, all named in plan.md's taxonomy:

    INVALID_TOOL_CALL  a tool that does not exist, or arguments that do not fit
                       its schema. Caught before dispatch and answered with the
                       schema, rather than with whatever TypeError the call
                       would have raised.

    COMMAND_FAILURE    a command that keeps failing the same way. The first one
                       is an observation and the tool result already carries it;
                       the fourth identical one is a fact about the agent.

    AGENT_LOOP         the same call, with the same arguments, producing the
                       same result, over and over. The agent is not learning
                       anything and neither is the experiment.

The repeat rule is deliberately strict: a repeat only counts when the call *and
its result* are identical. Running the test suite twice is how the job is done
-- it is only a loop when the output does not change. That distinction is what
keeps the honest edit-test-edit cycle from being flagged, and it is why the
digest covers the output and not just the arguments.

Escalation is the part that makes this measurable. Being told is the recovery;
being told twice and doing it again anyway is a result, and the run ends as
AGENT_LOOP rather than spending thirty more turns proving the point. Whether
being told helps at all is exactly what plan.md's Experiment 3 compares, so all
of it is configurable and can be switched off to get the control arm.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .failures import FailureType

#: Prefixes anything the harness says to the model, so a note about its own
#: behaviour is never mistaken for output a tool produced.
HARNESS_PREFIX = "[harness]"

DEFAULT_REPEAT_THRESHOLD = 3
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_MAX_NUDGES = 2
DEFAULT_WINDOW = 12

#: JSON schema type -> what Python accepts for it. `bool` is excluded from the
#: numeric types by hand below: True is an int in Python and is not one here.
_JSON_TYPES = {
    "string": str,
    "boolean": bool,
    "object": dict,
    "array": list,
    "integer": int,
    "number": (int, float),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ContextPolicy:
    """When to say something, and when to stop the run instead."""

    #: Off gives the control arm: failures still reach the model as tool
    #: results, but the harness never comments and never intervenes.
    enabled: bool = True
    #: Identical call + identical result this many times before the first note.
    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD
    #: Same tool failing the same way this many times, arguments aside.
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    #: How many times to say it before ending the run instead of repeating it.
    max_nudges: int = DEFAULT_MAX_NUDGES
    #: How far back a repeat counts. Reading one file twice forty calls apart
    #: is not a loop; twice in a row is worth noticing.
    window: int = DEFAULT_WINDOW
    #: Check tool calls against their schema before dispatching them.
    validate_tool_calls: bool = True

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "repeat_threshold": self.repeat_threshold,
            "failure_threshold": self.failure_threshold,
            "max_nudges": self.max_nudges,
            "window": self.window,
            "validate_tool_calls": self.validate_tool_calls,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ContextPolicy":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            repeat_threshold=int(data.get("repeat_threshold", DEFAULT_REPEAT_THRESHOLD)),
            failure_threshold=int(data.get("failure_threshold", DEFAULT_FAILURE_THRESHOLD)),
            max_nudges=int(data.get("max_nudges", DEFAULT_MAX_NUDGES)),
            window=int(data.get("window", DEFAULT_WINDOW)),
            validate_tool_calls=bool(data.get("validate_tool_calls", True)),
        )


def validate_context_policy(policy: ContextPolicy) -> List[str]:
    errors: List[str] = []
    if policy.repeat_threshold < 2:
        # At 1 the first call is a repeat of itself.
        errors.append(f"repeat_threshold must be at least 2, got {policy.repeat_threshold}")
    if policy.failure_threshold < 2:
        errors.append(f"failure_threshold must be at least 2, got {policy.failure_threshold}")
    if policy.max_nudges < 0:
        errors.append(f"max_nudges must be zero or more, got {policy.max_nudges}")
    if policy.window < policy.repeat_threshold:
        errors.append(
            f"window ({policy.window}) must be at least repeat_threshold "
            f"({policy.repeat_threshold}), or a repeat can never be seen"
        )
    return errors


@dataclass
class Guidance:
    """Something the harness decided the agent needed to be told."""

    failure: str
    #: Which rule fired: invalid_tool_call, repeated_call, repeated_failure.
    reason: str
    #: What the model is shown, verbatim.
    message: str
    tool: str = ""
    occurrences: int = 0
    #: How many times this same thing has now been said.
    nudge: int = 0
    #: True when saying it again would be pointless and the run should end.
    escalate: bool = False
    at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "failure": self.failure,
            "reason": self.reason,
            "message": self.message,
            "tool": self.tool,
            "occurrences": self.occurrences,
            "nudge": self.nudge,
            "escalate": self.escalate,
            "at": self.at,
        }


# ---- invalid tool calls ----------------------------------------------------


def describe_parameters(spec: dict) -> str:
    """The tool's signature, the way a docstring would put it."""
    schema = spec.get("input_schema") or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if not properties:
        return "it takes no documented parameters"
    parts = []
    for name, definition in properties.items():
        kind = (definition or {}).get("type", "any")
        parts.append(f"{name} ({kind}{', required' if name in required else ''})")
    return "expected " + ", ".join(parts)


def _type_error(name: str, value: Any, expected: str) -> str:
    accepted = _JSON_TYPES.get(expected)
    if accepted is None:
        return ""
    # True is an instance of int, and a boolean where a count belongs is a real
    # mistake rather than a technicality worth waving through.
    if expected in ("integer", "number") and isinstance(value, bool):
        return f"{name} must be a {expected}, got a boolean"
    if isinstance(value, accepted):
        return ""
    return f"{name} must be a {expected}, got {type(value).__name__}"


def validate_tool_call(name: str, arguments: Any, specs: List[dict]) -> str:
    """Empty if the call can be dispatched, else what is wrong with it.

    Checked against the same specs the model was given, so the complaint can
    quote the contract the model was working from rather than describing the
    Python signature behind it -- which the model has never seen.
    """
    by_name = {spec.get("name"): spec for spec in specs}

    if name not in by_name:
        available = ", ".join(sorted(str(key) for key in by_name if key))
        close = difflib.get_close_matches(str(name or ""), [k for k in by_name if k], n=1, cutoff=0.6)
        hint = f" Did you mean {close[0]}?" if close else ""
        return f"there is no tool called {name!r}.{hint} Available tools: {available}."

    spec = by_name[name]
    if not isinstance(arguments, dict):
        return f"{name} takes an object of arguments, got {type(arguments).__name__}."

    schema = spec.get("input_schema") or {}
    properties = schema.get("properties") or {}
    required = schema.get("required") or []

    problems = []
    missing = [key for key in required if key not in arguments]
    if missing:
        problems.append(f"missing required argument(s): {', '.join(missing)}")

    # Only when the spec actually documents its properties. A spec without them
    # is not claiming to be exhaustive, and rejecting against it would reject
    # perfectly good calls.
    if properties:
        unknown = [key for key in arguments if key not in properties]
        if unknown:
            close = []
            for key in unknown:
                match = difflib.get_close_matches(key, list(properties), n=1, cutoff=0.6)
                if match:
                    close.append(f"{key} -> {match[0]}")
            hint = f" (did you mean {', '.join(close)}?)" if close else ""
            problems.append(f"unknown argument(s): {', '.join(unknown)}{hint}")

        for key, value in arguments.items():
            definition = properties.get(key) or {}
            wrong = _type_error(key, value, definition.get("type", ""))
            if wrong:
                problems.append(wrong)

    if not problems:
        return ""
    return f"{name} was called incorrectly: {'; '.join(problems)}. {describe_parameters(spec)}."


# ---- loops and repeated failures -------------------------------------------


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 -- a fingerprint, never a correctness path
        return repr(value)


def _digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(_canonical(part) for part in parts).encode()).hexdigest()[:16]


def call_failed(result: Any) -> bool:
    """Did this attempt accomplish anything?

    Wider than `ToolResult.success` on purpose. A command that runs and exits
    nonzero is a successful *tool call* -- the tool did its job -- but it is not
    a step forward, and the fourth identical one is the thing worth naming.
    """
    if not getattr(result, "success", True):
        return True
    metadata = getattr(result, "metadata", None) or {}
    exit_code = metadata.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return True
    output = getattr(result, "output", None)
    if isinstance(output, dict):
        code = output.get("exit_code")
        return isinstance(code, int) and code != 0
    return False


@dataclass
class _Observation:
    fingerprint: str
    tool: str
    error_key: str
    failed: bool


class ContextAdvisor:
    """Watches the tool calls of one run and speaks up when they stop helping.

    Holds no reference to the loop, the toolset or the model: it is handed a
    call and a result and answers with text or with nothing. That keeps the
    decision to intervene reviewable on its own, and keeps `AgentLoop` from
    growing a second job.
    """

    def __init__(self, policy: Optional[ContextPolicy] = None, specs: Optional[List[dict]] = None):
        self.policy = policy or ContextPolicy()
        self.specs = list(specs or [])
        self.history: List[_Observation] = []
        self.given: List[Guidance] = []
        self._nudges: Dict[str, int] = {}
        #: Set once, the first time saying it again would be pointless.
        self.escalated: Optional[Guidance] = None

    # -- before dispatch ----------------------------------------------------

    def check_call(self, name: str, arguments: Any) -> Optional[Guidance]:
        """Reject a call that cannot work, with the schema it should have met.

        Returned rather than raised, and checked before the toolset sees it, so
        a malformed call never becomes a command and never depends on whatever
        error the underlying function would have produced.
        """
        if not (self.policy.enabled and self.policy.validate_tool_calls and self.specs):
            return None
        problem = validate_tool_call(name, arguments, self.specs)
        if not problem:
            return None
        return self._record(
            Guidance(
                failure=FailureType.INVALID_TOOL_CALL.value,
                reason="invalid_tool_call",
                message=f"{HARNESS_PREFIX} {problem}",
                tool=str(name),
            ),
            key=f"invalid:{name}:{_digest(problem)}",
            failed=True,
        )

    # -- after dispatch -----------------------------------------------------

    def observe(self, name: str, arguments: Any, result: Any) -> Optional[Guidance]:
        """Record what happened and say something if it has stopped mattering."""
        failed = call_failed(result)
        error_key = ""
        if failed:
            # What went wrong, not which arguments it went wrong for: the point
            # of this key is to catch the agent varying the arguments while
            # getting the identical complaint back.
            error_key = _digest(name, (getattr(result, "error", None) or "")[:200])

        observation = _Observation(
            fingerprint=_digest(name, arguments, self._outcome(result)),
            tool=str(name),
            error_key=error_key,
            failed=failed,
        )
        self.history.append(observation)
        if not self.policy.enabled:
            return None

        recent = self.history[-self.policy.window :]

        repeats = sum(1 for entry in recent if entry.fingerprint == observation.fingerprint)
        if repeats >= self.policy.repeat_threshold:
            return self._record(
                Guidance(
                    failure=FailureType.AGENT_LOOP.value,
                    reason="repeated_call",
                    message=self._repeat_message(name, repeats, failed),
                    tool=str(name),
                    occurrences=repeats,
                ),
                key=f"repeat:{observation.fingerprint}",
                failed=failed,
            )

        if failed and error_key:
            same = sum(1 for entry in recent if entry.error_key == error_key)
            if same >= self.policy.failure_threshold:
                return self._record(
                    Guidance(
                        failure=FailureType.COMMAND_FAILURE.value,
                        reason="repeated_failure",
                        message=self._failure_message(name, same, result),
                        tool=str(name),
                        occurrences=same,
                    ),
                    key=f"failure:{error_key}",
                    failed=True,
                )
        return None

    # -- the record ---------------------------------------------------------

    @property
    def interventions(self) -> List[dict]:
        return [guidance.to_dict() for guidance in self.given]

    def summary_line(self) -> str:
        if not self.given:
            return "no interventions"
        counts: Dict[str, int] = {}
        for guidance in self.given:
            counts[guidance.reason] = counts.get(guidance.reason, 0) + 1
        parts = ", ".join(f"{reason} x{count}" for reason, count in sorted(counts.items()))
        return f"{len(self.given)} intervention(s) ({parts})" + (
            "; run ended for looping" if self.escalated else ""
        )

    # -- internals ----------------------------------------------------------

    def _outcome(self, result: Any) -> str:
        """A digest of what came back, so a repeat means "and nothing changed".

        Hashed rather than kept: a tool result can be a whole file, and the
        advisor only ever needs to know whether two of them were the same.
        """
        return _digest(
            getattr(result, "success", None),
            getattr(result, "error", None),
            getattr(result, "output", None),
        )

    def _record(self, guidance: Guidance, key: str, failed: bool) -> Optional[Guidance]:
        nudges = self._nudges.get(key, 0) + 1
        self._nudges[key] = nudges
        guidance.nudge = nudges

        if nudges > self.policy.max_nudges:
            if not failed:
                # Repeating something harmless is wasteful, not broken. It has
                # been mentioned; there is nothing further worth saying and
                # nothing worth ending a run over.
                return None
            guidance.escalate = True
            guidance.message = (
                f"{guidance.message}\n{HARNESS_PREFIX} this has now been pointed out "
                f"{self.policy.max_nudges} time(s) and nothing changed, so the run is being stopped."
            )
            if self.escalated is None:
                self.escalated = guidance

        self.given.append(guidance)
        return guidance

    def _repeat_message(self, name: str, repeats: int, failed: bool) -> str:
        if failed:
            return (
                f"{HARNESS_PREFIX} you have called {name} with these exact arguments "
                f"{repeats} times and received the identical failure each time. Repeating it "
                "will not produce a different result. Change the approach: inspect the current "
                "state of the file or the environment before acting, or try a different tool."
            )
        return (
            f"{HARNESS_PREFIX} you have called {name} with these exact arguments {repeats} "
            "times and received the identical result each time. The answer is already above -- "
            "re-reading it is not making progress."
        )

    def _failure_message(self, name: str, count: int, result: Any) -> str:
        error = (getattr(result, "error", None) or "").strip()
        detail = f" The failure is: {error[:300]}" if error else ""
        return (
            f"{HARNESS_PREFIX} {name} has now failed {count} times with the same error, "
            f"despite different arguments.{detail} Whatever assumption these calls share is "
            "the wrong one -- verify it directly before calling this tool again."
        )
