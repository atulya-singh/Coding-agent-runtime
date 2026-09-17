"""Phase 6: the recovery manager. Strategy A (retry) is what exists so far.

A leaf package on purpose. It imports nothing from Agent, Sandbox, Execution,
Evaluation, State or Tasks, so every one of them can call into it without a
cycle and without dragging the rest of the system into a unit test.

plan.md Phase 6 lists six strategies; this package holds the first two, plus
the failure taxonomy all six are written in:

    A  retry                 retry.py    -- absorb what the agent could not have
                                            helped: timeouts, drops, rate limits
    B  retry with context    context.py  -- name what the agent did wrong and
                                            hand it back: bad calls, loops
    C  rollback              \\
    D  escalate model         |  not built; the taxonomy is already shaped
    E  verifier-guided        |  to describe what they would act on
    F  restart               /

The split between A and B is the useful one. A is invisible to the agent by
design -- a rate limit is not something it can act on, and telling it would only
spend context. B is the opposite: the only thing that can fix the failure is the
agent, so the failure is the message.
"""
from .context import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_MAX_NUDGES,
    DEFAULT_REPEAT_THRESHOLD,
    DEFAULT_WINDOW,
    HARNESS_PREFIX,
    ContextAdvisor,
    ContextPolicy,
    Guidance,
    call_failed,
    describe_parameters,
    validate_context_policy,
    validate_tool_call,
)
from .failures import Failure, FailureType, classify, retry_after_seconds
from .retry import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MULTIPLIER,
    MAX_RETRY_AFTER,
    MIN_JITTER,
    RetryAttempt,
    RetryLog,
    RetryPolicy,
    retry_call,
    validate_policy,
)

__all__ = [
    "FailureType",
    "Failure",
    "classify",
    "retry_after_seconds",
    "ContextPolicy",
    "ContextAdvisor",
    "Guidance",
    "validate_context_policy",
    "validate_tool_call",
    "describe_parameters",
    "call_failed",
    "HARNESS_PREFIX",
    "DEFAULT_REPEAT_THRESHOLD",
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_MAX_NUDGES",
    "DEFAULT_WINDOW",
    "RetryPolicy",
    "RetryAttempt",
    "RetryLog",
    "retry_call",
    "validate_policy",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BASE_DELAY",
    "DEFAULT_MULTIPLIER",
    "DEFAULT_MAX_DELAY",
    "MIN_JITTER",
    "MAX_RETRY_AFTER",
]
