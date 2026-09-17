"""Phase 6: the recovery manager. Strategy A (retry) is what exists so far.

A leaf package on purpose. It imports nothing from Agent, Sandbox, Execution,
Evaluation, State or Tasks, so every one of them can call into it without a
cycle and without dragging the rest of the system into a unit test.

plan.md Phase 6 lists six strategies; this package holds the first, plus the
failure taxonomy all six are written in:

    A  retry        here
    B  retry with context    \\
    C  rollback               |  not built; the taxonomy is already shaped
    D  escalate model         |  to describe what they would act on
    E  verifier-guided       /
    F  restart
"""
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
