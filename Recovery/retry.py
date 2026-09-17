"""Strategy A: do it again, later each time.

    attempt 1 fails -> wait 1s -> 2 -> wait 2s -> 3 -> wait 4s -> 4 -> wait 8s
                    -> 5 -> wait 16s -> 6 -> give up

Five retries, so six attempts, and at most ~31 seconds of waiting before an
operation is declared lost. The exponential shape is the point: a provider that
is rate-limiting or restarting needs room, and a fixed interval either hammers
it or wastes the cases that would have recovered on the first repeat.

Three properties this is built around:

  It re-raises the original exception. `retry_call` is transparent -- callers
  see the failure they would have seen anyway, wrapped in nothing. Only the log
  reveals that anything was retried, which means retry can be threaded through
  existing call sites without changing what any of them catch.

  Every retry is recorded before it is slept on. A retry that fixes a run is
  invisible in the result, so if it is not written down the run looks clean and
  the environment looks healthy. Principle 4: a failure the system absorbed is
  still data, and "retry rate" is a metric this project promised to report.

  Sleeping is injected. The selftest drives all six attempts and asserts the
  exact delay schedule without waiting 31 seconds to do it.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .failures import Failure, classify, retry_after_seconds

logger = logging.getLogger("recovery")

#: The limit the whole package is specified around: five retries per operation.
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MULTIPLIER = 2.0
#: Ceiling on one computed wait. Past roughly half a minute the exponent stops
#: buying anything -- a provider that is still refusing after 30s is not going
#: to be fixed by 60 -- and the remaining retries are better spent as attempts.
DEFAULT_MAX_DELAY = 30.0

#: Jitter multiplies the computed delay by [MIN_JITTER, 1.0] rather than by
#: [0, 1.0]. Full jitter decorrelates better but flattens the curve -- a "16
#: second" wait that rolls 0.3s is not a backoff -- and the exponential shape
#: is the behaviour being asked for here.
MIN_JITTER = 0.5

#: A Retry-After is obeyed over our own schedule, but not without limit: a
#: header saying "come back in two hours" should end the operation, not park a
#: run for two hours.
MAX_RETRY_AFTER = 300.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RetryPolicy:
    """How many times, and how long between. Configuration, not code."""

    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    multiplier: float = DEFAULT_MULTIPLIER
    max_delay: float = DEFAULT_MAX_DELAY
    jitter: bool = True
    #: Prefer the server's own Retry-After over the computed delay. Off only
    #: makes sense in a test that wants a fixed schedule.
    respect_retry_after: bool = True

    @property
    def max_attempts(self) -> int:
        """Retries plus the original call. Named because the off-by-one between
        "5 retries" and "5 attempts" is worth being unable to get wrong."""
        return self.max_retries + 1

    def backoff(self, retry: int) -> float:
        """The un-jittered wait before retry number `retry` (1-based)."""
        if retry < 1:
            raise ValueError(f"retry is 1-based, got {retry}")
        return min(self.base_delay * (self.multiplier ** (retry - 1)), self.max_delay)

    def delay_for(
        self,
        retry: int,
        retry_after: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> float:
        """The actual wait, given what the server said and how the dice fell.

        A server-supplied wait is taken as-is: it is not a guess, so jittering
        it would only make us early, and being early to a rate limit is the one
        outcome the header exists to prevent.
        """
        if retry_after is not None and self.respect_retry_after:
            return min(max(retry_after, 0.0), MAX_RETRY_AFTER)
        delay = self.backoff(retry)
        if not self.jitter:
            return delay
        return delay * (rng or random).uniform(MIN_JITTER, 1.0)

    def schedule(self) -> List[float]:
        """Every un-jittered wait, in order. What `python -m Recovery` prints."""
        return [self.backoff(retry) for retry in range(1, self.max_retries + 1)]

    def worst_case_wait(self) -> float:
        """Total seconds an operation can spend asleep before it is given up on."""
        return sum(self.schedule())

    def to_dict(self) -> dict:
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "multiplier": self.multiplier,
            "max_delay": self.max_delay,
            "jitter": self.jitter,
            "respect_retry_after": self.respect_retry_after,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RetryPolicy":
        data = data or {}
        return cls(
            max_retries=int(data.get("max_retries", DEFAULT_MAX_RETRIES)),
            base_delay=float(data.get("base_delay", DEFAULT_BASE_DELAY)),
            multiplier=float(data.get("multiplier", DEFAULT_MULTIPLIER)),
            max_delay=float(data.get("max_delay", DEFAULT_MAX_DELAY)),
            jitter=bool(data.get("jitter", True)),
            respect_retry_after=bool(data.get("respect_retry_after", True)),
        )


def validate_policy(policy: RetryPolicy) -> List[str]:
    """Human-readable errors, mirroring Agent.config.validate_config's shape."""
    errors: List[str] = []
    if policy.max_retries < 0:
        errors.append(f"max_retries must be zero or more, got {policy.max_retries}")
    if policy.base_delay <= 0:
        errors.append(f"base_delay must be positive, got {policy.base_delay}")
    if policy.multiplier < 1:
        # Below 1 the waits shrink, which is the opposite of a backoff.
        errors.append(f"multiplier must be at least 1, got {policy.multiplier}")
    if policy.max_delay < policy.base_delay:
        errors.append(
            f"max_delay ({policy.max_delay}) must be at least base_delay ({policy.base_delay})"
        )
    return errors


@dataclass
class RetryAttempt:
    """One retry, as it will appear on the run record and in the checkpoint."""

    operation: str
    #: 1-based: retry 1 is the second attempt at this operation.
    retry: int
    failure: str
    error: str
    delay_s: float
    #: True when the wait came from the server's Retry-After rather than from
    #: the policy -- the difference between "we backed off" and "we were told".
    from_retry_after: bool = False
    at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "retry": self.retry,
            "failure": self.failure,
            "error": self.error[:500],
            "delay_s": round(self.delay_s, 3),
            "from_retry_after": self.from_retry_after,
            "at": self.at,
        }


class RetryLog:
    """Every retry an object made, in order. Usable directly as `on_retry`."""

    def __init__(self, attempts: Optional[List[RetryAttempt]] = None):
        self.attempts: List[RetryAttempt] = list(attempts or [])

    def __call__(self, attempt: RetryAttempt) -> None:
        self.attempts.append(attempt)

    def __len__(self) -> int:
        return len(self.attempts)

    def __iter__(self):
        return iter(self.attempts)

    def __bool__(self) -> bool:
        # Without this, `if log:` on an empty-but-present log reads as absent.
        return bool(self.attempts)

    def to_dicts(self) -> List[dict]:
        return [attempt.to_dict() for attempt in self.attempts]

    def counts_by_failure(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for attempt in self.attempts:
            counts[attempt.failure] = counts.get(attempt.failure, 0) + 1
        return counts

    def summary_line(self) -> str:
        if not self.attempts:
            return "no retries"
        parts = ", ".join(f"{name} x{count}" for name, count in sorted(self.counts_by_failure().items()))
        waited = sum(attempt.delay_s for attempt in self.attempts)
        return f"{len(self.attempts)} retr{'y' if len(self.attempts) == 1 else 'ies'} ({parts}), {waited:.1f}s waited"


def retry_call(
    operation: str,
    func: Callable[[], Any],
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[RetryAttempt], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: Optional[random.Random] = None,
    classifier: Callable[[BaseException], Failure] = classify,
) -> Any:
    """Call `func`, repeating it on transient failures with growing waits.

    Returns whatever `func` returns. Raises whatever `func` raised, once the
    failure is one repetition cannot fix or the retries run out -- deliberately
    the original exception and not a wrapper, so a caller that already catches
    `anthropic.APITimeoutError` keeps working once retry is wrapped around it,
    and so the error a run records is the one that actually happened rather
    than a description of the harness's reaction to it.

    `operation` is a label for the record ("messages.create"), not a key --
    nothing dispatches on it.
    """
    policy = policy or RetryPolicy()
    retries = 0

    while True:
        try:
            return func()
        except BaseException as exc:  # noqa: BLE001 -- classified, then re-raised
            # KeyboardInterrupt and SystemExit are not failures of the
            # operation; they are someone stopping the process, and sleeping
            # five times before honouring that would be indefensible.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failure = classifier(exc)
            if not failure.retryable or retries >= policy.max_retries:
                if retries:
                    logger.warning(
                        "%s failed after %d retr%s: %s",
                        operation, retries, "y" if retries == 1 else "ies", failure.error[:200],
                    )
                raise

            retries += 1
            after = retry_after_seconds(exc) if policy.respect_retry_after else None
            delay = policy.delay_for(retries, after, rng)
            attempt = RetryAttempt(
                operation=operation,
                retry=retries,
                failure=failure.type.value,
                error=failure.error,
                delay_s=delay,
                from_retry_after=after is not None,
            )
            logger.warning(
                "%s: %s (%s), retry %d/%d in %.1fs",
                operation, failure.type.value, failure.exception,
                retries, policy.max_retries, delay,
            )
            # Recorded before the sleep, so a run killed mid-backoff still shows
            # what it was waiting on.
            if on_retry is not None:
                on_retry(attempt)
            sleep(delay)
