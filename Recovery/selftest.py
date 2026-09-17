"""Prove the retry policy before it is trusted with a real run.

    python -m Recovery.selftest

No Docker, no network, no API key -- and no waiting, because `retry_call` takes
its `sleep` as an argument and every check here passes a counter instead. That
is deliberate: the interesting cases are the slow ones (five retries, 31
seconds of backoff, a server-sent Retry-After), and a suite that had to live
through them in real time would not be run often enough to be worth having.

`--offline` is accepted and ignored, so this matches how the other packages are
invoked; there is no online half to skip.

The one import outside this package is the shared `Checks` helper. Recovery is
a leaf at runtime -- nothing in the package proper imports Agent, State,
Evaluation or Sandbox -- and the checks below deliberately reach into Agent to
confirm the wiring at the call site actually took.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from typing import List, Optional

from Evaluation.selftest import Checks
from Execution.tools.base import ToolTimeoutError

from .failures import FailureType, classify, retry_after_seconds
from .retry import (
    MAX_RETRY_AFTER,
    MIN_JITTER,
    RetryAttempt,
    RetryLog,
    RetryPolicy,
    retry_call,
    validate_policy,
)

#: Exception names the table claims come from the anthropic SDK. Checked against
#: the installed SDK below, because a renamed exception class would silently
#: stop being retried and nothing else would notice.
SDK_EXCEPTIONS = [
    "APITimeoutError",
    "APIConnectionError",
    "APIResponseValidationError",
    "RateLimitError",
    "OverloadedError",
    "AuthenticationError",
    "PermissionDeniedError",
    "BadRequestError",
    "NotFoundError",
    "UnprocessableEntityError",
    "RequestTooLargeError",
    "DeadlineExceededError",
    "CredentialsError",
    "RetryableError",
]


# ---- doubles ---------------------------------------------------------------


class Sleeper:
    """Records what it was asked to wait for instead of waiting for it."""

    def __init__(self) -> None:
        self.delays: List[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.delays)


class Flaky:
    """Raises the queued exceptions in order, then returns a value."""

    def __init__(self, failures: List[BaseException], value: str = "ok"):
        self.failures = list(failures)
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.value


class Headers:
    def __init__(self, values: Optional[dict] = None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)


class FakeResponse:
    def __init__(self, headers: Optional[dict] = None):
        self.headers = Headers(headers)


# Named to match the SDK's classes, since classification is by name. Real
# instances of the SDK's own exceptions need an httpx request to construct;
# the name check further down is what ties these stand-ins to the real ones.
class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class RateLimitError(Exception):
    status_code = 429

    def __init__(self, message="rate limited", retry_after=None):
        super().__init__(message)
        self.response = FakeResponse({"retry-after": retry_after} if retry_after else {})


class AuthenticationError(Exception):
    status_code = 401


class OverloadedError(Exception):
    status_code = 529


class InternalServerError(Exception):
    status_code = 500


class WeirdProviderError(Exception):
    """A class the table has never heard of, carrying only a status code."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class SubclassedTimeout(APITimeoutError):
    """Classification must walk the MRO, not just look at the leaf class."""


# ---- the policy itself -----------------------------------------------------


def check_policy(c: Checks) -> None:
    print("\n--- the backoff schedule")
    policy = RetryPolicy()

    c.equal("five retries by default", policy.max_retries, 5)
    c.equal("which is six attempts", policy.max_attempts, 6)
    c.equal("the waits double", policy.schedule(), [1.0, 2.0, 4.0, 8.0, 16.0])
    c.equal("31 seconds of patience in total", policy.worst_case_wait(), 31.0)

    capped = RetryPolicy(base_delay=10.0, multiplier=10.0, max_delay=30.0, max_retries=4)
    c.equal("no single wait exceeds max_delay", capped.schedule(), [10.0, 30.0, 30.0, 30.0])

    c.raises("backoff() rejects a 0th retry", ValueError, lambda: policy.backoff(0))

    zero = RetryPolicy(max_retries=0)
    c.equal("max_retries 0 means one attempt and no waits", zero.schedule(), [])

    print("\n--- jitter")
    jittered = RetryPolicy(jitter=True)
    rng = random.Random(1234)
    draws = [jittered.delay_for(3, rng=rng) for _ in range(200)]
    c.check(
        "jitter stays inside [0.5x, 1.0x] of the computed wait",
        all(4.0 * MIN_JITTER <= draw <= 4.0 for draw in draws),
        f"min {min(draws):.3f}, max {max(draws):.3f}, expected within [2.0, 4.0]",
    )
    c.check("jitter actually varies", len(set(draws)) > 100, f"{len(set(draws))} distinct values")
    c.check(
        "even a jittered wait grows with the retry number",
        jittered.delay_for(1, rng=random.Random(7)) < jittered.delay_for(5, rng=random.Random(7)),
    )
    c.equal("jitter off gives the bare exponential", RetryPolicy(jitter=False).delay_for(4), 8.0)

    print("\n--- policy as configuration")
    custom = RetryPolicy(max_retries=3, base_delay=0.5, multiplier=3.0, max_delay=9.0, jitter=False)
    c.equal("roundtrips through a dict", RetryPolicy.from_dict(custom.to_dict()), custom)
    c.equal("an absent block is the defaults", RetryPolicy.from_dict(None), RetryPolicy())
    c.equal("a partial block fills in the rest", RetryPolicy.from_dict({"max_retries": 2}).base_delay, 1.0)

    c.equal("a valid policy has no complaints", validate_policy(RetryPolicy()), [])
    c.check("negative retries are rejected", validate_policy(RetryPolicy(max_retries=-1)))
    c.check("a zero base delay is rejected", validate_policy(RetryPolicy(base_delay=0)))
    c.check("a shrinking multiplier is rejected", validate_policy(RetryPolicy(multiplier=0.5)))
    c.check(
        "a cap below the base delay is rejected",
        validate_policy(RetryPolicy(base_delay=10.0, max_delay=1.0)),
    )


# ---- what counts as worth repeating ---------------------------------------


def check_classification(c: Checks) -> None:
    print("\n--- classifying failures")

    cases = [
        (APITimeoutError("took too long"), FailureType.MODEL_TIMEOUT, True),
        (APIConnectionError("connection reset"), FailureType.NETWORK_FAILURE, True),
        (ConnectionError("dns died"), FailureType.NETWORK_FAILURE, True),
        (TimeoutError("socket timeout"), FailureType.MODEL_TIMEOUT, True),
        (RateLimitError(), FailureType.RATE_LIMIT, True),
        (OverloadedError("529"), FailureType.RATE_LIMIT, True),
        (InternalServerError("500"), FailureType.NETWORK_FAILURE, True),
        (AuthenticationError("bad key"), FailureType.INVALID_REQUEST, False),
    ]
    for exc, expected_type, expected_retryable in cases:
        failure = classify(exc)
        c.equal(f"{type(exc).__name__} -> {expected_type.value}", failure.type, expected_type)
        c.equal(f"  ...and is {'retried' if expected_retryable else 'not retried'}",
                failure.retryable, expected_retryable)

    c.equal(
        "an unknown class is classified by its status code (503)",
        classify(WeirdProviderError(503)).type,
        FailureType.NETWORK_FAILURE,
    )
    c.check("  ...and retried", classify(WeirdProviderError(503)).retryable)
    c.equal(
        "an unknown 404 is classified as an invalid request",
        classify(WeirdProviderError(404)).type,
        FailureType.INVALID_REQUEST,
    )
    c.check("  ...and not retried", not classify(WeirdProviderError(404)).retryable)
    c.equal(
        "an unknown 429 is still a rate limit",
        classify(WeirdProviderError(429)).type,
        FailureType.RATE_LIMIT,
    )

    c.equal(
        "classification walks the MRO to the nearest known ancestor",
        classify(SubclassedTimeout("derived")).type,
        FailureType.MODEL_TIMEOUT,
    )

    unknown = classify(ValueError("a bug, not a blip"))
    c.equal("an unrecognised exception is UNKNOWN", unknown.type, FailureType.UNKNOWN)
    c.check("  ...and is never retried", not unknown.retryable)

    failure = classify(RuntimeError("boom"))
    c.equal("the class name is kept for the record", failure.exception, "RuntimeError")
    c.equal("so is the message", failure.error, "boom")
    c.check("and it serialises", isinstance(failure.to_dict()["retryable"], bool))

    print("\n--- sandbox failures are recorded, not retried here")
    # The decision this encodes: Sandbox._run destroys the container when a
    # command times out, so there is nothing left to call again. Recovering
    # from one means rebuilding from a checkpoint, which is a run-level
    # operation and not part of this slice.
    tool = classify(ToolTimeoutError("pytest timed out after 600s"))
    c.equal("a real ToolTimeoutError is named as one", tool.type, FailureType.TOOL_TIMEOUT)
    c.check("  ...and is not retried in place", not tool.retryable)


def check_retry_after(c: Checks) -> None:
    print("\n--- Retry-After")
    c.equal("read from the response headers", retry_after_seconds(RateLimitError(retry_after="7")), 7.0)
    c.equal("absent when the server sent none", retry_after_seconds(RateLimitError()), None)
    c.equal("absent on an exception with no response", retry_after_seconds(ValueError("x")), None)
    c.equal(
        "an HTTP-date is ignored rather than guessed at",
        retry_after_seconds(RateLimitError(retry_after="Wed, 21 Oct 2026 07:28:00 GMT")),
        None,
    )
    c.equal("a negative value is ignored", retry_after_seconds(RateLimitError(retry_after="-5")), None)

    policy = RetryPolicy(jitter=False)
    c.equal("it overrides the computed wait", policy.delay_for(1, retry_after=12.0), 12.0)
    c.equal(
        "and is capped, so a run cannot be parked for an hour",
        policy.delay_for(1, retry_after=9999.0),
        MAX_RETRY_AFTER,
    )
    c.equal(
        "ignored entirely when the policy says so",
        RetryPolicy(jitter=False, respect_retry_after=False).delay_for(2, retry_after=99.0),
        2.0,
    )


# ---- retry_call ------------------------------------------------------------


def check_retry_call(c: Checks) -> None:
    print("\n--- retry_call")
    policy = RetryPolicy(jitter=False)

    sleeper, log = Sleeper(), RetryLog()
    func = Flaky([])
    c.equal(
        "a call that works is called once",
        retry_call("op", func, policy, on_retry=log, sleep=sleeper),
        "ok",
    )
    c.equal("  ...with no retries", len(log), 0)
    c.equal("  ...and no waiting", sleeper.delays, [])

    sleeper, log = Sleeper(), RetryLog()
    func = Flaky([APITimeoutError("1"), APIConnectionError("2")])
    c.equal(
        "two transient failures then success returns the value",
        retry_call("op", func, policy, on_retry=log, sleep=sleeper),
        "ok",
    )
    c.equal("  ...after three attempts", func.calls, 3)
    c.equal("  ...with two retries recorded", len(log), 2)
    c.equal("  ...waiting 1s then 2s", sleeper.delays, [1.0, 2.0])
    c.equal(
        "  ...each named by what went wrong",
        [attempt.failure for attempt in log],
        [FailureType.MODEL_TIMEOUT.value, FailureType.NETWORK_FAILURE.value],
    )
    c.equal("  ...numbered from one", [attempt.retry for attempt in log], [1, 2])

    print("\n--- giving up")
    sleeper, log = Sleeper(), RetryLog()
    func = Flaky([APITimeoutError(f"attempt {i}") for i in range(20)])
    c.raises(
        "an operation that never recovers raises the original exception",
        APITimeoutError,
        lambda: retry_call("op", func, policy, on_retry=log, sleep=sleeper),
    )
    c.equal("  ...after exactly six attempts", func.calls, 6)
    c.equal("  ...having retried exactly five times", len(log), 5)
    c.equal("  ...with the full schedule", sleeper.delays, [1.0, 2.0, 4.0, 8.0, 16.0])
    c.equal("  ...and 31 seconds of waiting, never more", sleeper.total, 31.0)
    c.check(
        "  ...with no sleep after the final failure",
        len(sleeper.delays) == policy.max_retries,
        f"slept {len(sleeper.delays)} times for {policy.max_retries} retries",
    )

    print("\n--- failures that repeating cannot fix")
    sleeper, log = Sleeper(), RetryLog()
    func = Flaky([AuthenticationError("bad key")])
    c.raises(
        "a bad key fails immediately",
        AuthenticationError,
        lambda: retry_call("op", func, policy, on_retry=log, sleep=sleeper),
    )
    c.equal("  ...on the first attempt", func.calls, 1)
    c.equal("  ...with nothing retried", len(log), 0)
    c.equal("  ...and nothing waited", sleeper.delays, [])

    sleeper = Sleeper()
    func = Flaky([KeyboardInterrupt()])
    c.raises(
        "an interrupt is honoured rather than slept on",
        KeyboardInterrupt,
        lambda: retry_call("op", func, policy, sleep=sleeper),
    )
    c.equal("  ...immediately", sleeper.delays, [])

    print("\n--- a rate limit with a Retry-After")
    sleeper, log = Sleeper(), RetryLog()
    func = Flaky([RateLimitError(retry_after="4.5")])
    retry_call("op", func, policy, on_retry=log, sleep=sleeper)
    c.equal("the server's wait is used", sleeper.delays, [4.5])
    c.check("and flagged as the server's, not ours", log.attempts[0].from_retry_after)

    sleeper, log = Sleeper(), RetryLog()
    retry_call("op", Flaky([APITimeoutError("x")]), policy, on_retry=log, sleep=sleeper)
    c.check("our own backoff is not flagged that way", not log.attempts[0].from_retry_after)


def check_retry_log(c: Checks) -> None:
    print("\n--- the retry log")
    log = RetryLog()
    c.check("an empty log is falsey", not log)
    c.equal("and reports nothing", log.summary_line(), "no retries")
    c.equal("and serialises to an empty list", log.to_dicts(), [])

    for failure in ("model_timeout", "model_timeout", "rate_limit"):
        log(RetryAttempt(operation="messages.create", retry=1, failure=failure, error="e", delay_s=2.0))

    c.equal("counts by failure type", log.counts_by_failure(), {"model_timeout": 2, "rate_limit": 1})
    c.check("summarises in one line", "3 retries" in log.summary_line(), log.summary_line())
    c.check("reports what it waited", "6.0s" in log.summary_line(), log.summary_line())

    record = log.to_dicts()[0]
    c.equal("every attempt carries its operation", record["operation"], "messages.create")
    c.check("and when it happened", bool(record["at"]))
    c.check(
        "records are JSON-native",
        all(isinstance(value, (str, int, float, bool)) for value in record.values()),
        str(record),
    )

    long_error = RetryAttempt("op", 1, "unknown", "x" * 5000, 1.0).to_dict()["error"]
    c.check("a huge error message is truncated", len(long_error) == 500, f"{len(long_error)} chars")


# ---- the call site ---------------------------------------------------------


class FakeMessage:
    content: list = []
    stop_reason = "end_turn"
    usage = None
    stop_details = None


class FakeCount:
    input_tokens = 7


class FakeSDK:
    """The two methods ModelClient calls, and nothing else."""

    def __init__(self, create_failures=None, count_failures=None):
        failing_create = Flaky(list(create_failures or []), value=FakeMessage())
        failing_count = Flaky(list(count_failures or []), value=FakeCount())

        class Messages:
            @staticmethod
            def create(**_kwargs):
                return failing_create()

            @staticmethod
            def count_tokens(**_kwargs):
                return failing_count()

        self.messages = Messages()
        self.create_calls = failing_create
        self.count_calls = failing_count


def check_model_client(c: Checks) -> None:
    """Retry is only worth anything if it is actually wired into the call."""
    print("\n--- ModelClient")
    from Agent.client import ModelClient, ModelError
    from Agent.config import AgentConfig

    # Real sleeps, but microscopic ones: this exercises the production path
    # (ModelClient does not expose a sleep seam) without costing 31 seconds.
    fast = RetryPolicy(base_delay=0.001, multiplier=2.0, max_delay=0.01, jitter=False)
    config = AgentConfig(retry=fast)

    sdk = FakeSDK(create_failures=[APITimeoutError("slow"), RateLimitError()])
    client = ModelClient(config, client=sdk)
    response = client.generate([{"role": "user", "content": "hi"}], [], "sys")
    c.check("generate survives a timeout and a rate limit", response is not None)
    c.equal("  ...taking three attempts", sdk.create_calls.calls, 3)
    c.equal("  ...and recording both retries", len(client.retry_log), 2)
    c.equal(
        "  ...labelled with the operation that failed",
        {attempt.operation for attempt in client.retry_log},
        {"messages.create"},
    )

    sdk = FakeSDK(create_failures=[APITimeoutError("x")] * 20)
    client = ModelClient(config, client=sdk)
    c.raises(
        "an unrecoverable turn still ends as ModelError",
        ModelError,
        lambda: client.generate([], [], ""),
    )
    c.equal("  ...after six attempts", sdk.create_calls.calls, 6)
    c.equal("  ...with five retries on the record", len(client.retry_log), 5)

    sdk = FakeSDK(create_failures=[AuthenticationError("bad key")])
    client = ModelClient(config, client=sdk)
    c.raises("a bad key is not retried", ModelError, lambda: client.generate([], [], ""))
    c.equal("  ...costing one attempt", sdk.create_calls.calls, 1)
    c.equal("  ...and no retries", len(client.retry_log), 0)

    sdk = FakeSDK(count_failures=[APIConnectionError("flaky wifi")])
    client = ModelClient(config, client=sdk)
    c.equal("preflight retries too", client.preflight(), 7)
    c.equal("  ...and says so", len(client.retry_log), 1)

    c.equal(
        "a fresh client starts with an empty log, not a missing one",
        ModelClient(config, client=FakeSDK()).retry_log.to_dicts(),
        [],
    )

    print("\n--- one retry layer, not two")
    try:
        import anthropic
    except ImportError:
        print("  SKIP  the anthropic SDK is not installed")
        return

    seen: dict = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    original = anthropic.Anthropic
    anthropic.Anthropic = Recorder
    try:
        # Never sent anywhere: Recorder makes no request. The broker is
        # bypassed so this does not touch a real credential.
        ModelClient(AgentConfig(), api_key="not-a-real-key")
    finally:
        anthropic.Anthropic = original

    c.equal(
        "the SDK's own retries are disabled so counts stay honest",
        seen.get("max_retries"),
        0,
    )

    missing = [name for name in SDK_EXCEPTIONS if not hasattr(anthropic, name)]
    c.check(
        "every SDK exception the table names still exists",
        not missing,
        f"gone from anthropic {anthropic.__version__}: {', '.join(missing)}",
    )


def check_agent_config(c: Checks) -> None:
    print("\n--- the policy as agent configuration")
    from pathlib import Path

    from Agent.config import AgentConfig, load_config, validate_config

    config = AgentConfig()
    c.equal("an agent has a retry policy by default", config.retry, RetryPolicy())
    c.equal(
        "it roundtrips with the rest of the config",
        AgentConfig.from_dict(config.to_dict()).retry,
        config.retry,
    )
    c.equal(
        "a bad policy is reported through the config's own errors",
        validate_config(AgentConfig(retry=RetryPolicy(max_retries=-2))),
        ["retry: max_retries must be zero or more, got -2"],
    )

    shipped = Path(__file__).parent.parent / "Agent" / "config.yaml"
    if shipped.exists():
        policy = load_config(shipped).retry
        c.equal("the shipped config asks for five retries", policy.max_retries, 5)
        c.equal("  ...with the doubling schedule", policy.schedule(), [1.0, 2.0, 4.0, 8.0, 16.0])
        c.equal("  ...and no complaints", validate_policy(policy), [])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m Recovery.selftest")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="accepted for symmetry with the other packages; every check is already offline",
    )
    parser.parse_args(argv)

    # Every retry below is deliberate, so the warnings they emit are noise here
    # -- and being unbuffered on stderr, they would otherwise all land above the
    # first check rather than beside the one that caused them.
    logging.getLogger("recovery").setLevel(logging.CRITICAL)

    c = Checks()
    check_policy(c)
    check_classification(c)
    check_retry_after(c)
    check_retry_call(c)
    check_retry_log(c)
    check_model_client(c)
    check_agent_config(c)

    failures = c.failures
    print(f"\n{len(c.results) - len(failures)}/{len(c.results)} checks passed")
    for name, _, detail in failures:
        print(f"  FAIL  {name}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
