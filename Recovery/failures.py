"""What went wrong, named -- the vocabulary the rest of recovery is written in.

Two things live here and they are deliberately separate:

    what kind of failure this was      FailureType, for the record
    whether doing it again could help  Failure.retryable, for the policy

They are separate because most of the taxonomy is about failures nothing can
retry its way out of, and because "retry rate by failure type" (plan.md's
metrics) is only a question you can ask if both are written down per event.

Classification walks the exception's MRO by *class name* rather than using
isinstance. That keeps this module import-free: `Agent.client` imports the
anthropic SDK lazily so the scripted path needs no SDK at all, and a recovery
layer that dragged the SDK back in at import time would undo that. The status
code is consulted second, which covers provider errors this table has never
heard of -- a 503 from a class added next quarter is still a 503.

Unrecognised failures are deliberately *not* retryable. Retrying an unknown
exception five times turns one clear traceback into six, thirty-one seconds
later, and a programming error is not a transient one.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FailureType(str, Enum):
    # plan.md Phase 6's taxonomy, verbatim.
    MODEL_TIMEOUT = "model_timeout"
    TOOL_TIMEOUT = "tool_timeout"
    COMMAND_FAILURE = "command_failure"
    TEST_FAILURE = "test_failure"
    SANDBOX_CRASH = "sandbox_crash"
    WORKER_CRASH = "worker_crash"
    NETWORK_FAILURE = "network_failure"
    RESOURCE_LIMIT = "resource_limit"
    AGENT_LOOP = "agent_loop"
    INVALID_TOOL_CALL = "invalid_tool_call"

    # Three the taxonomy predates, added because folding them into the list
    # above would have made the record say something untrue:
    #: 429 and 529. Not a timeout (the call arrived) and not a network failure
    #: (the wire was fine); it is the one failure with a known, server-supplied
    #: answer to "when should I try again". RESOURCE_LIMIT is reserved for the
    #: sandbox's own CPU/memory/pid ceilings, which are a different event.
    RATE_LIMIT = "rate_limit"
    #: A 4xx that will fail identically every time -- a bad key, a mistyped
    #: model id, a malformed request. Distinct from Agent.loop's REFUSAL, which
    #: is the model declining a well-formed request.
    INVALID_REQUEST = "invalid_request"
    #: Nothing recognised it. Never retried; see the module docstring.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Failure:
    """One classified failure. JSON-native, like everything else on the record."""

    type: FailureType
    retryable: bool
    exception: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "retryable": self.retryable,
            "exception": self.exception,
            "error": self.error,
        }


#: Exception class name -> (what it was, is doing it again worth anything).
#: Keyed by name so this module never imports the SDK; the MRO walk below means
#: a subclass is classified as its nearest listed ancestor.
_BY_NAME = {
    # -- the model call did not complete in time ----------------------------
    "APITimeoutError": (FailureType.MODEL_TIMEOUT, True),
    "DeadlineExceededError": (FailureType.MODEL_TIMEOUT, True),
    "ReadTimeout": (FailureType.MODEL_TIMEOUT, True),
    "PoolTimeout": (FailureType.MODEL_TIMEOUT, True),
    "TimeoutException": (FailureType.MODEL_TIMEOUT, True),
    "TimeoutError": (FailureType.MODEL_TIMEOUT, True),
    # -- it never got there, or never got back ------------------------------
    "APIConnectionError": (FailureType.NETWORK_FAILURE, True),
    "ConnectTimeout": (FailureType.NETWORK_FAILURE, True),
    "ConnectError": (FailureType.NETWORK_FAILURE, True),
    "RemoteProtocolError": (FailureType.NETWORK_FAILURE, True),
    "ConnectionError": (FailureType.NETWORK_FAILURE, True),
    # A response that arrived but did not parse. Transient far more often than
    # not, and the alternative -- failing the whole run on one bad frame -- is
    # worse than one extra call.
    "APIResponseValidationError": (FailureType.NETWORK_FAILURE, True),
    # -- the provider said "not now" ----------------------------------------
    "RateLimitError": (FailureType.RATE_LIMIT, True),
    "OverloadedError": (FailureType.RATE_LIMIT, True),
    # The SDK's own marker for "this one is safe to repeat".
    "RetryableError": (FailureType.NETWORK_FAILURE, True),
    # -- the provider said no, and will say no again ------------------------
    "AuthenticationError": (FailureType.INVALID_REQUEST, False),
    "PermissionDeniedError": (FailureType.INVALID_REQUEST, False),
    "CredentialsError": (FailureType.INVALID_REQUEST, False),
    "BadRequestError": (FailureType.INVALID_REQUEST, False),
    "NotFoundError": (FailureType.INVALID_REQUEST, False),
    "UnprocessableEntityError": (FailureType.INVALID_REQUEST, False),
    "RequestTooLargeError": (FailureType.INVALID_REQUEST, False),
    # -- sandbox-side, recorded but not retried here ------------------------
    # A tool timeout has already cost the container: Sandbox._run destroys it,
    # because killing `docker exec` on the host leaves the process running
    # inside. So by the time this is classified there is nothing left to call
    # into, and repeating the call is not a recovery -- rebuilding from a
    # checkpoint is. That is a run-level operation and is not part of this
    # slice; `python -m Agent <task> --resume` does it by hand today.
    "ToolTimeoutError": (FailureType.TOOL_TIMEOUT, False),
    "SandboxToolError": (FailureType.COMMAND_FAILURE, False),
}

#: Anything not in _BY_NAME and carrying an HTTP status gets classified by it,
#: so a provider error class this table has never seen is still handled.
_RETRY_STATUS = {429: FailureType.RATE_LIMIT, 529: FailureType.RATE_LIMIT}


def classify(exc: BaseException) -> Failure:
    """Name a failure and say whether repeating it could plausibly work."""
    error = str(exc) or repr(exc)
    name = type(exc).__name__

    # Nearest listed ancestor wins: the SDK's APITimeoutError subclasses
    # APIConnectionError, and it appears first in its own MRO, so a timeout is
    # reported as a timeout rather than as the connection error it inherits.
    for klass in type(exc).__mro__:
        known = _BY_NAME.get(klass.__name__)
        if known is not None:
            failure_type, retryable = known
            return Failure(failure_type, retryable, name, error)

    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        if status in _RETRY_STATUS:
            return Failure(_RETRY_STATUS[status], True, name, error)
        if 500 <= status < 600:
            return Failure(FailureType.NETWORK_FAILURE, True, name, error)
        if 400 <= status < 500:
            return Failure(FailureType.INVALID_REQUEST, False, name, error)

    return Failure(FailureType.UNKNOWN, False, name, error)


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """The server's own answer to "when should I come back", if it gave one.

    Only the numeric form is read. The HTTP-date form is legal but the SDK does
    not send one, and guessing at a date parse to save a rare case would risk
    turning a 2-second wait into a 2-hour one on a clock skew.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except Exception:  # noqa: BLE001 -- a header bag that will not be read from
        return None
    if raw in (None, ""):
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None
