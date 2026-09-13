from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional

DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0

logger = logging.getLogger("agent.tools")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class ToolTimeoutError(Exception):
    pass


class ToolExecutionError(Exception):
    pass


@dataclass
class ToolResult:
    tool: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def log_tool_event(tool_name: str, success: bool, duration_ms: float, error: Optional[str], metadata: dict) -> None:
    """Emit the one canonical `tool_call` log line.

    Public so that tool implementations which can't use the @tool decorator (e.g.
    the dispatcher rejecting an unknown tool name) still produce an identical event.
    """
    event = {
        "event": "tool_call",
        "tool": tool_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round(duration_ms, 2),
        "success": success,
    }
    if error:
        event["error"] = error
    if metadata:
        event["metadata"] = metadata
    (logger.info if success else logger.error)(json.dumps(event))


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> Any:
    # For pure-Python calls with no native timeout. Subprocess-based tools should
    # use subprocess's own `timeout` instead, since only that actually kills the
    # underlying process rather than leaving an orphaned thread running.
    kwargs = kwargs or {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise ToolTimeoutError(f"operation timed out after {timeout}s") from exc


def tool(name: str) -> Callable:
    """Wrap a function into a logged call that always returns a ToolResult."""

    def decorator(func: Callable[..., Any]) -> Callable[..., ToolResult]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> ToolResult:
            start = time.monotonic()
            try:
                result = func(*args, **kwargs)
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                    output, metadata = result
                else:
                    output, metadata = result, {}
                duration_ms = (time.monotonic() - start) * 1000
                log_tool_event(name, True, duration_ms, None, metadata)
                return ToolResult(tool=name, success=True, output=output, duration_ms=duration_ms, metadata=metadata)
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                log_tool_event(name, False, duration_ms, str(exc), {})
                return ToolResult(tool=name, success=False, error=str(exc), duration_ms=duration_ms)

        return wrapper

    return decorator
