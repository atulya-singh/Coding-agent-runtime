"""Show what the retry policy would actually do, without doing any of it.

    python -m Recovery                              # the built-in defaults
    python -m Recovery --config Agent/config.yaml   # what the agent really uses
    python -m Recovery --max-retries 3 --base-delay 2 --no-jitter

Read-only: no container, no network, no API key. It exists because a backoff
schedule is easy to misconfigure and impossible to eyeball from the numbers --
"multiplier 3, max_delay 30, 5 retries" is 1+3+9+27+30 seconds, and knowing
that before a run rather than during one is the whole point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .context import ContextPolicy, validate_context_policy
from .failures import FailureType, _BY_NAME
from .retry import MAX_RETRY_AFTER, MIN_JITTER, RetryPolicy, validate_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m Recovery", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--config", metavar="FILE", help="read the `retry:` block from a YAML config")
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--base-delay", type=float)
    parser.add_argument("--multiplier", type=float)
    parser.add_argument("--max-delay", type=float)
    parser.add_argument("--no-jitter", action="store_true", help="show the bare exponential")
    parser.add_argument(
        "--taxonomy", action="store_true", help="also list every failure type and its verdict"
    )
    return parser


def _blocks_from_config(path: Path) -> dict:
    """Read the `retry:` and `context:` blocks.

    Deliberately not via Agent.config: Agent imports this package, and an
    inspector reaching back the other way would put a cycle in the graph for
    the sake of two dictionaries.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load(args) -> RetryPolicy:
    data = _blocks_from_config(Path(args.config)) if args.config else {}
    policy = RetryPolicy.from_dict(data.get("retry"))
    if args.max_retries is not None:
        policy.max_retries = args.max_retries
    if args.base_delay is not None:
        policy.base_delay = args.base_delay
    if args.multiplier is not None:
        policy.multiplier = args.multiplier
    if args.max_delay is not None:
        policy.max_delay = args.max_delay
    if args.no_jitter:
        policy.jitter = False
    return policy


def print_schedule(policy: RetryPolicy, source: str) -> None:
    print(f"retry policy  ({source})")
    print(
        f"  max_retries {policy.max_retries}   base {policy.base_delay}s   "
        f"x{policy.multiplier}   cap {policy.max_delay}s   "
        f"jitter {'on' if policy.jitter else 'off'}   "
        f"retry-after {'honoured' if policy.respect_retry_after else 'ignored'}"
    )
    print(f"\n  {policy.max_attempts} attempts at most -- one call plus {policy.max_retries} retries\n")

    elapsed = 0.0
    print("  attempt   wait before it        cumulative wait")
    print("  --------  --------------------  ---------------")
    print(f"  {1:<8}  {'(none)':<20}  {0.0:>13.1f}s")
    for retry, delay in enumerate(policy.schedule(), start=1):
        elapsed += delay
        if policy.jitter:
            window = f"{delay * MIN_JITTER:.1f}-{delay:.1f}s"
        else:
            window = f"{delay:.1f}s"
        print(f"  {retry + 1:<8}  {window:<20}  {elapsed:>13.1f}s")

    print(f"\n  worst case: {policy.worst_case_wait():.1f}s of waiting before an operation is given up on")
    if policy.jitter:
        print(f"  with jitter: between {policy.worst_case_wait() * MIN_JITTER:.1f}s and that")
    print(f"  a server-sent Retry-After overrides the wait, capped at {MAX_RETRY_AFTER:.0f}s")


#: Reached by the status-code fallback rather than by an exception name, so a
#: listing keyed on names alone would wrongly show them as unreachable.
_BY_STATUS_CODE = {
    FailureType.RATE_LIMIT: "any 429 or 529",
    FailureType.NETWORK_FAILURE: "any other 5xx",
    FailureType.INVALID_REQUEST: "any other 4xx",
    FailureType.UNKNOWN: "anything unrecognised -- never retried, on purpose",
}


def print_taxonomy() -> None:
    print("\nfailure types")
    by_type: dict = {}
    for name, (failure_type, retryable) in _BY_NAME.items():
        by_type.setdefault(failure_type, [retryable, []])[1].append(name)

    for failure_type in FailureType:
        entry = by_type.get(failure_type)
        fallback = _BY_STATUS_CODE.get(failure_type)
        if entry is None and fallback is None:
            print(f"  {failure_type.value:<18} no call site produces this yet")
            continue
        verdict = "retried" if entry and entry[0] else "recorded, not retried"
        print(f"  {failure_type.value:<18} {verdict}")
        sources = sorted(entry[1]) if entry else []
        if fallback:
            sources.append(fallback)
        print(f"                     {', '.join(sources)}")


def print_context(policy: ContextPolicy) -> None:
    print("\ncontext (Strategy B: tell the agent what it did)")
    if not policy.enabled:
        print("  off -- failures still reach the model as tool results, but the")
        print("  harness never comments and never ends a run for looping.")
        return

    print(
        f"  repeat_threshold {policy.repeat_threshold}   "
        f"failure_threshold {policy.failure_threshold}   "
        f"max_nudges {policy.max_nudges}   window {policy.window}   "
        f"schema checks {'on' if policy.validate_tool_calls else 'off'}"
    )
    print("\n  identical call, identical result:")
    for repeat in range(1, policy.repeat_threshold + policy.max_nudges + 1):
        if repeat < policy.repeat_threshold:
            verdict = "silent"
        elif repeat - policy.repeat_threshold < policy.max_nudges:
            verdict = f"told (#{repeat - policy.repeat_threshold + 1})"
        else:
            verdict = "run ends as agent_loop, if the result was a failure"
        print(f"    call {repeat:<3} {verdict}")
    print(
        f"\n  a repeat only counts when the result is identical too, so the\n"
        f"  edit-test-edit cycle is never flagged; window is {policy.window} calls."
    )


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    policy = _load(args)
    context = ContextPolicy.from_dict(
        _blocks_from_config(Path(args.config)).get("context") if args.config else None
    )

    errors = [f"retry: {error}" for error in validate_policy(policy)]
    errors += [f"context: {error}" for error in validate_context_policy(context)]
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2

    print_schedule(policy, args.config or "built-in defaults")
    print_context(context)
    if args.taxonomy:
        print_taxonomy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
