"""The model side of the loop. Host-side only -- never reachable from a container.

The API key is fetched through Sandbox.secrets.SecretBroker, the same broker the
sandbox package documents as the host-side-only path for credentials. That is
what makes the Phase 1 rule hold structurally rather than by convention: this
object is never constructed inside, passed into, or referenced by a Sandbox, so
there is no path by which the key reaches the container.

Responses are normalised to plain dicts here, at the one boundary where SDK
objects enter the system. Everything downstream -- the message list, the tool
history, and later the checkpoint on disk -- is then JSON-serialisable by
construction rather than by a conversion step someone has to remember.

This is also where Recovery's retry policy is applied. A timeout, a dropped
connection or a 429 is not a fact about the agent's ability to solve the task,
and letting one end a forty-turn run twenty turns in would put an environmental
hiccup in the middle of a capability measurement. Retries are absorbed here and
written down on `retry_log`, so the run continues and the interruption is still
on the record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from Recovery.retry import RetryLog, RetryPolicy, retry_call
from Sandbox.secrets import SecretBroker

from .config import AgentConfig

API_KEY_NAME = "ANTHROPIC_API_KEY"


class ModelError(Exception):
    """The model call failed. An infrastructure failure, not an agent one."""


@dataclass
class ModelResponse:
    content: List[dict] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)
    #: Populated only when stop_reason is "refusal"; None otherwise.
    stop_details: Optional[dict] = None

    def tool_use_blocks(self) -> List[dict]:
        return [block for block in self.content if block.get("type") == "tool_use"]

    def text(self) -> str:
        return "\n".join(
            block.get("text", "") for block in self.content if block.get("type") == "text"
        )


def _as_dict(block: Any) -> dict:
    if isinstance(block, dict):
        return block
    # SDK content blocks are pydantic models.
    return block.model_dump()


def _usage_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    raw = usage if isinstance(usage, dict) else usage.model_dump()
    return {
        "input_tokens": raw.get("input_tokens", 0) or 0,
        "output_tokens": raw.get("output_tokens", 0) or 0,
        "cache_creation_input_tokens": raw.get("cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": raw.get("cache_read_input_tokens", 0) or 0,
    }


def _stop_details(details: Any) -> Optional[dict]:
    """Only present on a refusal, and null for every other stop reason."""
    if details is None:
        return None
    return details if isinstance(details, dict) else details.model_dump()


class ModelClient:
    """Wraps the Anthropic Messages API for one run.

    `client` is injectable so the loop can be exercised against a scripted model
    with no API key and no network -- the seam the selftest uses.
    """

    @staticmethod
    def _resolve_key(api_key: Optional[str], broker: Optional[SecretBroker]) -> Optional[str]:
        """Ask the broker for the key, and let the SDK try if it has none.

        An unset ANTHROPIC_API_KEY does not mean there are no credentials: the
        SDK also reads ANTHROPIC_AUTH_TOKEN and the profile written by
        `ant auth login`. Returning None hands it that chain instead of failing
        on a key the broker was never going to hold.
        """
        if api_key is not None:
            return api_key
        try:
            return (broker or SecretBroker()).get_secret(API_KEY_NAME)
        except KeyError:
            return None

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None,
        client: Any = None,
        broker: Optional[SecretBroker] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        self.config = config or AgentConfig()
        self.retry_policy = retry_policy or self.config.retry
        #: Every retry this client made, in order. Read by AgentLoop onto the
        #: run record; a client that never retried carries an empty log rather
        #: than no log, so "we did not retry" and "we did not look" stay apart.
        self.retry_log = RetryLog()
        if client is not None:
            self._client = client
        else:
            import anthropic  # imported lazily: the scripted path needs no SDK

            self._client = anthropic.Anthropic(
                api_key=self._resolve_key(api_key, broker),
                timeout=self.config.request_timeout,
                # The SDK retries twice on its own by default. Left on, that
                # would nest inside our five and make one recorded retry mean
                # three real HTTP requests -- so every count this project
                # reports, and every backoff it claims to have waited, would be
                # wrong. One retry layer, and it is the one that writes things
                # down.
                max_retries=0,
            )

    def request_kwargs(self, messages: List[dict], tools: List[dict], system: str) -> dict:
        """Everything sent to the API, as one dict -- so the request that produced
        a result can be inspected and recorded rather than reconstructed."""
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "tools": tools,
            "messages": messages,
        }
        # Adaptive is the only on-mode the current models accept; the older
        # fixed `budget_tokens` form is rejected by them.
        if self.config.thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
        if self.config.effort:
            kwargs["output_config"] = {"effort": self.config.effort}
        return kwargs

    def preflight(self) -> int:
        """Prove the credentials and the model id work, before anything expensive.

        Token counting is free and does not generate, so this costs nothing but
        catches the failures worth catching early: no key, a key without access,
        a mistyped model id, no network. Without it those surface only after a
        container has been built and the task's setup has run.
        """
        response = self._call(
            "messages.count_tokens",
            lambda: self._client.messages.count_tokens(
                model=self.config.model,
                messages=[{"role": "user", "content": "ping"}],
            ),
        )
        return getattr(response, "input_tokens", 0)

    def _call(self, operation: str, func) -> Any:
        """One API call, retried per the policy, failures normalised to ModelError.

        Retrying preflight matters as much as retrying a turn: the whole reason
        preflight exists is to fail before a container is built, and a flaky
        network would otherwise make it the most likely thing to abort a run
        that was going to be fine.
        """
        spent_before = len(self.retry_log)
        try:
            return retry_call(
                operation, func, policy=self.retry_policy, on_retry=self.retry_log
            )
        except Exception as exc:
            # Kept as one type on purpose: from the loop's point of view every
            # one of these is "the turn did not happen". The class name is
            # preserved in the message so a rate limit stays distinguishable
            # from a bad request when the record is read back.
            spent = len(self.retry_log) - spent_before
            suffix = f" (after {spent} retr{'y' if spent == 1 else 'ies'})" if spent else ""
            raise ModelError(f"{type(exc).__name__}: {exc}{suffix}") from exc

    def generate(self, messages: List[dict], tools: List[dict], system: str) -> ModelResponse:
        response = self._call(
            "messages.create",
            lambda: self._client.messages.create(**self.request_kwargs(messages, tools, system)),
        )
        return ModelResponse(
            content=[_as_dict(block) for block in response.content],
            stop_reason=getattr(response, "stop_reason", "") or "",
            usage=_usage_dict(getattr(response, "usage", None)),
            stop_details=_stop_details(getattr(response, "stop_details", None)),
        )
