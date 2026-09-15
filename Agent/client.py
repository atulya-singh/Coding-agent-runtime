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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

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


class ModelClient:
    """Wraps the Anthropic Messages API for one run.

    `client` is injectable so the loop can be exercised against a scripted model
    with no API key and no network -- the seam the selftest uses.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None,
        client: Any = None,
        broker: Optional[SecretBroker] = None,
    ):
        self.config = config or AgentConfig()
        if client is not None:
            self._client = client
        else:
            import anthropic  # imported lazily: the scripted path needs no SDK

            key = api_key or (broker or SecretBroker()).get_secret(API_KEY_NAME)
            self._client = anthropic.Anthropic(
                api_key=key,
                timeout=self.config.request_timeout,
            )

    def generate(self, messages: List[dict], tools: List[dict], system: str) -> ModelResponse:
        try:
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )
        except Exception as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc

        return ModelResponse(
            content=[_as_dict(block) for block in response.content],
            stop_reason=getattr(response, "stop_reason", "") or "",
            usage=_usage_dict(getattr(response, "usage", None)),
        )
