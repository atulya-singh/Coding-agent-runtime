"""The agent loop: send the task, run what the model asks for, send back what happened.

    generate -> tool_use blocks -> execute in the sandbox -> tool_result blocks -> generate

There is no separate planning call and no separate verification call. Both were
considered and left out: planning-vs-direct-execution is one of the experiments
plan.md Phase 13 exists to measure, so hardcoding it here would foreclose the
comparison, and a model asked to judge whether a tool did what it asked adds a
second opinion where there is already a fact -- the ToolResult, and the test run
the model can trigger itself. Principle 1 applies inside the loop, not only at
grading time.

Every exit is a named StopReason rather than a bare bool, because "the model
said it was done", "it ran out of turns" and "the API failed" are three
different events and Principle 5 turns on telling them apart.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from Execution.sandbox_tools import SandboxToolset
from Execution.tools.base import ToolResult
from Tasks.schema import Task

from .client import ModelClient, ModelError
from .config import AgentConfig
from .pricing import cost_usd
from .prompts import build_initial_messages, build_system_prompt
from .tools import FINISH_TOOL_NAME, build_tool_specs

#: Cap on one tool result reaching the model. The sandbox already truncates
#: command output, but a read_file or search_code can still be arbitrarily
#: large, and one of those can otherwise consume a whole context window.
MAX_TOOL_OUTPUT_CHARS = 20_000


class StopReason(str, Enum):
    #: The model called `finish`.
    FINISHED = "finished"
    #: The model stopped calling tools without saying it was done.
    FINISHED_IMPLICIT = "finished_implicit"
    MAX_TURNS = "max_turns"
    BUDGET = "budget"
    #: A turn was cut off by max_tokens, so its last tool call may be truncated;
    #: continuing from it would send the API a malformed request.
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    MODEL_ERROR = "model_error"
    #: A safety classifier declined the request. Deliberately in neither bucket
    #: below: the harness did not break and the agent did not decide anything,
    #: so the run measures nothing and belongs in neither rate.
    REFUSAL = "refusal"
    #: The container is gone -- Sandbox tears itself down when a command times
    #: out, since a killed `docker exec` leaves the process inside still running.
    #: Every later tool call would fail identically, so stop instead of spending
    #: the remaining turns talking to a dead environment.
    SANDBOX_GONE = "sandbox_gone"

    @property
    def is_agent_decision(self) -> bool:
        """Did the agent choose to stop, rather than being stopped?"""
        return self in (StopReason.FINISHED, StopReason.FINISHED_IMPLICIT)

    @property
    def is_infrastructure(self) -> bool:
        """Did the harness break, rather than the agent fail? (Principle 5.)

        MAX_TURNS and BUDGET are deliberately not here: those are limits this
        project chose, and a run that hits one is a real result about the model.
        """
        return self in (StopReason.MODEL_ERROR, StopReason.SANDBOX_GONE)


def zero_usage() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def add_usage(total: dict, delta: dict) -> dict:
    return {key: total.get(key, 0) + delta.get(key, 0) for key in zero_usage()}


def total_tokens(usage: dict) -> int:
    return sum(usage.get(key, 0) for key in zero_usage())


@dataclass
class AgentRun:
    """Everything one run produced. The record Phase 5 will checkpoint."""

    task_id: str
    model: str
    stop_reason: StopReason
    turns: int = 0
    #: Raw Anthropic message list, JSON-native and replayable as-is.
    messages: List[dict] = field(default_factory=list)
    #: Tool-call level history. Outputs are not duplicated here -- they are
    #: already in `messages` as tool_result blocks.
    tool_calls: List[dict] = field(default_factory=list)
    usage: dict = field(default_factory=zero_usage)
    cost_usd: Optional[float] = None
    #: What `finish` reported, or the model's last words if it stopped implicitly.
    summary: str = ""
    #: Why the run broke, for the reasons that are failures.
    error: str = ""

    @property
    def total_tokens(self) -> int:
        return total_tokens(self.usage)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "stop_reason": self.stop_reason.value,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "summary": self.summary,
            "error": self.error,
            "messages": self.messages,
        }


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Keep the head and the tail. A test run's verdict is in its last lines and
    a file's signature is in its first, so dropping either end loses the part
    the model actually needs."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    dropped = len(text) - limit
    return f"{text[:head]}\n...[truncated {dropped} chars]...\n{text[-tail:]}"


def format_tool_output(result: ToolResult) -> str:
    if not result.success:
        return result.error or "tool failed with no error message"
    output = result.output
    text = output if isinstance(output, str) else json.dumps(output, indent=2, default=str)
    # An empty tool_result is rejected by the API, and "it worked and said
    # nothing" is a real outcome for write_file.
    return truncate(text) if text.strip() else "(no output)"


def refusal_error(details: Optional[dict]) -> str:
    """Say which classifier declined and why, rather than only that one did."""
    details = details or {}
    category = details.get("category") or "unspecified"
    explanation = details.get("explanation") or ""
    return f"the model declined the request (category: {category}) {explanation}".strip()


def tool_result_block(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }


class AgentLoop:
    def __init__(
        self,
        task: Task,
        toolset: SandboxToolset,
        client: ModelClient,
        config: Optional[AgentConfig] = None,
    ):
        self.task = task
        self.toolset = toolset
        self.client = client
        self.config = config or client.config
        self.system = build_system_prompt(task)
        self.tool_specs = build_tool_specs(toolset)

    def run(self) -> AgentRun:
        messages = build_initial_messages(self.task)
        usage = zero_usage()
        tool_calls: List[dict] = []
        turns = 0

        def finish(reason: StopReason, summary: str = "", error: str = "") -> AgentRun:
            return AgentRun(
                task_id=self.task.task_id,
                model=self.config.model,
                stop_reason=reason,
                turns=turns,
                messages=messages,
                tool_calls=tool_calls,
                usage=usage,
                cost_usd=cost_usd(usage, self.config.pricing_for_model()),
                summary=summary,
                error=error,
            )

        while True:
            if turns >= self.config.max_turns:
                return finish(StopReason.MAX_TURNS)
            exceeded = self._budget_exceeded(usage)
            if exceeded:
                return finish(StopReason.BUDGET, error=exceeded)

            try:
                response = self.client.generate(messages, self.tool_specs, self.system)
            except ModelError as exc:
                return finish(StopReason.MODEL_ERROR, error=str(exc))

            usage = add_usage(usage, response.usage)
            messages.append({"role": "assistant", "content": response.content})
            turns += 1

            if response.stop_reason == "refusal":
                return finish(StopReason.REFUSAL, error=refusal_error(response.stop_details))

            if response.stop_reason == "max_tokens":
                return finish(StopReason.MAX_OUTPUT_TOKENS, error="model output hit max_tokens")

            blocks = response.tool_use_blocks()
            if not blocks:
                return finish(StopReason.FINISHED_IMPLICIT, summary=response.text())

            results = []
            summary = None
            for block in blocks:
                if block.get("name") == FINISH_TOOL_NAME:
                    summary = (block.get("input") or {}).get("summary", "")
                    results.append(tool_result_block(block["id"], "acknowledged"))
                    continue
                # Sequential even when the model asks for several at once: they
                # share one filesystem, and a read after a write must see it.
                result = self.toolset.call(block.get("name"), block.get("input") or {})
                tool_calls.append(
                    {
                        "turn": turns,
                        "tool": block.get("name"),
                        "arguments": block.get("input") or {},
                        "success": result.success,
                        "error": result.error,
                        "duration_ms": round(result.duration_ms, 2),
                        "metadata": result.metadata,
                    }
                )
                results.append(
                    tool_result_block(
                        block["id"], format_tool_output(result), is_error=not result.success
                    )
                )

            messages.append({"role": "user", "content": results})

            # Checked before `finish`: if the container died, nothing the model
            # says about being done can be acted on -- the diff can no longer be
            # collected -- and the run must be attributed to the harness.
            if self._sandbox_gone():
                return finish(
                    StopReason.SANDBOX_GONE,
                    summary=summary or "",
                    error="the sandbox was torn down (a command timed out)",
                )

            if summary is not None:
                return finish(StopReason.FINISHED, summary=summary)

    def _sandbox_gone(self) -> bool:
        """True once the toolset has no live container behind it.

        Read through getattr so the loop still runs against any object exposing
        `call`/`tool_specs` -- the scripted toolset the selftest uses has no
        sandbox at all, and should not need to grow a fake one.
        """
        sandbox = getattr(self.toolset, "sandbox", None)
        return sandbox is not None and getattr(sandbox, "container_id", "live") is None

    def _budget_exceeded(self, usage: dict) -> str:
        limit = self.config.max_total_tokens
        if limit is not None and total_tokens(usage) >= limit:
            return f"token budget exhausted: {total_tokens(usage)} >= {limit}"
        budget = self.config.max_cost_usd
        if budget is not None:
            spent = cost_usd(usage, self.config.pricing_for_model())
            if spent is not None and spent >= budget:
                return f"cost budget exhausted: ${spent:.4f} >= ${budget:.4f}"
        return ""
