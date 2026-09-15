"""Prove the loop's control flow before spending a single real token.

Every exit path, budget and malformed-input case is driven here against a
scripted model and a fake toolset, so the whole thing runs with no Docker, no
API key and no network. That matters because the expensive half of this system
(a real model, a real container) is the worst place to discover that the loop
mishandles a truncated turn or forgets to answer a tool_use block.

    python -m Agent.selftest
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from Evaluation.selftest import Checks
from Execution.tools.base import ToolResult
from Tasks.schema import Difficulty, Task, TaskCategory

from .client import ModelError, ModelResponse
from .config import AgentConfig, validate_config
from .loop import (
    AgentLoop,
    StopReason,
    add_usage,
    format_tool_output,
    total_tokens,
    truncate,
    zero_usage,
)
from .pricing import ModelPricing, cost_usd
from .tools import FINISH_TOOL_NAME, build_tool_specs

# ---- doubles ---------------------------------------------------------------


class ScriptedClient:
    """Replays a fixed list of ModelResponses (or raises a queued exception)."""

    def __init__(self, responses: List, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.responses = list(responses)
        self.seen_messages: List[List[dict]] = []

    def generate(self, messages, tools, system) -> ModelResponse:
        self.seen_messages.append([dict(message) for message in messages])
        if not self.responses:
            raise ModelError("scripted client ran out of responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeToolset:
    """Stands in for SandboxToolset: same two methods the loop depends on."""

    SPECS = [
        {"name": "read_file", "description": "", "input_schema": {"type": "object"}},
        {"name": "run_tests", "description": "", "input_schema": {"type": "object"}},
    ]

    def __init__(self, results: Optional[dict] = None):
        self.results = results or {}
        self.calls: List[tuple] = []

    def tool_specs(self) -> list:
        return [dict(spec) for spec in self.SPECS]

    def call(self, tool_name, arguments=None) -> ToolResult:
        self.calls.append((tool_name, arguments or {}))
        if tool_name in self.results:
            return self.results[tool_name]
        return ToolResult(tool=tool_name, success=True, output=f"ran {tool_name}", duration_ms=1.0)


USAGE = {"input_tokens": 100, "output_tokens": 50}


def tool_turn(name: str, payload: dict, block_id: str = "tu_1") -> ModelResponse:
    return ModelResponse(
        content=[{"type": "tool_use", "id": block_id, "name": name, "input": payload}],
        stop_reason="tool_use",
        usage=dict(USAGE),
    )


def finish_turn(summary: str = "done") -> ModelResponse:
    return tool_turn(FINISH_TOOL_NAME, {"summary": summary}, block_id="tu_finish")


def text_turn(text: str = "all set") -> ModelResponse:
    return ModelResponse(
        content=[{"type": "text", "text": text}], stop_reason="end_turn", usage=dict(USAGE)
    )


def sample_task() -> Task:
    return Task(
        task_id="selftest-001",
        repository="https://github.com/example/example",
        commit="0" * 40,
        language="python",
        category=TaskCategory.BUG_FIX,
        difficulty=Difficulty.EASY,
        description="Fix the thing.",
        public_tests="pytest -q",
    )


def run_loop(responses, toolset=None, config=None):
    config = config or AgentConfig()
    toolset = toolset or FakeToolset()
    client = ScriptedClient(responses, config)
    loop = AgentLoop(sample_task(), toolset, client, config)
    return loop.run(), toolset, client


# ---- checks ----------------------------------------------------------------


def check_tool_specs(c: Checks) -> None:
    specs = build_tool_specs(FakeToolset())
    names = [spec["name"] for spec in specs]
    c.check(f"{FINISH_TOOL_NAME} is offered alongside the real tools", FINISH_TOOL_NAME in names)
    c.equal("no tool is offered twice", len(names), len(set(names)))

    class Colliding(FakeToolset):
        SPECS = [{"name": FINISH_TOOL_NAME, "description": "", "input_schema": {}}]

    c.raises(
        "a toolset defining its own `finish` is rejected rather than shadowed",
        ValueError,
        lambda: build_tool_specs(Colliding()),
    )


def check_finish_path(c: Checks) -> None:
    run, toolset, _ = run_loop([tool_turn("read_file", {"path": "a.py"}), finish_turn("fixed it")])
    c.equal("calling finish ends the run", run.stop_reason, StopReason.FINISHED)
    c.equal("the finish summary is kept", run.summary, "fixed it")
    c.equal("both turns are counted", run.turns, 2)
    c.equal("only the real tool call is recorded", len(run.tool_calls), 1)
    c.equal("the recorded call names the tool", run.tool_calls[0]["tool"], "read_file")
    c.check(
        "finish is never dispatched to the toolset",
        all(name != FINISH_TOOL_NAME for name, _ in toolset.calls),
        f"toolset saw {toolset.calls}",
    )
    c.check("the agent chose to stop", run.stop_reason.is_agent_decision)


def check_message_shape(c: Checks) -> None:
    run, _, _ = run_loop([tool_turn("read_file", {"path": "a.py"}), finish_turn()])
    roles = [message["role"] for message in run.messages]
    c.equal("messages alternate from the initial user turn", roles, ["user", "assistant", "user", "assistant", "user"])

    requested, answered = [], []
    for message in run.messages:
        content = message["content"]
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                requested.append(block["id"])
            if block.get("type") == "tool_result":
                answered.append(block["tool_use_id"])
    # An unanswered tool_use is rejected by the API on the very next call.
    c.equal("every tool_use gets exactly one tool_result", sorted(answered), sorted(requested))


def check_parallel_tool_calls(c: Checks) -> None:
    both = ModelResponse(
        content=[
            {"type": "tool_use", "id": "a", "name": "read_file", "input": {"path": "x"}},
            {"type": "tool_use", "id": "b", "name": "run_tests", "input": {}},
        ],
        stop_reason="tool_use",
        usage=dict(USAGE),
    )
    run, toolset, _ = run_loop([both, finish_turn()])
    c.equal("both tools in one turn run", [name for name, _ in toolset.calls], ["read_file", "run_tests"])
    c.equal("both are recorded", len(run.tool_calls), 2)
    # Only this turn's answers: the finish turn that follows is answered too.
    first_results = next(
        m["content"] for m in run.messages
        if isinstance(m["content"], list) and m["content"][0].get("type") == "tool_result"
    )
    c.equal("both are answered", sorted(block["tool_use_id"] for block in first_results), ["a", "b"])


def check_failed_tool(c: Checks) -> None:
    failing = ToolResult(tool="read_file", success=False, error="no such file: a.py")
    toolset = FakeToolset({"read_file": failing})
    run, _, _ = run_loop(
        [tool_turn("read_file", {"path": "a.py"}), finish_turn()], toolset=toolset
    )
    blocks = [b for m in run.messages if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"]
    failed_block = next(block for block in blocks if block["tool_use_id"] == "tu_1")
    c.check("a failed tool is flagged is_error to the model", failed_block["is_error"] is True)
    c.check("the error text is passed through", "no such file" in failed_block["content"])
    c.equal("a failed tool does not end the run", run.stop_reason, StopReason.FINISHED)
    c.check("the failure is recorded in the history", run.tool_calls[0]["success"] is False)


def check_implicit_finish(c: Checks) -> None:
    run, _, _ = run_loop([text_turn("I think that's everything")])
    c.equal("no tool call ends the run", run.stop_reason, StopReason.FINISHED_IMPLICIT)
    c.equal("the model's last words become the summary", run.summary, "I think that's everything")


def check_max_turns(c: Checks) -> None:
    config = AgentConfig(max_turns=3)
    never_done = [tool_turn("read_file", {"path": "a.py"}) for _ in range(10)]
    run, _, _ = run_loop(never_done, config=config)
    c.equal("the turn limit stops the run", run.stop_reason, StopReason.MAX_TURNS)
    c.equal("it stops at exactly the limit", run.turns, 3)
    c.check("being stopped is not an agent decision", not run.stop_reason.is_agent_decision)


def check_truncated_turn(c: Checks) -> None:
    cut_off = ModelResponse(
        content=[{"type": "tool_use", "id": "x", "name": "read_file", "input": {}}],
        stop_reason="max_tokens",
        usage=dict(USAGE),
    )
    run, toolset, _ = run_loop([cut_off, finish_turn()])
    c.equal("a truncated turn stops the run", run.stop_reason, StopReason.MAX_OUTPUT_TOKENS)
    c.equal("its possibly-incomplete tool call is not executed", toolset.calls, [])


def check_model_error(c: Checks) -> None:
    run, _, _ = run_loop([ModelError("503 overloaded")])
    c.equal("an API failure stops the run", run.stop_reason, StopReason.MODEL_ERROR)
    c.check("the failure is reported", "503" in run.error)


def check_token_budget(c: Checks) -> None:
    config = AgentConfig(max_turns=50, max_total_tokens=200)
    run, _, _ = run_loop([tool_turn("read_file", {"path": "a.py"})] * 10, config=config)
    c.equal("the token budget stops the run", run.stop_reason, StopReason.BUDGET)
    c.check("the reason says which budget", "token" in run.error, run.error)
    c.check("it stops at the budget, not long after", run.total_tokens < 400, str(run.usage))


def check_cost_budget(c: Checks) -> None:
    config = AgentConfig(
        max_turns=50,
        max_cost_usd=0.001,
        pricing={"claude-sonnet-5": ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)},
    )
    run, _, _ = run_loop([tool_turn("read_file", {"path": "a.py"})] * 10, config=config)
    c.equal("the cost budget stops the run", run.stop_reason, StopReason.BUDGET)
    c.check("the reason says which budget", "cost" in run.error, run.error)
    c.check("the run reports what it spent", run.cost_usd is not None and run.cost_usd > 0)


def check_usage_accounting(c: Checks) -> None:
    total = add_usage(zero_usage(), {"input_tokens": 10, "output_tokens": 5})
    total = add_usage(total, {"input_tokens": 1, "cache_read_input_tokens": 7})
    c.equal("input tokens accumulate", total["input_tokens"], 11)
    c.equal("cache reads accumulate", total["cache_read_input_tokens"], 7)
    c.equal("every kind of token counts toward the total", total_tokens(total), 23)

    run, _, _ = run_loop([tool_turn("read_file", {"path": "a"}), finish_turn()])
    c.equal("the run sums usage across turns", run.usage["input_tokens"], 200)


def check_pricing(c: Checks) -> None:
    rates = ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)
    c.equal(
        "cost is per million tokens",
        cost_usd({"input_tokens": 1_000_000, "output_tokens": 0}, rates),
        3.0,
    )
    c.equal(
        "input and output are priced separately",
        cost_usd({"input_tokens": 1_000_000, "output_tokens": 1_000_000}, rates),
        18.0,
    )
    c.equal(
        "cache reads fall back to the input rate when unpriced",
        cost_usd({"cache_read_input_tokens": 1_000_000}, rates),
        3.0,
    )
    # An unknown cost and a free call are different measurements.
    c.equal("an unpriced model reports no cost rather than zero", cost_usd(USAGE, None), None)


def check_config_guards(c: Checks) -> None:
    unpriced = AgentConfig(max_cost_usd=5.0)
    errors = validate_config(unpriced)
    c.check(
        "a cost budget without rates is rejected as unenforceable",
        any("pricing" in error for error in errors),
        str(errors),
    )
    priced = AgentConfig(
        max_cost_usd=5.0,
        pricing={"claude-sonnet-5": ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)},
    )
    c.equal("the same budget is fine once rates exist", validate_config(priced), [])


def check_output_handling(c: Checks) -> None:
    c.equal("short output is untouched", truncate("hello", 100), "hello")
    long_text = "A" * 500 + "TAIL"
    cut = truncate(long_text, 100)
    c.check("truncation keeps the head", cut.startswith("A"))
    c.check("truncation keeps the tail, where a test verdict lives", cut.endswith("TAIL"))
    c.check("truncation says how much it dropped", "truncated" in cut)

    c.equal(
        "empty output is never sent as an empty tool_result",
        format_tool_output(ToolResult(tool="write_file", success=True, output="")),
        "(no output)",
    )
    c.check(
        "structured output is serialised",
        '"exit_code": 0'
        in format_tool_output(ToolResult(tool="run_tests", success=True, output={"exit_code": 0})),
    )
    c.equal(
        "a failure sends its error",
        format_tool_output(ToolResult(tool="read_file", success=False, error="boom")),
        "boom",
    )


def check_prompt(c: Checks) -> None:
    _, _, client = run_loop([finish_turn()])
    system = AgentLoop(sample_task(), FakeToolset(), client).system
    c.check("the system prompt names the test command", "pytest -q" in system)
    c.check("the task description is the first user message", client.seen_messages[0][0]["content"] == "Fix the thing.")


def run_offline(c: Checks) -> None:
    check_tool_specs(c)
    check_finish_path(c)
    check_message_shape(c)
    check_parallel_tool_calls(c)
    check_failed_tool(c)
    check_implicit_finish(c)
    check_max_turns(c)
    check_truncated_turn(c)
    check_model_error(c)
    check_token_budget(c)
    check_cost_budget(c)
    check_usage_accounting(c)
    check_pricing(c)
    check_config_guards(c)
    check_output_handling(c)
    check_prompt(c)


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    for name in ("agent.tools", "sandbox", "sandbox.docker"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    c = Checks()
    print("=== agent loop checks (no Docker, no API key)")
    run_offline(c)

    total = len(c.results)
    failed = len(c.failures)
    print(f"\n{total - failed}/{total} checks passed")
    if failed:
        return 1
    print("every exit path, budget and malformed turn is handled as intended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
