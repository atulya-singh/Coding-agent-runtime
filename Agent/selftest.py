"""Prove the loop's control flow before spending a single real token.

Every exit path, budget and malformed-input case is driven here against a
scripted model and a fake toolset, so the whole thing runs with no Docker, no
API key and no network. That matters because the expensive half of this system
(a real model, a real container) is the worst place to discover that the loop
mishandles a truncated turn or forgets to answer a tool_use block.

The live half then runs the whole thing for real -- container, setup, toolhost,
tool calls, diff, grade -- with a scripted agent that replays the task's known
upstream fix. It needs Docker but still no API key: if that run does not grade
as solved, the wiring is wrong, because the patch is correct by construction.

    python -m Agent.selftest              # offline checks, then one live task
    python -m Agent.selftest --offline    # no Docker, no network
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from Evaluation.result import Outcome
from Evaluation.selftest import Checks
from Execution.tools.base import ToolResult
from Recovery.context import ContextPolicy
from Recovery.retry import RetryAttempt, RetryLog
from Tasks.harness import DEFAULT_REPO_CACHE, ensure_repo, file_at_commit
from Tasks.loader import discover_task_dirs, load_task
from Sandbox.secrets import SecretBroker
from Tasks.schema import Difficulty, Task, TaskCategory

from .client import ModelClient, ModelError, ModelResponse
from .config import DEFAULT_MODEL, AgentConfig, validate_config
from .loop import (
    IN_PROGRESS,
    AgentLoop,
    Snapshot,
    StopReason,
    add_usage,
    format_tool_output,
    total_tokens,
    truncate,
    zero_usage,
)
from .pricing import ModelPricing, cost_usd
from .replay import SCRIPTED_USAGE as USAGE
from .replay import ScriptedClient, finish_turn, replay_fix, text_turn, tool_turn
from .run import RunConfig, TaskRun, run_task
from .tools import FINISH_TOOL_NAME, build_tool_specs

DATASET_DIR = Path(__file__).parent.parent / "Tasks" / "dataset"

# ---- doubles ---------------------------------------------------------------


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


class DyingToolset(FakeToolset):
    """A FakeToolset whose sandbox tears itself down on the first call.

    Sandbox.run_command destroys the container when a command times out, because
    a killed `docker exec` leaves the process inside still running. Everything
    after that point fails identically, which is the case this stands in for.
    """

    class _Sandbox:
        container_id = "abc123"

    def __init__(self):
        super().__init__()
        self.sandbox = self._Sandbox()

    def call(self, tool_name, arguments=None) -> ToolResult:
        result = super().call(tool_name, arguments)
        self.sandbox.container_id = None
        return ToolResult(tool=tool_name, success=False, error="command timed out")


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


def run_loop(responses, toolset=None, config=None, on_turn=None, resume_from=None):
    config = config or AgentConfig()
    toolset = toolset or FakeToolset()
    client = ScriptedClient(responses, config)
    loop = AgentLoop(
        sample_task(), toolset, client, config, on_turn=on_turn, resume_from=resume_from
    )
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
        pricing={DEFAULT_MODEL: ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)},
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
        pricing={DEFAULT_MODEL: ModelPricing(input_per_mtok=3.0, output_per_mtok=15.0)},
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


def check_refusal(c: Checks) -> None:
    """A declined request is not an agent that decided it was finished."""
    refused = ModelResponse(
        content=[],
        stop_reason="refusal",
        usage=dict(USAGE),
        stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"},
    )
    run, _, _ = run_loop([refused, finish_turn()])
    c.equal("a refusal is its own stop reason", run.stop_reason, StopReason.REFUSAL)
    c.check("it is not read as the agent finishing", not run.stop_reason.is_agent_decision)
    c.check("it is not blamed on the harness", not run.stop_reason.is_infrastructure)
    c.check("the category is recorded", "cyber" in run.error)


def check_request_shape(c: Checks) -> None:
    """What actually goes on the wire, for the parameters the model rejects if
    they are wrong."""
    client = ModelClient(AgentConfig(), client=object())
    kwargs = client.request_kwargs([{"role": "user", "content": "hi"}], [], "sys")
    c.equal("adaptive thinking is requested", kwargs["thinking"], {"type": "adaptive"})
    c.equal("effort rides inside output_config", kwargs["output_config"], {"effort": "high"})
    c.check("the deprecated token budget is never sent", "budget_tokens" not in str(kwargs))

    off = ModelClient(AgentConfig(thinking="off", effort=""), client=object())
    bare = off.request_kwargs([], [], "")
    c.check("turning them off omits them entirely", "thinking" not in bare and "output_config" not in bare)

    c.equal(
        "a missing key falls through to the SDK's own credential chain",
        ModelClient._resolve_key(None, SecretBroker(lambda name: None)),
        None,
    )
    c.equal(
        "a brokered key is used when there is one",
        ModelClient._resolve_key(None, SecretBroker(lambda name: "sk-test")),
        "sk-test",
    )

    # Catches SDK drift without spending a request: a parameter the installed
    # SDK does not accept is a 400 at the worst possible moment otherwise.
    try:
        import anthropic
    except ImportError:
        c.check("the SDK is not installed, so its signature is unchecked", True)
        return
    import inspect

    accepted = set(
        inspect.signature(anthropic.Anthropic(api_key="placeholder").messages.create).parameters
    )
    c.equal(
        "every parameter sent is one the installed SDK accepts",
        sorted(set(kwargs) - accepted),
        [],
    )


def check_sandbox_gone(c: Checks) -> None:
    """A dead container must end the run, not be talked to for 39 more turns."""
    toolset = DyingToolset()
    run, _, client = run_loop(
        [tool_turn("run_tests", {"command": "pytest -q"}), finish_turn(), finish_turn()],
        toolset=toolset,
    )
    c.equal("a torn-down sandbox stops the run", run.stop_reason, StopReason.SANDBOX_GONE)
    c.equal("it stops on the turn it happened", run.turns, 1)
    c.check("the model is not asked again", len(client.seen_messages) == 1)
    c.check("the failed call is still answered", run.messages[-1]["role"] == "user")
    c.check("the reason says what broke", "torn down" in run.error)


def check_stop_reason_attribution(c: Checks) -> None:
    """Principle 5 lives or dies on this split, so state it as a test."""
    agent = {reason for reason in StopReason if reason.is_agent_decision}
    infra = {reason for reason in StopReason if reason.is_infrastructure}
    c.equal(
        "the agent's own decisions are exactly finish + implicit finish",
        agent,
        {StopReason.FINISHED, StopReason.FINISHED_IMPLICIT},
    )
    c.equal(
        "infrastructure failures are exactly the model error and the dead sandbox",
        infra,
        {StopReason.MODEL_ERROR, StopReason.SANDBOX_GONE},
    )
    c.check("nothing is counted as both", not (agent & infra))
    c.check(
        "hitting a limit we chose is a result, not a breakage",
        not StopReason.MAX_TURNS.is_infrastructure
        and not StopReason.BUDGET.is_infrastructure,
    )


def check_agent_record(c: Checks) -> None:
    """What the grader will record about who attempted the task."""
    run, _, _ = run_loop([tool_turn("read_file", {"path": "a.py"}), finish_turn("fixed it")])
    record = TaskRun(task_id="t", model=run.model, agent=run).agent_record()
    c.equal("the record names the model", record["model"], run.model)
    c.equal("it carries the stop reason", record["stop_reason"], "finished")
    c.check("it separates a decision from a limit", record["agent_decided_to_stop"] is True)
    c.equal("it counts real tool calls only", record["tool_calls"], 1)
    c.equal("it reports unknown cost as unknown", record["cost_usd"], None)
    c.check("an empty run still produces a record", "model" in TaskRun(task_id="t").agent_record())


def check_run_infrastructure(c: Checks) -> None:
    """A broken harness must come back as a record, not an exception: a crash
    here would be indistinguishable from an agent that failed the task."""
    run = run_task(
        sample_task(),
        Path("/nonexistent-task-dir"),
        client=ScriptedClient([finish_turn()]),
        repo=Path("/nonexistent-repo"),
        config=RunConfig(quiet=True),
    )
    c.check("a missing repository does not raise", isinstance(run, TaskRun))
    c.check("it is recorded as infrastructure", bool(run.infrastructure_error))
    c.check("no agent result is invented", run.agent is None)
    c.check("the attempt is not counted as fair", not run.ok)
    c.check("it is still serialisable", isinstance(run.to_dict(), dict))


def check_snapshots(c: Checks) -> None:
    """What the loop offers up for persisting, and when."""
    seen: List[Snapshot] = []
    run, _, _ = run_loop(
        [tool_turn("read_file", {"path": "a.py"}), finish_turn("fixed it")],
        on_turn=seen.append,
    )
    c.equal("a snapshot is taken before the first turn", seen[0].step, 0)
    c.equal("the first one is not yet an ending", seen[0].status, IN_PROGRESS)
    c.equal("one per turn boundary, plus the terminal one", [s.step for s in seen], [0, 1, 2])
    c.equal("the last one carries the stop reason", seen[-1].status, StopReason.FINISHED.value)
    c.check("the last one is marked terminal", seen[-1].is_terminal)
    c.check("only the last one is", not any(s.is_terminal for s in seen[:-1]))
    c.equal("it carries the summary the run ended with", seen[-1].summary, "fixed it")
    c.equal("the terminal snapshot matches the run", seen[-1].usage, run.usage)

    # A snapshot has to be resumable, and a message list with an unanswered
    # tool_use is rejected by the API -- so every one must be balanced.
    for snapshot in seen:
        requested = [
            block["id"]
            for message in snapshot.messages
            if isinstance(message["content"], list)
            for block in message["content"]
            if block.get("type") == "tool_use"
        ]
        answered = [
            block["tool_use_id"]
            for message in snapshot.messages
            if isinstance(message["content"], list)
            for block in message["content"]
            if block.get("type") == "tool_result"
        ]
        if not c.equal(
            f"the snapshot at step {snapshot.step} answers every tool call it shows",
            sorted(answered),
            sorted(requested),
        ):
            break

    mid_run: List[Snapshot] = []
    run, _, _ = run_loop([tool_turn("read_file", {"path": "a.py"})] * 5, config=AgentConfig(max_turns=2), on_turn=mid_run.append)
    c.equal("a run that is stopped still records why", mid_run[-1].status, StopReason.MAX_TURNS.value)


def check_resume(c: Checks) -> None:
    """Picking a run up again, with no checkpoint format involved."""
    prior = {
        "messages": [
            {"role": "user", "content": "Fix the thing."},
            {"role": "assistant", "content": [{"type": "text", "text": "looking"}]},
            {"role": "user", "content": "carry on"},
        ],
        "tool_calls": [{"turn": 1, "tool": "read_file", "success": True}],
        "usage": {"input_tokens": 500, "output_tokens": 100},
        "turns": 7,
    }
    run, _, client = run_loop([finish_turn("done now")], resume_from=prior)
    c.equal("the earlier conversation is carried in", client.seen_messages[0][:3], prior["messages"])
    c.equal("turns continue rather than restart", run.turns, 8)
    c.equal("earlier tool calls are still on the record", len(run.tool_calls), 1)
    c.equal("tokens already spent still count", run.usage["input_tokens"], 500 + USAGE["input_tokens"])

    # The budget bounds the attempt at the task, not the process making it, so a
    # resumed run must not be handed a fresh allowance.
    exhausted = dict(prior, turns=40)
    run, _, _ = run_loop([finish_turn()], config=AgentConfig(max_turns=40), resume_from=exhausted)
    c.equal(
        "a resumed run does not get its turn budget back",
        run.stop_reason,
        StopReason.MAX_TURNS,
    )


def check_retries_on_the_record(c: Checks) -> None:
    """A retry the client absorbed still has to reach the run record.

    The loop never sees one -- Recovery swallows it inside ModelClient, which
    is the point: a rate limit should not end a forty-turn run. But a harness
    that hides its own interruptions makes every environment look healthy, so
    the one thing that must not be silent is the record.
    """

    class RetryingClient(ScriptedClient):
        """A scripted client that also claims to have retried."""

        def __init__(self, responses, config, retries):
            super().__init__(responses, config)
            self.retry_log = RetryLog(retries)

    timeout = RetryAttempt("messages.create", 1, "model_timeout", "slow", 1.0)
    limited = RetryAttempt("messages.create", 2, "rate_limit", "429", 2.0)
    config = AgentConfig()

    seen: List[Snapshot] = []
    client = RetryingClient([finish_turn("done")], config, [timeout, limited])
    run = AgentLoop(sample_task(), FakeToolset(), client, config, on_turn=seen.append).run()
    c.equal("the client's retries reach the run record", len(run.retries), 2)
    c.equal(
        "each named by what went wrong",
        [entry["failure"] for entry in run.retries],
        ["model_timeout", "rate_limit"],
    )
    c.equal("every snapshot carries them too", [len(s.retries) for s in seen], [2, 2])
    c.equal("they survive serialisation", len(run.to_dict()["retries"]), 2)
    c.equal(
        "and the grader is told how many there were",
        TaskRun(task_id="t", agent=run).agent_record()["retries"],
        2,
    )

    clean, _, _ = run_loop([finish_turn()])
    c.equal("a client that never retried reports none, not nothing", clean.retries, [])
    c.equal("and the grader sees a zero", TaskRun(task_id="t", agent=clean).agent_record()["retries"], 0)

    prior = {
        "messages": [{"role": "user", "content": "Fix the thing."}],
        "turns": 3,
        "retries": [timeout.to_dict()],
    }
    client = RetryingClient([finish_turn()], config, [limited])
    resumed = AgentLoop(
        sample_task(), FakeToolset(), client, config, resume_from=prior
    ).run()
    c.equal("a resumed run keeps the retries from before the crash", len(resumed.retries), 2)
    c.equal(
        "oldest first, so the record reads in run order",
        [entry["failure"] for entry in resumed.retries],
        ["model_timeout", "rate_limit"],
    )


class StrictToolset(FakeToolset):
    """A FakeToolset whose specs actually document their arguments, so a call
    can be wrong in the way a real one can."""

    SPECS = [
        {
            "name": "read_file",
            "description": "",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "edit_file",
            "description": "",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    ]


def _edit_turn(index: int) -> ModelResponse:
    return tool_turn(
        "edit_file",
        {"path": "a.py", "old_string": "x", "new_string": "y"},
        f"tu_edit_{index}",
    )


def _blocks(run, role: str, kind: str) -> List[dict]:
    return [
        block
        for message in run.messages
        if message["role"] == role and isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == kind
    ]


def check_invalid_tool_calls(c: Checks) -> None:
    """A call that cannot work is answered with the schema, not dispatched."""
    toolset = StrictToolset()
    run, _, _ = run_loop(
        [tool_turn("read_file", {"file": "a.py"}), finish_turn("ok")], toolset=toolset
    )
    c.equal("a call that does not fit its schema never reaches the toolset", toolset.calls, [])
    c.equal("it is still on the record as a call the model made", len(run.tool_calls), 1)
    c.check("marked as never dispatched", run.tool_calls[0]["dispatched"] is False)
    c.equal("and recorded as an intervention", len(run.interventions), 1)
    c.equal("named for what it was", run.interventions[0]["reason"], "invalid_tool_call")

    answer = _blocks(run, "user", "tool_result")[0]
    c.check("the model is told the argument was missing", "missing required" in answer["content"])
    c.check("and which one it did not recognise", "unknown argument" in answer["content"])
    c.check("and what the tool actually takes", "expected path (string" in answer["content"])
    c.check("as an error, so the model does not read it as output", answer["is_error"])
    c.equal("the run is otherwise unaffected", run.stop_reason, StopReason.FINISHED)

    toolset = StrictToolset()
    run, _, _ = run_loop(
        [tool_turn("reed_file", {"path": "a.py"}), finish_turn()], toolset=toolset
    )
    answer = _blocks(run, "user", "tool_result")[0]
    c.check("a misspelled tool name gets the nearest real one", "read_file" in answer["content"])
    c.check("and the full list of what exists", "edit_file" in answer["content"])
    c.equal("and nothing is run", toolset.calls, [])


def check_loop_detection(c: Checks) -> None:
    """Being told is the recovery; being told twice and doing it again is a result."""
    failing = ToolResult(tool="edit_file", success=False, error="old_string not found in a.py")
    script = [_edit_turn(index) for index in range(8)] + [finish_turn("done")]

    seen: List[Snapshot] = []
    run, toolset, _ = run_loop(
        list(script),
        toolset=StrictToolset({"edit_file": failing}),
        config=AgentConfig(max_turns=20),
        on_turn=seen.append,
    )
    c.equal("a run that cannot get itself unstuck ends as a loop", run.stop_reason, StopReason.AGENT_LOOP)
    c.equal("after two warnings and one more identical attempt", run.turns, 5)
    c.check("the record says why", "being stopped" in run.error)
    c.equal("with every warning kept", len(run.interventions), 3)
    c.check("the last of which escalated", run.interventions[-1]["escalate"])
    c.check("and none before it did", not any(entry["escalate"] for entry in run.interventions[:-1]))
    c.equal("the terminal snapshot carries them too", len(seen[-1].interventions), 3)
    c.equal(
        "and the grader is told how many there were",
        TaskRun(task_id="t", agent=run).agent_record()["interventions"],
        3,
    )
    c.check("turns were saved rather than spent", run.turns < 9)

    notes = _blocks(run, "user", "text")
    c.equal("the harness speaks in its own block", len(notes), 3)
    c.check("identified as the harness", all(note["text"].startswith("[harness]") for note in notes))
    c.check(
        "tool output is left exactly as the tool said it",
        all("[harness]" not in block["content"] for block in _blocks(run, "user", "tool_result")),
    )
    for message in run.messages:
        if message["role"] != "user" or not isinstance(message["content"], list):
            continue
        kinds = [block.get("type") for block in message["content"]]
        if "text" in kinds and not c.check(
            "every tool result comes before the note about it",
            kinds.index("text") == len(kinds) - 1,
            str(kinds),
        ):
            break

    print()
    off, toolset, _ = run_loop(
        list(script),
        toolset=StrictToolset({"edit_file": failing}),
        config=AgentConfig(max_turns=20, context=ContextPolicy(enabled=False)),
    )
    c.equal("the control arm is never interrupted", off.stop_reason, StopReason.FINISHED)
    c.equal("  ...and says nothing", off.interventions, [])
    c.equal("  ...so the same script runs to the end", off.turns, 9)
    c.check("  ...making the two comparable on one variable", run.turns != off.turns)

    # An agent that announces it is done in the same turn it repeats itself
    # gets to be done: the point is to stop a run going nowhere, not to
    # override one that has arrived.
    both = ModelResponse(
        content=[
            {
                "type": "tool_use",
                "id": "tu_edit_last",
                "name": "edit_file",
                "input": {"path": "a.py", "old_string": "x", "new_string": "y"},
            },
            {"type": "tool_use", "id": "tu_fin", "name": FINISH_TOOL_NAME, "input": {"summary": "done"}},
        ],
        stop_reason="tool_use",
        usage=dict(USAGE),
    )
    finished, _, _ = run_loop(
        [_edit_turn(index) for index in range(4)] + [both],
        toolset=StrictToolset({"edit_file": failing}),
        config=AgentConfig(max_turns=20),
    )
    c.equal("finishing beats escalating in the same turn", finished.stop_reason, StopReason.FINISHED)

    c.check(
        "looping is neither the harness breaking nor the agent deciding",
        not StopReason.AGENT_LOOP.is_infrastructure
        and not StopReason.AGENT_LOOP.is_agent_decision,
    )


def check_honest_repetition(c: Checks) -> None:
    """The edit-test-edit cycle must never be mistaken for a loop."""
    class ChangingToolset(StrictToolset):
        def __init__(self):
            super().__init__()
            self.runs = 0

        def call(self, tool_name, arguments=None):
            self.calls.append((tool_name, arguments or {}))
            self.runs += 1
            # The same command, a different result each time: this is what
            # making progress looks like.
            return ToolResult(
                tool=tool_name,
                success=True,
                output={"exit_code": 0, "stdout": f"{self.runs} passed"},
                duration_ms=1.0,
            )

    script = [tool_turn("read_file", {"path": "a.py"}, f"tu_{i}") for i in range(8)]
    run, _, _ = run_loop(
        script + [finish_turn()],
        toolset=ChangingToolset(),
        config=AgentConfig(max_turns=20),
    )
    c.equal("a call whose result keeps changing is not a loop", run.interventions, [])
    c.equal("so the run is left alone", run.stop_reason, StopReason.FINISHED)
    c.equal("all nine turns ran", run.turns, 9)


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
    check_refusal(c)
    check_request_shape(c)
    check_sandbox_gone(c)
    check_stop_reason_attribution(c)
    check_agent_record(c)
    check_run_infrastructure(c)
    check_snapshots(c)
    check_resume(c)
    check_retries_on_the_record(c)
    check_invalid_tool_calls(c)
    check_loop_detection(c)
    check_honest_repetition(c)


# ---- live: the same loop, a real container, a scripted agent ---------------


def run_end_to_end(c: Checks, task_ids, dataset_dir: Path, repo_cache: Path) -> None:
    tasks = [
        (load_task(task_dir), task_dir)
        for task_dir in discover_task_dirs(dataset_dir)
        if not task_ids or load_task(task_dir).task_id in task_ids
    ]
    if not task_ids:
        # One container per run; the whole dataset is the grader's job, not this one.
        tasks = tasks[:1]

    for task, task_dir in tasks:
        repo = ensure_repo(
            task.repository, repo_cache, [task.commit, task.reference_commit], quiet=True
        )
        client = ScriptedClient(replay_fix(task, repo))
        run = run_task(
            task,
            task_dir,
            client=client,
            repo=repo,
            config=RunConfig(repo_cache=repo_cache, grade=True, quiet=True),
        )

        name = task.task_id
        if not c.check(
            f"{name}: the environment came up and the agent ran",
            not run.infrastructure_error,
            run.infrastructure_error,
        ):
            continue

        c.equal(f"{name}: the run ended because finish was called", run.agent.stop_reason, StopReason.FINISHED)
        c.check(f"{name}: the tools really executed in the container", len(run.agent.tool_calls) == len(client.seen_messages) - 1)
        c.check(f"{name}: every tool call succeeded", all(call["success"] for call in run.agent.tool_calls))
        c.check(f"{name}: the tools ran sandboxed", all(call["metadata"].get("sandboxed") for call in run.agent.tool_calls))
        c.check(f"{name}: a diff was collected", run.patch is not None and not run.patch.is_empty)
        c.equal(f"{name}: the diff covers exactly what was written", run.patch.files, sorted(task.reference_paths))
        c.check(
            f"{name}: setup's installed packages stayed out of the diff",
            not any("egg-info" in path or "__pycache__" in path for path in run.patch.files),
        )
        c.equal(
            f"{name}: the known-good fix grades as solved",
            run.evaluation.outcome,
            Outcome.SOLVED,
        )
        c.equal(
            f"{name}: the grade records who attempted it",
            run.evaluation.agent["stop_reason"],
            "finished",
        )


# ---- entry point -----------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m Agent.selftest", description=__doc__.splitlines()[0]
    )
    parser.add_argument("task_ids", nargs="*", help="end-to-end check only these tasks")
    parser.add_argument("--offline", action="store_true", help="skip everything needing Docker")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--repo-cache", default=str(DEFAULT_REPO_CACHE))
    args = parser.parse_args(argv)

    for name in ("agent.tools", "agent.run", "sandbox", "sandbox.docker", "evaluation"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    c = Checks()
    print("=== agent loop checks (no Docker, no API key)")
    run_offline(c)

    if not args.offline:
        print("\n=== end-to-end: real container, scripted agent, graded patch")
        run_end_to_end(c, args.task_ids, Path(args.dataset), Path(args.repo_cache))

    total = len(c.results)
    failed = len(c.failures)
    print(f"\n{total - failed}/{total} checks passed")
    if failed:
        for name, _, detail in c.failures:
            print(f"  FAIL {name}" + (f"\n       {detail}" if detail else ""))
        return 1
    print("every exit path, budget and malformed turn is handled as intended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
