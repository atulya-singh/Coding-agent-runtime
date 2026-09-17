"""A stand-in for the model that replays a fixed script of tool calls.

Two jobs, both of which a real model is the wrong tool for.

It exercises the harness without spending anything. Everything between the loop
and the grade -- the container, the toolhost, the diff, the checkpoints -- is the
same code whether a model or a script is driving, so a scripted run that does not
end SOLVED localises the fault to the harness. `replay_fix` builds that script
from the task's own reference commit, which makes the patch correct by
construction: there is no version of "the agent just wasn't good enough" to hide
behind.

And it is the oracle baseline. A capability number means nothing without knowing
the ceiling is reachable under the same budgets, the same tools and the same
grading -- so an agent that already knows the answer is the control every
experiment here is measured against.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from Tasks.harness import file_at_commit
from Tasks.schema import Task

from .client import ModelError, ModelResponse
from .config import AgentConfig
from .tools import FINISH_TOOL_NAME

#: Nonzero so usage accounting is exercised rather than bypassed; it is not a
#: claim about what a real turn costs.
SCRIPTED_USAGE = {"input_tokens": 100, "output_tokens": 50}


class ScriptedClient:
    """Replays a fixed list of ModelResponses, or raises a queued exception.

    Same three-argument `generate` as ModelClient, and nothing else, which is
    exactly the surface AgentLoop uses.
    """

    def __init__(self, responses: List, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.responses = list(responses)
        #: Every message list it was called with, so a test can assert what the
        #: loop actually sent rather than what it meant to send.
        self.seen_messages: List[List[dict]] = []

    def generate(self, messages, tools, system) -> ModelResponse:
        self.seen_messages.append([dict(message) for message in messages])
        if not self.responses:
            raise ModelError("scripted client ran out of responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool_turn(name: str, payload: dict, block_id: str = "tu_1") -> ModelResponse:
    return ModelResponse(
        content=[{"type": "tool_use", "id": block_id, "name": name, "input": payload}],
        stop_reason="tool_use",
        usage=dict(SCRIPTED_USAGE),
    )


def finish_turn(summary: str = "done") -> ModelResponse:
    return tool_turn(FINISH_TOOL_NAME, {"summary": summary}, block_id="tu_finish")


def text_turn(text: str = "all set") -> ModelResponse:
    return ModelResponse(
        content=[{"type": "text", "text": text}], stop_reason="end_turn", usage=dict(SCRIPTED_USAGE)
    )


def replay_fix(task: Task, repo: Path) -> List[ModelResponse]:
    """The task's known upstream fix, as the tool calls an agent would have made.

    Read the file, write the version upstream shipped, run the public tests, say
    it is done. Deliberately dull: the point is that every step goes through the
    same sandboxed tools a real run uses.
    """
    responses = [tool_turn("read_file", {"path": task.reference_paths[0]}, "tu_read")]
    for index, rel_path in enumerate(task.reference_paths):
        content = file_at_commit(repo, task.reference_commit, rel_path).decode("utf-8")
        responses.append(
            tool_turn("write_file", {"path": rel_path, "content": content}, f"tu_write_{index}")
        )
    responses.append(tool_turn("run_tests", {"command": task.public_tests}, "tu_tests"))
    responses.append(finish_turn("applied the upstream fix"))
    return responses
