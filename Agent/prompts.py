"""What the model is told about the job and the environment.

Split the way the API splits: the system prompt is standing context about the
harness (where the code is, how to finish), the first user message is the task
itself. Nothing here mentions hidden tests or the reference solution -- those
are grading-only (Tasks/schema.py) and never reach the container the agent
works in, let alone its context.
"""
from __future__ import annotations

from Tasks.schema import Task

SYSTEM_PROMPT = """\
You are an autonomous software engineer working inside an isolated container on \
a checkout of a real repository.

Environment:
- The working directory is the repository root. Tool paths are relative to it.
- You have no network access unless the task says otherwise.
- Only your changes to files in the repository count. Describing a fix is not \
making one.

How to work:
- Read before you write. Locate the relevant code with search_code and read_file \
rather than guessing at paths or APIs.
- Prefer edit_file over write_file for changes to existing files, so you do not \
discard code you have not read.
- Verify your work by running tests before you finish.
- When the task is done, or you are certain you cannot make progress, call finish.
"""


def build_system_prompt(task: Task, extra: str = "") -> str:
    parts = [SYSTEM_PROMPT]
    if task.public_tests:
        parts.append(f"The test command for this repository is: {task.public_tests}")
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)


def build_initial_messages(task: Task) -> list:
    return [{"role": "user", "content": task.description}]
