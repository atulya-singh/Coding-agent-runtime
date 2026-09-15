"""One task, end to end: a real container, a real model, a real diff.

This is the seam the loop was built against. `AgentLoop` knows only about a
toolset and a client; everything a real attempt also needs -- a pristine
checkout, a container, the task's setup steps, the diff at the end, optionally
the Phase 4 grade -- is assembled here, so the loop stays testable without any
of it.

The order matters, and each step is a different kind of failure:

    export the base commit      host-side git; ours if it breaks
    start the container         infrastructure
    run task.setup              infrastructure -- a broken environment is not a
                                failed agent, and must not be counted as one
    install the toolhost        infrastructure; done before the first token is
                                spent, so a protocol mismatch costs nothing
    run the loop                the agent's attempt
    collect the diff            the unit of record; graded in a *different*
                                container, never this one

The model client is built here, on the host, and is never passed to the Sandbox
or the toolset. That is what keeps the Phase 1 rule -- no credentials in the
container -- a property of the wiring rather than a habit.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from Evaluation.evaluator import EvaluationConfig, Evaluator
from Evaluation.patch import Patch, collect_patch
from Evaluation.result import EvaluationResult
from Execution.sandbox_tools import SandboxToolset
from Sandbox import Sandbox
from Tasks.harness import DEFAULT_REPO_CACHE, ensure_repo, export_commit, tail
from Tasks.schema import Task

from .client import ModelClient
from .config import AgentConfig
from .loop import AgentLoop, AgentRun

logger = logging.getLogger("agent.run")

#: Same ceiling the grader gives setup: `pip install -e .` on a cold cache is
#: minutes, and a shorter limit would fail honest tasks on a slow network.
DEFAULT_SETUP_TIMEOUT = 900.0

PROVIDER = "anthropic"


@dataclass
class RunConfig:
    repo_cache: Path = DEFAULT_REPO_CACHE
    setup_timeout: float = DEFAULT_SETUP_TIMEOUT
    #: Keep the agent's final tree and the sandbox run log under here. Off by
    #: default: it is a full copy of the repository per run.
    artifacts_dir: Optional[Path] = None
    #: Grade the resulting patch through the Phase 4 pipeline (a second container).
    grade: bool = False
    quiet: bool = False


@dataclass
class TaskRun:
    """What one attempt produced, with the harness and the agent kept apart."""

    task_id: str
    model: str = ""
    agent: Optional[AgentRun] = None
    patch: Optional[Patch] = None
    evaluation: Optional[EvaluationResult] = None
    #: Set when the harness broke rather than the agent failing. Principle 5:
    #: these belong outside the capability rates, not inside them as zeroes.
    infrastructure_error: str = ""
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """Did the agent get a fair attempt? Not "did it succeed"."""
        return not self.infrastructure_error and self.agent is not None

    def agent_record(self) -> dict:
        """The `agent` block on an EvaluationResult.

        `Evaluation.result` has carried this field as a plumbed-but-empty dict
        since Phase 4, waiting for exactly this: who attempted the task, under
        what budget, and what it cost.
        """
        run = self.agent
        if run is None:
            return {"model": self.model, "provider": PROVIDER}
        return {
            "model": run.model,
            "provider": PROVIDER,
            "stop_reason": run.stop_reason.value,
            "agent_decided_to_stop": run.stop_reason.is_agent_decision,
            "turns": run.turns,
            "tool_calls": len(run.tool_calls),
            "usage": run.usage,
            "total_tokens": run.total_tokens,
            "cost_usd": run.cost_usd,
            "summary": run.summary,
        }

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "duration_ms": round(self.duration_ms, 2),
            "infrastructure_error": self.infrastructure_error,
            "agent": self.agent.to_dict() if self.agent else None,
            "patch": self.patch.to_dict() if self.patch else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
        }


def run_task(
    task: Task,
    task_dir: Path,
    client: Any = None,
    agent_config: Optional[AgentConfig] = None,
    repo: Optional[Path] = None,
    config: Optional[RunConfig] = None,
) -> TaskRun:
    """Attempt one task and return what happened. Never raises for a failed
    attempt -- a failure is a result, and the record says whose it was."""
    config = config or RunConfig()
    agent_config = agent_config or getattr(client, "config", None) or AgentConfig()
    run = TaskRun(task_id=task.task_id, model=agent_config.model)
    started = time.monotonic()

    workspace = Path(tempfile.mkdtemp(prefix=f"agent-{task.task_id}-"))
    base = workspace / "base"
    repo_path: Optional[Path] = Path(repo) if repo else None

    try:
        if repo_path is None:
            repo_path = ensure_repo(
                task.repository, config.repo_cache, [task.commit], quiet=config.quiet
            )
        export_commit(repo_path, task.commit, base)

        # Host-side, before the container exists, and deliberately not reachable
        # from it: this object holds the API key.
        client = client or ModelClient(agent_config)

        sandbox_config = task.resource_limit.to_sandbox_config()
        with Sandbox(task_id=f"agent-{task.task_id}", config=sandbox_config) as sandbox:
            sandbox.initialize(str(base))

            setup_error = _run_setup(sandbox, task, config)
            if setup_error:
                run.infrastructure_error = setup_error
            else:
                toolset = SandboxToolset(sandbox)
                toolset.install()
                run.agent = AgentLoop(task, toolset, client, agent_config).run()
                run.patch = _collect(sandbox, base, task, config)
                if run.patch is None:
                    run.infrastructure_error = (
                        "the container was torn down before the diff could be collected "
                        "(a command timed out); the agent's work is not recoverable"
                    )
    except Exception as exc:  # noqa: BLE001 -- anything escaping here is ours
        run.infrastructure_error = f"{type(exc).__name__}: {exc}"
        logger.exception("running %s failed before the agent could be graded", task.task_id)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if config.grade and run.patch is not None:
        run.evaluation = _grade(task, task_dir, run, repo_path, config)

    run.duration_ms = (time.monotonic() - started) * 1000
    return run


def _run_setup(sandbox: Sandbox, task: Task, config: RunConfig) -> str:
    """Bring the environment up. A failure here is infrastructure: the agent has
    not been asked anything yet, so nothing about it has been measured."""
    for step in task.setup:
        result = sandbox.run_command(step, timeout=config.setup_timeout)
        if not result.success:
            return f"setup step `{step}` exited {result.exit_code}: {tail(result)}"
    return ""


def _collect(sandbox: Sandbox, base: Path, task: Task, config: RunConfig) -> Optional[Patch]:
    """Diff the agent's container against the pristine base export.

    Returns None when the container is already gone -- `Sandbox` destroys itself
    on a command timeout, and there is then nothing left to copy out.
    """
    if sandbox.container_id is None:
        return None
    artifacts = (
        Path(config.artifacts_dir) / "agent" / task.task_id if config.artifacts_dir else None
    )
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
    return collect_patch(sandbox, base, artifacts)


def _grade(
    task: Task,
    task_dir: Path,
    run: TaskRun,
    repo: Optional[Path],
    config: RunConfig,
) -> EvaluationResult:
    """Grade the patch in a fresh container, never the agent's own.

    Artifacts go under `graded/` so they cannot overwrite the agent's tree: the
    two are different answers to different questions -- what the agent left
    behind, and what that diff does on a clean checkout.
    """
    evaluation_config = EvaluationConfig(
        repo_cache=config.repo_cache,
        artifacts_dir=Path(config.artifacts_dir) / "graded" if config.artifacts_dir else None,
        quiet=config.quiet,
    )
    evaluator = Evaluator(task, task_dir, repo=repo, config=evaluation_config)
    return evaluator.evaluate(run.patch, agent=run.agent_record())
