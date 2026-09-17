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

Resuming a killed run is the same function with two additions: the checkpoint's
accumulated diff is applied to the base export before the container starts, and
the loop is handed the conversation it had reached. Deliberately not a separate
`resume()` -- one code path means a recovered run cannot drift from a fresh one,
which is the only thing that makes their grades comparable.
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
from Evaluation.patch import Patch, apply_patch, collect_patch
from Evaluation.result import EvaluationResult
from Execution.sandbox_tools import SandboxToolset
from Sandbox import Sandbox
from State.checkpoint import Checkpoint
from State.recorder import CheckpointRecorder
from State.store import CheckpointStore
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
    #: Distinguishes this attempt's container from another at the same task --
    #: a resumed run must not collide with the orphan the killed one left behind.
    sandbox_suffix: str = ""


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
    #: The last checkpoint this attempt wrote, and the one it started from.
    #: Together they are the claim "this grade came from a recovered run", and
    #: they are on the record so that claim can be checked rather than trusted.
    checkpoint_id: str = ""
    resumed_from: str = ""

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
        lineage = {"checkpoint_id": self.checkpoint_id, "resumed_from": self.resumed_from}
        run = self.agent
        if run is None:
            return {"model": self.model, "provider": PROVIDER, **lineage}
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
            **lineage,
        }

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "duration_ms": round(self.duration_ms, 2),
            "infrastructure_error": self.infrastructure_error,
            "checkpoint_id": self.checkpoint_id,
            "resumed_from": self.resumed_from,
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
    store: Optional[CheckpointStore] = None,
    resume_from: Optional[Checkpoint] = None,
) -> TaskRun:
    """Attempt one task and return what happened. Never raises for a failed
    attempt -- a failure is a result, and the record says whose it was.

    Pass `store` to checkpoint every turn, and `resume_from` to continue a run
    that was interrupted. Resuming needs a store too: a recovered run that could
    not itself be recovered would be a strange thing to measure.
    """
    config = config or RunConfig()
    agent_config = agent_config or getattr(client, "config", None) or AgentConfig()
    run = TaskRun(
        task_id=task.task_id,
        model=agent_config.model,
        resumed_from=resume_from.checkpoint_id if resume_from else "",
    )
    started = time.monotonic()

    workspace = Path(tempfile.mkdtemp(prefix=f"agent-{task.task_id}-"))
    base = workspace / "base"
    repo_path: Optional[Path] = Path(repo) if repo else None

    try:
        if resume_from is not None:
            refusal = resume_from.resumable
            if refusal:
                raise ValueError(f"checkpoint {resume_from.checkpoint_id} cannot be resumed: {refusal}")

        if repo_path is None:
            repo_path = ensure_repo(
                task.repository, config.repo_cache, [task.commit], quiet=config.quiet
            )
        export_commit(repo_path, task.commit, base)

        # What every diff is taken against. On a resumed run the container starts
        # from work already done, but the patch that gets graded still has to be
        # the whole change since the base commit -- so the tree the container is
        # seeded from and the tree it is compared to stop being the same thing,
        # and a second pristine export is what keeps the comparison honest.
        diff_base = base
        if resume_from is not None and resume_from.patch_text:
            diff_base = workspace / "pristine"
            export_commit(repo_path, task.commit, diff_base)
            # Replayed onto the export before the container sees it, so a
            # recovered environment is built the way a graded one is: base commit
            # plus a diff, never a filesystem rescued from a dead container.
            apply_patch(Patch.from_text(resume_from.patch_text), base)

        # Host-side, before the container exists, and deliberately not reachable
        # from it: this object holds the API key.
        client = client or ModelClient(agent_config)

        sandbox_config = task.resource_limit.to_sandbox_config()
        sandbox_id = f"agent-{task.task_id}{config.sandbox_suffix}"
        with Sandbox(task_id=sandbox_id, config=sandbox_config) as sandbox:
            sandbox.initialize(str(base))

            setup_error = _run_setup(sandbox, task, config)
            if setup_error:
                run.infrastructure_error = setup_error
            else:
                toolset = SandboxToolset(sandbox)
                toolset.install()
                recorder = (
                    _recorder(store, sandbox, diff_base, task, agent_config, resume_from)
                    if store is not None
                    else None
                )
                run.agent = AgentLoop(
                    task,
                    toolset,
                    client,
                    agent_config,
                    on_turn=recorder,
                    resume_from=resume_from.resume_state() if resume_from else None,
                ).run()
                if recorder is not None:
                    run.checkpoint_id = recorder.last_checkpoint_id
                run.patch = _collect(sandbox, diff_base, task, config)
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


def _recorder(
    store: CheckpointStore,
    sandbox: Sandbox,
    diff_base: Path,
    task: Task,
    agent_config: AgentConfig,
    resume_from: Optional[Checkpoint],
) -> CheckpointRecorder:
    return CheckpointRecorder(
        store=store,
        sandbox=sandbox,
        base_dir=diff_base,
        agent_config=agent_config.to_dict(),
        provenance={
            "repository": task.repository,
            "commit": task.commit,
            "image": task.resource_limit.to_sandbox_config().image,
        },
        provider=PROVIDER,
        parent_checkpoint_id=resume_from.checkpoint_id if resume_from else "",
    )


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
