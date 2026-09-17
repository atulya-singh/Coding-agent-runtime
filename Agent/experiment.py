"""Kill a run partway through and see whether it can still finish the task.

    python -m Agent.experiment requests-001-netrc-empty-default --replay
    python -m Agent.experiment requests-001-netrc-empty-default --fractions 0.25,0.75

The question plan.md Phase 5 asks is whether checkpoint/resume actually preserves
an agent's progress, and the only convincing way to answer it is to destroy a run
rather than ask it to stop. So each trial launches `python -m Agent` as a child
process, watches the checkpoints appear, and SIGKILLs the whole process group at
the chosen fraction of max_turns -- no cleanup, no final write, nothing the loop
could have done to prepare. Whatever is on disk at that instant is all there is.

The recovered run is then graded by exactly the Phase 4 pipeline that grades an
uninterrupted one, and the result carries `killed_at_fraction` and `resumed_from`
so a reader can tell a recovered grade from an ordinary one.

Two things make this a fair test rather than a demonstration:

  * The container from the killed run is force-removed before resuming, so the
    recovered run cannot accidentally inherit a filesystem that should be gone.
    Correctness never depends on this working -- resume always builds a new
    container -- but if it silently did, the experiment would prove nothing.
  * max_turns is cumulative across the kill, so a resumed run gets the remainder
    of the original budget and not a fresh one.

`--replay` runs the whole sweep against the oracle agent for no cost, which is
how the machinery is checked before a real model is pointed at it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from Evaluation.metrics import format_report
from Evaluation.result import EvaluationResult
from Sandbox import docker_backend as docker
from State.store import CheckpointStore
from Tasks.harness import DEFAULT_REPO_CACHE, ensure_repo
from Tasks.loader import discover_task_dirs, load_task

from .config import AgentConfig, load_config
from .replay import ScriptedClient, replay_fix
from .run import RunConfig, run_task

DATASET_DIR = Path(__file__).parent.parent / "Tasks" / "dataset"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

#: plan.md Phase 5, verbatim: kill at 10/25/50/75/90% of expected execution.
DEFAULT_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 0.90)

#: How long to wait for a child to reach its kill point before giving up on the
#: trial. Generous: a cold `pip install -e .` before the first turn is minutes.
DEFAULT_WAIT_SECONDS = 1800.0

POLL_SECONDS = 1.0


@dataclass
class Trial:
    fraction: float
    target_step: int
    killed_at_step: int = -1
    #: Set when the run finished before it could be killed, which is a fact about
    #: the task being short rather than a failure of the experiment.
    skipped: str = ""
    error: str = ""
    resumed_from: str = ""
    result: Optional[EvaluationResult] = None
    checkpoints_before_kill: int = 0
    turns_after_resume: int = 0

    def to_dict(self) -> dict:
        return {
            "fraction": self.fraction,
            "target_step": self.target_step,
            "killed_at_step": self.killed_at_step,
            "skipped": self.skipped,
            "error": self.error,
            "resumed_from": self.resumed_from,
            "checkpoints_before_kill": self.checkpoints_before_kill,
            "turns_after_resume": self.turns_after_resume,
            "result": self.result.to_dict() if self.result else None,
        }

    def line(self) -> str:
        if self.skipped:
            return f"  {self.fraction:>5.0%}  skipped -- {self.skipped}"
        if self.error:
            return f"  {self.fraction:>5.0%}  ERROR -- {self.error[:90]}"
        outcome = self.result.outcome.value.upper() if self.result else "not graded"
        return (
            f"  {self.fraction:>5.0%}  killed at turn {self.killed_at_step:<3}"
            f"  +{self.turns_after_resume} turns after resume  -> {outcome}"
        )


@dataclass
class Sweep:
    task_id: str
    max_turns: int
    trials: List[Trial] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "max_turns": self.max_turns,
            "trials": [trial.to_dict() for trial in self.trials],
        }


def _spawn(task_id: str, state_root: Path, args) -> subprocess.Popen:
    """Start a real run as a child process, in its own process group.

    Its own group so the kill takes the `docker` client processes with it. The
    container itself survives -- SIGKILL does not reach into the daemon -- which
    is exactly the orphan a crashed run leaves behind, and exactly what the
    cleanup step then has to deal with.
    """
    argv = [
        sys.executable, "-m", "Agent", task_id,
        "--state-root", str(state_root),
        "--config", args.config,
        "--dataset", args.dataset,
        "--repo-cache", args.repo_cache,
        "--quiet",
    ]
    if args.replay:
        argv.append("--replay")
    if args.max_turns is not None:
        argv += ["--max-turns", str(args.max_turns)]
    return subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )


def _kill_group(child: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        child.kill()
    child.wait(timeout=30)


def _remove_orphan(task_id: str) -> None:
    """Best effort. Sandbox names its container after the task, and a killed run
    leaves that container running. `remove_container` takes no ignore-missing
    flag, so the caller wraps it -- there is nothing to do if it is already gone."""
    try:
        docker.remove_container(f"sandbox-agent-{task_id}", force=True)
    except Exception:  # noqa: BLE001 -- cost hygiene, never correctness
        pass


def _run_to_kill_point(task_id: str, target_step: int, store: CheckpointStore, args) -> Trial:
    """Launch a run and SIGKILL it once it checkpoints at `target_step`."""
    trial = Trial(fraction=0.0, target_step=target_step)
    child = _spawn(task_id, store.root, args)
    deadline = time.monotonic() + args.wait

    try:
        while True:
            if time.monotonic() > deadline:
                _kill_group(child)
                trial.error = f"gave up waiting for turn {target_step} after {args.wait:.0f}s"
                return trial

            latest = store.latest(task_id)
            if latest is not None and latest.step >= target_step and not latest.is_terminal:
                _kill_group(child)
                trial.killed_at_step = latest.step
                trial.checkpoints_before_kill = len(store.history(task_id))
                return trial

            if child.poll() is not None:
                stderr = (child.stderr.read() or "").strip() if child.stderr else ""
                reached = latest.step if latest else 0
                if latest is not None and latest.is_terminal:
                    # Not a failure: the task simply took fewer turns than the
                    # kill point asked for, so there was no moment to kill it at.
                    trial.skipped = (
                        f"the run finished at turn {reached} without ever sitting "
                        f"un-finished at turn {target_step}"
                    )
                else:
                    trial.error = f"the child exited (code {child.returncode}) -- {stderr[-300:]}"
                return trial

            time.sleep(POLL_SECONDS)
    finally:
        if child.poll() is None:
            _kill_group(child)


def run_trial(fraction: float, task, task_dir: Path, repo: Path, agent_config: AgentConfig, args) -> Trial:
    target_step = max(1, round(fraction * agent_config.max_turns))
    # A directory per trial: two sweeps of the same task must not read each
    # other's `latest`, and keeping them apart makes each one re-readable after.
    store = CheckpointStore(Path(args.state_root) / f"kill-{int(fraction * 100):02d}")

    trial = _run_to_kill_point(task.task_id, target_step, store, args)
    trial.fraction = fraction
    if trial.skipped or trial.error:
        _remove_orphan(task.task_id)
        return trial

    _remove_orphan(task.task_id)

    checkpoint = store.latest(task.task_id)
    refusal = checkpoint.resumable if checkpoint else "nothing was checkpointed"
    if refusal:
        trial.error = refusal
        return trial
    trial.resumed_from = checkpoint.checkpoint_id

    # The oracle agent's script is positional -- its Nth entry is its Nth turn --
    # so a run resuming at turn N carries on from entry N. A real model needs no
    # such bookkeeping: it is handed the conversation and works out where it got to.
    client = None
    if args.replay:
        client = ScriptedClient(replay_fix(task, repo)[checkpoint.step :], agent_config)

    run = run_task(
        task,
        task_dir,
        client=client,
        agent_config=agent_config,
        repo=repo,
        config=RunConfig(
            repo_cache=Path(args.repo_cache),
            grade=True,
            quiet=True,
            sandbox_suffix=f"-resume-{int(fraction * 100):02d}",
        ),
        store=store,
        resume_from=checkpoint,
    )

    if run.infrastructure_error:
        trial.error = run.infrastructure_error
        return trial

    trial.turns_after_resume = run.agent.turns - checkpoint.step
    trial.result = run.evaluation
    if trial.result is not None:
        # The one thing a reader of the report cannot reconstruct: this grade came
        # from a run that was killed, and from where it picked up.
        trial.result.agent["killed_at_fraction"] = fraction
        trial.result.agent["killed_at_step"] = trial.killed_at_step
    return trial


def _replay_caveat(args) -> None:
    if args.replay:
        print(
            "\nnote: --replay drives the sweep with the oracle agent, so this measures "
            "whether checkpoint/resume preserves progress, not whether a model can "
            "recover its own train of thought."
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m Agent.experiment", description=__doc__.splitlines()[0]
    )
    parser.add_argument("task_id", help="the task to kill and resume")
    parser.add_argument(
        "--fractions",
        default=",".join(str(f) for f in DEFAULT_FRACTIONS),
        help="comma-separated fractions of max_turns to kill at",
    )
    parser.add_argument("--replay", action="store_true", help="use the oracle agent -- spends nothing")
    parser.add_argument(
        "--max-turns",
        type=int,
        help="override the configured turn limit -- also the denominator the fractions are of",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--repo-cache", default=str(DEFAULT_REPO_CACHE))
    parser.add_argument("--state-root", required=True, help="where each trial's checkpoints go")
    parser.add_argument("--wait", type=float, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--json", metavar="FILE", help="write the full sweep record here")
    args = parser.parse_args(argv)

    # The sweep's own report is the output; the containers underneath it produce
    # thousands of JSON log lines that would bury it.
    for logger_name in ("agent.tools", "agent.run", "sandbox", "sandbox.docker", "evaluation", "state"):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    config_path = Path(args.config)
    agent_config = load_config(config_path) if config_path.exists() else AgentConfig()
    if args.max_turns is not None:
        agent_config.max_turns = args.max_turns

    matches = [
        (load_task(task_dir), task_dir)
        for task_dir in discover_task_dirs(Path(args.dataset))
        if load_task(task_dir).task_id == args.task_id
    ]
    if not matches:
        print(f"no such task_id: {args.task_id}", file=sys.stderr)
        return 2
    task, task_dir = matches[0]

    commits = [task.commit] + ([task.reference_commit] if args.replay else [])
    repo = ensure_repo(task.repository, Path(args.repo_cache), commits, quiet=True)

    try:
        fractions = [float(part) for part in args.fractions.split(",") if part.strip()]
    except ValueError:
        print(f"--fractions must be numbers, got {args.fractions!r}", file=sys.stderr)
        return 2

    sweep = Sweep(task_id=task.task_id, max_turns=agent_config.max_turns)
    print(f"killing {task.task_id} at {', '.join(f'{f:.0%}' for f in fractions)} of {agent_config.max_turns} turns")

    for fraction in fractions:
        print(f"\n--- {fraction:.0%} (turn {max(1, round(fraction * agent_config.max_turns))})")
        trial = run_trial(fraction, task, task_dir, repo, agent_config, args)
        sweep.trials.append(trial)
        print(trial.line())

    print("\n=== kill/resume sweep")
    for trial in sweep.trials:
        print(trial.line())

    graded = [trial.result for trial in sweep.trials if trial.result is not None]
    if graded:
        print(format_report(graded))
    _replay_caveat(args)

    if args.json:
        Path(args.json).write_text(json.dumps(sweep.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    # A trial that could not be run at all is the harness's failure; a resumed
    # run that then failed the task is a result.
    return 1 if any(trial.error for trial in sweep.trials) else 0


if __name__ == "__main__":
    sys.exit(main())
