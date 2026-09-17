"""Prove a killed run can actually be finished, not just that it was written down.

The offline half pins the parts where a bug is silent: the record roundtrips, a
crash mid-save cannot make `latest` point at a half-written checkpoint, a record
from a different version is refused rather than resumed under the wrong rules,
and the parent chain survives a resume.

The live half is the claim itself, and it is deliberately harsh. A task is run
partway in a real container, and then *everything in memory is thrown away* --
the sandbox, the loop, the client, the store object. What is left is a directory
on disk. A new store reads it, a brand-new container is built from the base
commit with the recorded diff replayed onto it, the run continues, and the result
goes through exactly the Phase 4 grader an uninterrupted run would. If that
grades SOLVED, the work survived a boundary nothing in memory crossed.

    python -m State.selftest              # offline checks, then one live task
    python -m State.selftest --offline    # no Docker, no network
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from Agent.config import AgentConfig
from Agent.loop import IN_PROGRESS as LOOP_IN_PROGRESS
from Agent.loop import Snapshot, StopReason
from Agent.replay import ScriptedClient, replay_fix
from Agent.run import RunConfig, run_task
from Evaluation.result import Outcome
from Evaluation.selftest import Checks
from Tasks.harness import DEFAULT_REPO_CACHE, ensure_repo
from Tasks.loader import discover_task_dirs, load_task

from .checkpoint import AGENT_DECIDED, CHECKPOINT_VERSION, IN_PROGRESS, Checkpoint
from .recorder import CheckpointRecorder
from .store import LATEST_FILENAME, CheckpointError, CheckpointStore

DATASET_DIR = Path(__file__).parent.parent / "Tasks" / "dataset"


# ---- doubles ---------------------------------------------------------------


class FakeSandbox:
    """Enough of a Sandbox for collect_patch: a tree that can be copied out.

    Real host-side git runs against it, so the diff in these checks is a real
    diff -- only the container is fake.
    """

    def __init__(self, tree: Path):
        self.tree = Path(tree)
        self.container_id = "fake-container"

    def collect_results(self, dest_dir: str) -> Path:
        out = Path(dest_dir)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.tree, out / "repository", dirs_exist_ok=True)
        return out


def sample_checkpoint(**overrides) -> Checkpoint:
    defaults = dict(
        task_id="selftest-001",
        step=3,
        model="claude-opus-5",
        provider="anthropic",
        messages=[{"role": "user", "content": "Fix the thing."}],
        tool_calls=[{"turn": 1, "tool": "read_file", "success": True}],
        usage={"input_tokens": 120, "output_tokens": 40},
        cost_usd=0.0012,
        retries=[
            {
                "operation": "messages.create",
                "retry": 1,
                "failure": "rate_limit",
                "error": "429",
                "delay_s": 1.0,
                "from_retry_after": False,
                "at": "2026-09-17T00:00:00+00:00",
            }
        ],
        interventions=[
            {
                "failure": "agent_loop",
                "reason": "repeated_call",
                "message": "[harness] you have called run_tests 3 times",
                "tool": "run_tests",
                "occurrences": 3,
                "nudge": 1,
                "escalate": False,
                "at": "2026-09-17T00:00:00+00:00",
            }
        ],
        patch_text="diff --git a/x.py b/x.py\n",
        patch_sha256="deadbeef",
        patch_meta={"files": ["x.py"]},
    )
    defaults.update(overrides)
    return Checkpoint(**defaults)


def snapshot(step: int, status: str = LOOP_IN_PROGRESS, **overrides) -> Snapshot:
    defaults = dict(
        task_id="selftest-001",
        model="claude-opus-5",
        step=step,
        status=status,
        messages=[{"role": "user", "content": "Fix the thing."}],
        tool_calls=[],
        usage={"input_tokens": 10},
    )
    defaults.update(overrides)
    return Snapshot(**defaults)


# ---- offline ---------------------------------------------------------------


def check_roundtrip(c: Checks) -> None:
    original = sample_checkpoint()
    restored = Checkpoint.from_dict(original.to_dict(), patch_text=original.patch_text)
    c.equal("the record survives a roundtrip", restored.to_dict(), original.to_dict())
    c.equal("so does the diff", restored.patch_text, original.patch_text)
    c.check(
        "the diff is not inlined into the record",
        "patch_text" not in original.to_dict(),
    )
    c.check(
        "the record is JSON, with nothing clever in it",
        isinstance(json.dumps(original.to_dict()), str),
    )
    c.equal("a step orders the filename", sample_checkpoint(step=7).filename_stem()[:6], "000007")


def check_resumability(c: Checks) -> None:
    c.equal("a mid-run checkpoint can be resumed", sample_checkpoint().resumable, "")
    c.check("and is not terminal", not sample_checkpoint().is_terminal)
    c.check(
        "a run the agent ended cannot be resumed",
        bool(sample_checkpoint(status="finished").resumable),
    )
    c.check(
        "being stopped by a limit can be",
        not sample_checkpoint(status="max_turns").resumable,
    )
    c.check(
        "so can a run whose API call failed",
        not sample_checkpoint(status="model_error").resumable,
    )
    # This is the dangerous one: the conversation is intact but the work is not,
    # so resuming would look fine and silently discard everything written.
    stranded = sample_checkpoint(patch_error="the container was gone", patch_text="")
    c.check("a checkpoint with no captured work is refused", bool(stranded.resumable))
    c.check("and says why", "never captured" in stranded.resumable)


def check_no_drift_from_the_loop(c: Checks) -> None:
    """AGENT_DECIDED is a copy of a fact that lives in Agent.loop. Copies rot."""
    c.equal(
        "the statuses treated as 'the agent finished' are exactly the loop's",
        sorted(AGENT_DECIDED),
        sorted(r.value for r in StopReason if r.is_agent_decision),
    )
    c.equal("the two in-progress markers are the same string", IN_PROGRESS, LOOP_IN_PROGRESS)
    c.check(
        "no stop reason collides with the in-progress marker",
        IN_PROGRESS not in {reason.value for reason in StopReason},
    )


def check_resume_state(c: Checks) -> None:
    state = sample_checkpoint().resume_state()
    c.equal(
        "the loop is handed exactly the six keys it reads",
        sorted(state),
        ["interventions", "messages", "retries", "tool_calls", "turns", "usage"],
    )
    c.equal("the step becomes the turn count", state["turns"], 3)
    # Otherwise a recovered run restarts these counts at zero and the record
    # understates both how badly the environment behaved and how much the
    # agent had to be corrected.
    c.equal("retries absorbed before the crash are handed back", len(state["retries"]), 1)
    c.equal("so are the corrections the agent was given", len(state["interventions"]), 1)


def check_store(c: Checks) -> None:
    root = Path(tempfile.mkdtemp(prefix="state-selftest-"))
    try:
        store = CheckpointStore(root)
        c.equal("an unknown task has no latest", store.latest("nope"), None)
        c.equal("and no history", store.history("nope"), [])

        first = sample_checkpoint(step=0)
        second = sample_checkpoint(step=1, parent_checkpoint_id=first.checkpoint_id)
        store.save(first)
        path = store.save(second)

        c.check("the record lands where the pointer says", path.exists())
        c.check("the diff is beside it, not inside it", path.with_suffix(".patch").exists())
        c.equal("latest is the newest save", store.latest("selftest-001").checkpoint_id, second.checkpoint_id)
        c.equal("the diff comes back with it", store.latest("selftest-001").patch_text, second.patch_text)
        c.equal("a checkpoint can be fetched by id", store.load("selftest-001", first.checkpoint_id).step, 0)
        c.equal("both are listed, in run order", [cp.step for cp in store.list_checkpoints("selftest-001")], [0, 1])
        c.equal("the task shows up in the index", store.task_ids(), ["selftest-001"])

        c.raises(
            "an unknown checkpoint id is an error, not a None to trip over later",
            CheckpointError,
            lambda: store.load("selftest-001", "nosuchid"),
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_crash_safety(c: Checks) -> None:
    """The whole reason for the temp-file-and-rename dance."""
    root = Path(tempfile.mkdtemp(prefix="state-selftest-"))
    try:
        store = CheckpointStore(root)
        good = sample_checkpoint(step=0)
        store.save(good)

        # A crash between the two writes leaves a .tmp behind. It is not a
        # checkpoint, and nothing may treat it as one.
        directory = store.checkpoints_dir("selftest-001")
        (directory / "000001-halfwritten.json.tmp").write_text("{ this is not", encoding="utf-8")

        c.equal("a stray temp file is not listed", [cp.step for cp in store.list_checkpoints("selftest-001")], [0])
        c.equal("and does not become latest", store.latest("selftest-001").checkpoint_id, good.checkpoint_id)

        # latest is written last, so it can never name a file that is not there.
        pointer = json.loads((store.task_dir("selftest-001") / LATEST_FILENAME).read_text())
        c.check(
            "latest names a record that exists",
            (directory / pointer["filename"]).exists(),
        )

        stale = store.task_dir("selftest-001") / LATEST_FILENAME
        stale.write_text(json.dumps({"filename": "000099-gone.json"}), encoding="utf-8")
        c.raises(
            "a pointer to a missing record is an error, not a silent restart",
            CheckpointError,
            lambda: store.latest("selftest-001"),
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_version_guard(c: Checks) -> None:
    root = Path(tempfile.mkdtemp(prefix="state-selftest-"))
    try:
        store = CheckpointStore(root)
        checkpoint = sample_checkpoint(step=0)
        path = store.save(checkpoint)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["checkpoint_version"] = CHECKPOINT_VERSION + 1
        path.write_text(json.dumps(record), encoding="utf-8")

        c.raises(
            "a checkpoint from another version is refused rather than misread",
            CheckpointError,
            lambda: store.latest("selftest-001"),
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_history(c: Checks) -> None:
    """Two runs of one task share a directory; only the parent links separate them."""
    root = Path(tempfile.mkdtemp(prefix="state-selftest-"))
    try:
        store = CheckpointStore(root)
        abandoned = sample_checkpoint(step=0)
        store.save(abandoned)

        first = sample_checkpoint(step=0)
        second = sample_checkpoint(step=1, parent_checkpoint_id=first.checkpoint_id)
        third = sample_checkpoint(step=2, parent_checkpoint_id=second.checkpoint_id)
        for checkpoint in (first, second, third):
            store.save(checkpoint)

        chain = store.history("selftest-001")
        c.equal(
            "history follows the parent links, oldest first",
            [cp.checkpoint_id for cp in chain],
            [first.checkpoint_id, second.checkpoint_id, third.checkpoint_id],
        )
        c.equal("the other run's checkpoint is still on disk", len(store.list_checkpoints("selftest-001")), 4)
        c.check("but is not in this run's history", abandoned.checkpoint_id not in {cp.checkpoint_id for cp in chain})
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_recorder(c: Checks) -> None:
    """The bridge: a loop snapshot plus a container becomes a saved checkpoint."""
    root = Path(tempfile.mkdtemp(prefix="state-selftest-"))
    base = root / "base"
    work = root / "work"
    try:
        base.mkdir(parents=True)
        (base / "app.py").write_text("value = 1\n", encoding="utf-8")
        shutil.copytree(base, work)

        store = CheckpointStore(root / "state")
        sandbox = FakeSandbox(work)
        recorder = CheckpointRecorder(
            store, sandbox, base, agent_config={"model": "m"}, provenance={"commit": "abc"}, provider="anthropic"
        )

        empty = recorder(snapshot(0))
        c.equal("nothing changed yet, so the diff is empty", empty.patch_text, "")
        c.equal("the first checkpoint has no parent", empty.parent_checkpoint_id, "")

        (work / "app.py").write_text("value = 2\n", encoding="utf-8")
        retried = [{"operation": "messages.create", "retry": 1, "failure": "model_timeout"}]
        changed = recorder(snapshot(1, retries=retried))
        c.check("the agent's edit shows up in the diff", "value = 2" in changed.patch_text)
        c.equal("retries carry from the snapshot onto the checkpoint", changed.retries, retried)
        c.equal(
            "and survive the trip through disk",
            store.latest("selftest-001").retries,
            retried,
        )
        c.equal("it records which files changed", changed.patch_meta["files"], ["app.py"])
        c.equal("each checkpoint points at the last", changed.parent_checkpoint_id, empty.checkpoint_id)
        c.equal("the config it ran under is kept", changed.agent_config, {"model": "m"})
        c.equal("so is the provenance", changed.provenance, {"commit": "abc"})
        c.equal("both were saved", len(store.list_checkpoints("selftest-001")), 2)
        c.equal("latest is the newer one", store.latest("selftest-001").checkpoint_id, changed.checkpoint_id)

        # The container tearing itself down is an ordinary ending, and the
        # terminal checkpoint for it still has a conversation worth keeping.
        sandbox.container_id = None
        stranded = recorder(snapshot(2, status=StopReason.SANDBOX_GONE.value))
        c.check("a dead container does not crash the recorder", stranded.step == 2)
        c.check("it records that the work was not captured", bool(stranded.patch_error))
        c.check("and that checkpoint refuses to be resumed", bool(stranded.resumable))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_offline(c: Checks) -> None:
    check_roundtrip(c)
    check_resumability(c)
    check_no_drift_from_the_loop(c)
    check_resume_state(c)
    check_store(c)
    check_crash_safety(c)
    check_version_guard(c)
    check_history(c)
    check_recorder(c)


# ---- live: kill it, throw the process away, finish it from disk ------------


def run_kill_and_resume(c: Checks, task_ids, dataset_dir: Path, repo_cache: Path) -> None:
    tasks = [
        (load_task(task_dir), task_dir)
        for task_dir in discover_task_dirs(dataset_dir)
        if not task_ids or load_task(task_dir).task_id in task_ids
    ]
    if not task_ids:
        tasks = tasks[:1]

    for task, task_dir in tasks:
        name = task.task_id
        repo = ensure_repo(
            task.repository, repo_cache, [task.commit, task.reference_commit], quiet=True
        )
        state_root = Path(tempfile.mkdtemp(prefix=f"state-live-{name}-"))
        try:
            script = replay_fix(task, repo)
            # Everything up to and including the writes; the tests and the finish
            # call are what the resumed run will have left to do.
            first_half = script[: 1 + len(task.reference_paths)]

            # --- the run that gets killed -------------------------------------
            cut_short = run_task(
                task,
                task_dir,
                client=ScriptedClient(first_half),
                agent_config=AgentConfig(max_turns=len(first_half)),
                repo=repo,
                config=RunConfig(repo_cache=repo_cache, quiet=True),
                store=CheckpointStore(state_root),
            )
            if not c.check(
                f"{name}: the first run got far enough to be worth resuming",
                not cut_short.infrastructure_error,
                cut_short.infrastructure_error,
            ):
                continue
            c.equal(
                f"{name}: it stopped partway, without finishing",
                cut_short.agent.stop_reason,
                StopReason.MAX_TURNS,
            )

            # --- everything in memory is now thrown away ----------------------
            # A fresh store over the same directory, and nothing else carried
            # over: no sandbox, no loop, no client, no patch object.
            del cut_short
            reloaded = CheckpointStore(state_root)
            checkpoint = reloaded.latest(name)

            if not c.check(f"{name}: a checkpoint is on disk to come back to", checkpoint is not None):
                continue
            c.equal(f"{name}: it is at the turn the run reached", checkpoint.step, len(first_half))
            c.equal(f"{name}: it is resumable", checkpoint.resumable, "")
            c.check(
                f"{name}: the work done before the kill is in the checkpoint",
                bool(checkpoint.patch_text) and all(
                    path in checkpoint.patch_text for path in task.reference_paths
                ),
            )
            c.check(
                f"{name}: so is the conversation",
                len(checkpoint.messages) >= 2 * len(first_half),
            )

            # --- finish it in a container that never saw the first run --------
            resumed = run_task(
                task,
                task_dir,
                client=ScriptedClient(script[1 + len(task.reference_paths) :]),
                agent_config=AgentConfig(max_turns=len(script) + 2),
                repo=repo,
                config=RunConfig(
                    repo_cache=repo_cache, grade=True, quiet=True, sandbox_suffix="-resumed"
                ),
                store=reloaded,
                resume_from=checkpoint,
            )
            if not c.check(
                f"{name}: the resumed run came up",
                not resumed.infrastructure_error,
                resumed.infrastructure_error,
            ):
                continue

            c.equal(f"{name}: it ran to a proper finish", resumed.agent.stop_reason, StopReason.FINISHED)
            c.equal(
                f"{name}: turns carried on rather than restarting",
                resumed.agent.turns,
                len(script),
            )
            # The real claim: the files written before the kill are in the final
            # diff, even though the container that held them no longer exists.
            c.equal(
                f"{name}: the pre-kill work is in the final patch",
                resumed.patch.files,
                sorted(task.reference_paths),
            )
            c.equal(
                f"{name}: a recovered run grades exactly like an uninterrupted one",
                resumed.evaluation.outcome,
                Outcome.SOLVED,
            )
            c.equal(
                f"{name}: the grade says which checkpoint it came from",
                resumed.evaluation.agent["resumed_from"],
                checkpoint.checkpoint_id,
            )
            chain = reloaded.history(name)
            c.check(
                f"{name}: the recovered run's chain reaches back through the kill",
                checkpoint.checkpoint_id in {cp.checkpoint_id for cp in chain},
            )
            c.check(
                f"{name}: and ends on a terminal checkpoint",
                chain[-1].is_terminal and chain[-1].status == StopReason.FINISHED.value,
            )
        finally:
            shutil.rmtree(state_root, ignore_errors=True)


# ---- entry point -----------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m State.selftest", description=__doc__.splitlines()[0]
    )
    parser.add_argument("task_ids", nargs="*", help="kill-and-resume only these tasks")
    parser.add_argument("--offline", action="store_true", help="skip everything needing Docker")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--repo-cache", default=str(DEFAULT_REPO_CACHE))
    args = parser.parse_args(argv)

    for logger_name in ("agent.tools", "agent.run", "sandbox", "sandbox.docker", "evaluation", "state"):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    c = Checks()
    print("=== checkpoint checks (no Docker, no network)")
    run_offline(c)

    if not args.offline:
        print("\n=== kill a run, discard everything, finish it from disk, grade it")
        run_kill_and_resume(c, args.task_ids, Path(args.dataset), Path(args.repo_cache))

    total = len(c.results)
    failed = len(c.failures)
    print(f"\n{total - failed}/{total} checks passed")
    if failed:
        for name, _, detail in c.failures:
            print(f"  FAIL {name}" + (f"\n       {detail}" if detail else ""))
        return 1
    print("a killed run can be finished from what is on disk, and grades the same")
    return 0


if __name__ == "__main__":
    sys.exit(main())
