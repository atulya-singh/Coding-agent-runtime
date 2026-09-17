"""Look at what has been checkpointed.

    python -m State                              # every task with checkpoints
    python -m State requests-001-netrc-empty-default
    python -m State requests-001-netrc-empty-default --all       # not just the live chain
    python -m State requests-001-netrc-empty-default --patch     # the diff at the tip

Read-only. Deleting a task's checkpoints is `--forget`, and it asks first,
because a checkpoint is the only copy of work whose container is already gone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .store import DEFAULT_STATE_ROOT, CheckpointError, CheckpointStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m State", description=__doc__.splitlines()[0])
    parser.add_argument("task_ids", nargs="*", help="tasks to inspect")
    parser.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))
    parser.add_argument(
        "--all",
        action="store_true",
        help="list every stored checkpoint, including ones from earlier runs",
    )
    parser.add_argument("--patch", action="store_true", help="print the diff at the latest checkpoint")
    parser.add_argument("--forget", action="store_true", help="delete the named tasks' checkpoints")
    return parser


def _show(store: CheckpointStore, task_id: str, show_all: bool, show_patch: bool) -> None:
    checkpoints = store.list_checkpoints(task_id) if show_all else store.history(task_id)
    if not checkpoints:
        print(f"{task_id}: no checkpoints")
        return

    tip = checkpoints[-1]
    label = "every stored checkpoint" if show_all else "the chain ending at latest"
    print(f"\n=== {task_id}  ({len(checkpoints)} checkpoints, {label})")
    for checkpoint in checkpoints:
        print(f"  {checkpoint.summary_line()}")

    refusal = tip.resumable
    print(f"  -> {'resumable' if not refusal else 'not resumable: ' + refusal}")
    if tip.usage:
        print(f"  -> {sum(tip.usage.values())} tokens over {len(tip.tool_calls)} tool call(s)")

    if show_patch:
        print(f"\n--- diff at {tip.checkpoint_id} ---")
        print(tip.patch_text or "(nothing changed yet)")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    store = CheckpointStore(Path(args.state_root))

    task_ids = args.task_ids or store.task_ids()
    if not task_ids:
        print(f"no checkpoints under {args.state_root}")
        return 0

    if args.forget:
        if not args.task_ids:
            print("name the tasks to forget; --forget does not take everything at once", file=sys.stderr)
            return 2
        answer = input(f"delete checkpoints for {', '.join(task_ids)}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("left alone")
            return 0
        for task_id in task_ids:
            store.clear(task_id)
            print(f"forgot {task_id}")
        return 0

    for task_id in task_ids:
        try:
            _show(store, task_id, args.all, args.patch)
        except CheckpointError as exc:
            print(f"{task_id}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
