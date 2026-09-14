"""Grade patches against the task dataset.

    python -m Evaluation requests-001-netrc-empty-default --patch fix.diff
    python -m Evaluation --all --reference --json results.json
    python -m Evaluation --all --empty          # the floor: no task may be solved

`--reference` grades the real upstream fix and `--empty` grades a patch that
changes nothing; between them they bracket every possible agent result, and both
run against exactly the pipeline that will grade an agent.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from Tasks.harness import DEFAULT_REPO_CACHE, ensure_repo
from Tasks.loader import discover_task_dirs, load_task

from .evaluator import EvaluationConfig, Evaluator
from .metrics import aggregate, by_category, format_report
from .patch import Patch, load_patch
from .reference import reference_patch

DATASET_DIR = Path(__file__).parent.parent / "Tasks" / "dataset"


def _silence_sandbox_logs() -> None:
    for name in ("sandbox", "sandbox.docker", "agent.tools", "evaluation"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m Evaluation", description=__doc__.splitlines()[0]
    )
    parser.add_argument("task_ids", nargs="*", help="tasks to evaluate")
    parser.add_argument("--all", action="store_true", help="evaluate every task in the dataset")

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--patch", metavar="FILE", help="unified diff to grade")
    source.add_argument(
        "--reference",
        action="store_true",
        help="grade the real upstream fix (expected: solved)",
    )
    source.add_argument(
        "--empty",
        action="store_true",
        help="grade a patch that changes nothing (expected: not solved)",
    )

    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--repo-cache", default=str(DEFAULT_REPO_CACHE))
    parser.add_argument("--artifacts", metavar="DIR", help="keep each graded tree and run log here")
    parser.add_argument("--json", metavar="FILE", help="write the full result records here")
    parser.add_argument("--quiet", action="store_true", help="silence sandbox JSON log lines")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        _silence_sandbox_logs()

    if not args.task_ids and not args.all:
        print("give one or more task_ids, or --all", file=sys.stderr)
        return 2
    if not (args.patch or args.reference or args.empty):
        print("choose a patch source: --patch FILE, --reference or --empty", file=sys.stderr)
        return 2

    tasks = []
    for task_dir in discover_task_dirs(Path(args.dataset)):
        task = load_task(task_dir)
        if args.all or task.task_id in args.task_ids:
            tasks.append((task, task_dir))

    unknown = set(args.task_ids) - {task.task_id for task, _ in tasks}
    if unknown:
        print(f"no such task_id: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    if not tasks:
        print(f"no tasks found in {args.dataset}", file=sys.stderr)
        return 2
    if args.patch and len(tasks) > 1:
        print("--patch grades one task; name exactly one task_id", file=sys.stderr)
        return 2

    config = EvaluationConfig(
        repo_cache=Path(args.repo_cache),
        artifacts_dir=Path(args.artifacts) if args.artifacts else None,
        quiet=args.quiet,
    )

    repos: dict = {}
    results = []
    for task, task_dir in tasks:
        if task.repository not in repos:
            commits = [t.commit for t, _ in tasks if t.repository == task.repository]
            commits += [t.reference_commit for t, _ in tasks if t.repository == task.repository]
            repos[task.repository] = ensure_repo(
                task.repository, Path(args.repo_cache), commits, quiet=args.quiet
            )
        repo = repos[task.repository]

        if args.patch:
            patch = load_patch(Path(args.patch))
        elif args.reference:
            patch = reference_patch(task, repo)
        else:
            patch = Patch.empty()

        print(f"\n=== {task.task_id}  [{task.category.value}]  patch {patch.sha256[:12]}")
        result = Evaluator(task, task_dir, repo=repo, config=config).evaluate(patch)
        results.append(result)

        for stage in result.stages:
            print(f"  {stage.status.value.upper():<8} {stage.stage.value:<18} {stage.detail[:110]}")
        print(f"  -> {result.outcome.value.upper()}")

    print(format_report(results))

    if args.json:
        payload = {
            "results": [result.to_dict() for result in results],
            "metrics": aggregate(results).to_dict(),
            "by_category": {name: m.to_dict() for name, m in by_category(results).items()},
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
