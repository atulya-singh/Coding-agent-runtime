"""Run the agent against the task dataset.

    python -m Agent requests-001-netrc-empty-default --grade
    python -m Agent --all --grade --json runs.json
    python -m Agent requests-001-netrc-empty-default --dry-run   # spends nothing

Defaults come from Agent/config.yaml, so the model and the budgets are edited in
one hand-written file rather than passed on the command line every time; --model
and --max-turns override it for a one-off.

This is the only entry point that spends money, so it says what it is about to
do before it does it, and --dry-run stops right there.
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

from .client import API_KEY_NAME, ModelClient, ModelError
from .config import AgentConfig, load_config, validate_config
from .run import RunConfig, TaskRun, run_task

DATASET_DIR = Path(__file__).parent.parent / "Tasks" / "dataset"
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m Agent", description=__doc__.splitlines()[0])
    parser.add_argument("task_ids", nargs="*", help="tasks to attempt")
    parser.add_argument("--all", action="store_true", help="attempt every task in the dataset")

    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="agent config YAML")
    parser.add_argument("--model", help="override the configured model")
    parser.add_argument("--max-turns", type=int, help="override the configured turn limit")

    parser.add_argument("--grade", action="store_true", help="grade each patch (a second container)")
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--repo-cache", default=str(DEFAULT_REPO_CACHE))
    parser.add_argument("--artifacts", metavar="DIR", help="keep each tree and run log here")
    parser.add_argument("--patch-out", metavar="FILE", help="write the diff here (one task only)")
    parser.add_argument("--json", metavar="FILE", help="write the full run records here")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would run, then stop -- no container, no API call",
    )
    parser.add_argument("--quiet", action="store_true", help="silence the sandbox/tool log lines")
    return parser


def _silence_logs() -> None:
    for name in ("sandbox", "sandbox.docker", "agent.tools", "agent.run", "evaluation"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _load_agent_config(args) -> AgentConfig:
    path = Path(args.config)
    config = load_config(path) if path.exists() else AgentConfig()
    if args.model:
        config.model = args.model
    if args.max_turns is not None:
        config.max_turns = args.max_turns
    return config


def _money(value: Optional[float]) -> str:
    # Never print 0.00 for an unpriced model: unknown and free are not the same
    # claim, and one of them would quietly become a number in a report.
    return f"${value:.4f}" if value is not None else "unknown (no pricing configured)"


def _describe(run: TaskRun) -> str:
    if run.infrastructure_error:
        return f"  !! infrastructure: {run.infrastructure_error[:160]}"
    agent = run.agent
    lines = [
        f"  stop      {agent.stop_reason.value}"
        + (f" -- {agent.error[:100]}" if agent.error else ""),
        f"  turns     {agent.turns}/{run.model}"
        f"  tools {len(agent.tool_calls)}"
        f"  tokens {agent.total_tokens}"
        f"  cost {_money(agent.cost_usd)}",
    ]
    if agent.summary:
        lines.append(f"  said      {agent.summary.strip()[:160]}")
    if run.patch is not None:
        lines.append(
            f"  patch     {len(run.patch.files)} file(s), "
            f"+{run.patch.lines_added}/-{run.patch.lines_deleted}, {run.patch.sha256[:12]}"
        )
    if run.evaluation is not None:
        lines.append(f"  -> {run.evaluation.outcome.value.upper()}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        _silence_logs()

    if not args.task_ids and not args.all:
        print("give one or more task_ids, or --all", file=sys.stderr)
        return 2

    agent_config = _load_agent_config(args)
    errors = validate_config(agent_config)
    if errors:
        for error in errors:
            print(f"config: {error}", file=sys.stderr)
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
    if args.patch_out and len(tasks) > 1:
        print("--patch-out writes one diff; name exactly one task_id", file=sys.stderr)
        return 2

    print(
        f"model {agent_config.model}  max_turns {agent_config.max_turns}  "
        f"max_tokens {agent_config.max_tokens}  "
        f"budget {_money(agent_config.max_cost_usd) if agent_config.max_cost_usd else 'none'}"
    )
    for task, _ in tasks:
        print(f"  would run {task.task_id} [{task.category.value}] in {task.resource_limit.to_sandbox_config().image}")
    if args.dry_run:
        print("\ndry run: nothing was started")
        return 0

    # Before any container: a bad key or a mistyped model id should cost
    # seconds, not the minutes it takes to build an environment first.
    client = ModelClient(agent_config)
    try:
        client.preflight()
    except ModelError as exc:
        print(f"\ncannot reach the model: {exc}", file=sys.stderr)
        print(
            f"set {API_KEY_NAME} in this shell, or run `ant auth login`; "
            "the key is used host-side only and never enters the container",
            file=sys.stderr,
        )
        return 2
    print("credentials and model id check out")

    config = RunConfig(
        repo_cache=Path(args.repo_cache),
        artifacts_dir=Path(args.artifacts) if args.artifacts else None,
        grade=args.grade,
        quiet=args.quiet,
    )

    repos: dict = {}
    runs: List[TaskRun] = []
    for task, task_dir in tasks:
        if task.repository not in repos:
            commits = [t.commit for t, _ in tasks if t.repository == task.repository]
            repos[task.repository] = ensure_repo(
                task.repository, Path(args.repo_cache), commits, quiet=args.quiet
            )

        print(f"\n=== {task.task_id}  [{task.category.value}]")
        run = run_task(
            task,
            task_dir,
            client=client,
            agent_config=agent_config,
            repo=repos[task.repository],
            config=config,
        )
        runs.append(run)
        print(_describe(run))

    if args.patch_out and runs[0].patch is not None:
        Path(args.patch_out).write_text(runs[0].patch.text, encoding="utf-8")
        print(f"\nwrote {args.patch_out}")

    if args.json:
        Path(args.json).write_text(
            json.dumps([run.to_dict() for run in runs], indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")

    broken = [run for run in runs if run.infrastructure_error]
    if broken:
        print(f"\n{len(broken)}/{len(runs)} run(s) failed for infrastructure reasons")
    # A non-zero exit means the harness broke, never that the agent failed the
    # task: "the model could not solve it" is a result, not an error.
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
