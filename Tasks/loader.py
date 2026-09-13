from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .schema import Task, validate_task

TASK_FILENAME = "task.yaml"


def load_task(task_dir: Path) -> Task:
    task_dir = Path(task_dir)
    task_file = task_dir / TASK_FILENAME
    if not task_file.exists():
        raise FileNotFoundError(f"no {TASK_FILENAME} in {task_dir}")
    with open(task_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Task.from_dict(data)


def save_task(task: Task, task_dir: Path) -> None:
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / TASK_FILENAME, "w", encoding="utf-8") as f:
        yaml.safe_dump(task.to_dict(), f, sort_keys=False)


def discover_task_dirs(dataset_dir: Path) -> List[Path]:
    dataset_dir = Path(dataset_dir)
    return sorted(p.parent for p in dataset_dir.glob(f"*/{TASK_FILENAME}"))


def load_all_tasks(dataset_dir: Path) -> List[Task]:
    return [load_task(d) for d in discover_task_dirs(dataset_dir)]


def validate_dataset(dataset_dir: Path) -> Dict[str, List[str]]:
    """Return {task_dir_name: [errors]} for every task with validation problems."""
    dataset_dir = Path(dataset_dir)
    report: Dict[str, List[str]] = {}
    seen_ids: set = set()
    for task_dir in discover_task_dirs(dataset_dir):
        try:
            task = load_task(task_dir)
        except Exception as exc:
            report[task_dir.name] = [f"failed to load: {exc}"]
            continue
        errors = validate_task(task, task_dir)
        if task.task_id in seen_ids:
            errors.append(f"duplicate task_id: {task.task_id}")
        seen_ids.add(task.task_id)
        if errors:
            report[task_dir.name] = errors
    return report
