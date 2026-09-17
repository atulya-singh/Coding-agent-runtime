"""Checkpoints on disk: one directory per task, one file pair per checkpoint.

Plain files, not a database, for the same reason every other persistence
mechanism here is plain files (`Tasks.harness`'s repo cache, `Evaluation`'s
artifacts directory): the records are read far more often by a person tailing a
directory than by code, and a run that crashes mid-write must leave something
inspectable rather than a locked page.

    <root>/<task_id>/checkpoints/000000-2f1a9c.json
                                 000000-2f1a9c.patch
                                 000001-8bd004.json   ...
                    /latest.json

The write order inside `save` is the only subtle part, and it is load-bearing:
the sidecar patch first, then the record, then `latest.json`. A crash at any
point leaves `latest` pointing at a checkpoint that is complete on disk, because
`latest` is only ever updated after everything it names has landed. Each
individual file is written to a `.tmp` and renamed, so no reader ever sees a
half-written one -- `os.replace` is atomic on every platform this runs on.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from .checkpoint import CHECKPOINT_VERSION, Checkpoint

#: Mirrors Tasks.harness.DEFAULT_REPO_CACHE: recoverable state, not precious
#: state. A checkpoint is worth surviving a crash, not a reboot.
DEFAULT_STATE_ROOT = Path(tempfile.gettempdir()) / "agent-runtime-state"

CHECKPOINTS_DIRNAME = "checkpoints"
LATEST_FILENAME = "latest.json"


class CheckpointError(Exception):
    """A checkpoint could not be stored, found, or trusted once found."""


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class CheckpointStore:
    def __init__(self, root: Path = DEFAULT_STATE_ROOT):
        self.root = Path(root)

    # ---- layout ------------------------------------------------------------

    def task_dir(self, task_id: str) -> Path:
        return self.root / task_id

    def checkpoints_dir(self, task_id: str) -> Path:
        return self.task_dir(task_id) / CHECKPOINTS_DIRNAME

    # ---- writing -----------------------------------------------------------

    def save(self, checkpoint: Checkpoint) -> Path:
        directory = self.checkpoints_dir(checkpoint.task_id)
        directory.mkdir(parents=True, exist_ok=True)
        stem = checkpoint.filename_stem()
        record_path = directory / f"{stem}.json"

        if checkpoint.patch_text:
            _write_atomic(directory / f"{stem}.patch", checkpoint.patch_text)
        _write_atomic(record_path, json.dumps(checkpoint.to_dict(), indent=2))
        _write_atomic(
            self.task_dir(checkpoint.task_id) / LATEST_FILENAME,
            json.dumps(
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "step": checkpoint.step,
                    "status": checkpoint.status,
                    "filename": record_path.name,
                    "created_at": checkpoint.created_at,
                },
                indent=2,
            ),
        )
        return record_path

    # ---- reading -----------------------------------------------------------

    def _read(self, record_path: Path) -> Checkpoint:
        try:
            data = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot read checkpoint {record_path}: {exc}") from exc

        stored_version = data.get("checkpoint_version")
        if stored_version != CHECKPOINT_VERSION:
            raise CheckpointError(
                f"{record_path.name} was written by checkpoint version {stored_version}, "
                f"this is version {CHECKPOINT_VERSION}; its fields may not mean the same "
                "thing, so it is not resumed"
            )

        sidecar = record_path.with_suffix(".patch")
        patch_text = sidecar.read_text(encoding="utf-8") if sidecar.exists() else ""
        return Checkpoint.from_dict(data, patch_text=patch_text)

    def record_paths(self, task_id: str) -> List[Path]:
        """Every stored record, in run order. `.tmp` files are not records --
        one left behind by a crash must not be mistaken for a checkpoint."""
        directory = self.checkpoints_dir(task_id)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.json"))

    def list_checkpoints(self, task_id: str) -> List[Checkpoint]:
        return [self._read(path) for path in self.record_paths(task_id)]

    def load(self, task_id: str, checkpoint_id: str) -> Checkpoint:
        for path in self.record_paths(task_id):
            if path.stem.endswith(f"-{checkpoint_id}"):
                return self._read(path)
        raise CheckpointError(f"no checkpoint {checkpoint_id} for task {task_id}")

    def latest(self, task_id: str) -> Optional[Checkpoint]:
        """The newest complete checkpoint, or None if this task has none."""
        pointer = self.task_dir(task_id) / LATEST_FILENAME
        if not pointer.exists():
            return None
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot read {pointer}: {exc}") from exc

        record_path = self.checkpoints_dir(task_id) / str(data.get("filename", ""))
        if not record_path.exists():
            raise CheckpointError(
                f"{pointer} points at {data.get('filename')!r}, which is not there"
            )
        return self._read(record_path)

    def history(self, task_id: str) -> List[Checkpoint]:
        """The chain ending at `latest`, oldest first.

        Not the same as `list_checkpoints`: running a task twice into one root
        leaves two interleaved chains in the directory, and only the parent links
        say which checkpoints belong to the run that is actually current.
        """
        tip = self.latest(task_id)
        if tip is None:
            return []
        by_id = {checkpoint.checkpoint_id: checkpoint for checkpoint in self.list_checkpoints(task_id)}
        chain = [tip]
        seen = {tip.checkpoint_id}
        while chain[-1].parent_checkpoint_id:
            parent = by_id.get(chain[-1].parent_checkpoint_id)
            if parent is None or parent.checkpoint_id in seen:
                break
            seen.add(parent.checkpoint_id)
            chain.append(parent)
        return list(reversed(chain))

    # ---- housekeeping ------------------------------------------------------

    def task_ids(self) -> List[str]:
        if not self.root.is_dir():
            return []
        return sorted(path.name for path in self.root.iterdir() if (path / LATEST_FILENAME).exists())

    def clear(self, task_id: str) -> None:
        shutil.rmtree(self.task_dir(task_id), ignore_errors=True)
