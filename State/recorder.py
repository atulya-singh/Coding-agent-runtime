"""Turn the loop's per-turn snapshots into saved checkpoints.

This is the only place that holds all three things at once: the snapshot the
loop hands out, the container the work happened in, and the store it goes to.
Keeping it here is what lets `Agent.loop` stay ignorant of checkpointing and
`State.checkpoint` stay ignorant of agents.

The expensive half is the diff. It is taken fresh at every turn boundary rather
than accumulated incrementally, because an incremental record would have to
model what each tool did to the filesystem -- and `run_command` can do anything.
Copying the tree out and diffing it asks the container what actually happened
instead of predicting it, which is the same reason `Evaluation` grades a patch
rather than trusting a log.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from Evaluation.patch import collect_patch

from .checkpoint import Checkpoint
from .store import CheckpointStore

if TYPE_CHECKING:  # import-free at runtime: State never depends on Agent
    from Agent.loop import Snapshot

logger = logging.getLogger("state")


class CheckpointRecorder:
    """Callable to hand to `AgentLoop(on_turn=...)`."""

    def __init__(
        self,
        store: CheckpointStore,
        sandbox,
        base_dir: Path,
        agent_config: Optional[dict] = None,
        provenance: Optional[dict] = None,
        provider: str = "",
        parent_checkpoint_id: str = "",
    ):
        self.store = store
        self.sandbox = sandbox
        self.base_dir = Path(base_dir)
        self.agent_config = agent_config or {}
        self.provenance = provenance or {}
        self.provider = provider
        #: Set when resuming, so the recovered run's first checkpoint points back
        #: at the killed run's last one and the lineage survives the crash.
        self.last_checkpoint_id = parent_checkpoint_id
        self.saved: int = 0

    def __call__(self, snapshot: "Snapshot") -> Checkpoint:
        patch_text, patch_sha, patch_meta, patch_error = self._current_diff()
        checkpoint = Checkpoint(
            task_id=snapshot.task_id,
            step=snapshot.step,
            status=snapshot.status,
            parent_checkpoint_id=self.last_checkpoint_id,
            model=snapshot.model,
            provider=self.provider,
            agent_config=self.agent_config,
            provenance=self.provenance,
            messages=snapshot.messages,
            tool_calls=snapshot.tool_calls,
            usage=snapshot.usage,
            cost_usd=snapshot.cost_usd,
            summary=snapshot.summary,
            error=snapshot.error,
            retries=snapshot.retries,
            patch_sha256=patch_sha,
            patch_meta=patch_meta,
            patch_error=patch_error,
            patch_text=patch_text,
        )
        self.store.save(checkpoint)
        self.last_checkpoint_id = checkpoint.checkpoint_id
        self.saved += 1
        return checkpoint

    def _current_diff(self):
        """The work so far, or why it could not be taken.

        Never raises. A container that has torn itself down after a timeout is a
        normal ending, not a bug, and the terminal checkpoint for that run still
        has a conversation worth keeping -- it is only the *resuming* of such a
        checkpoint that has to be refused, which `Checkpoint.resumable` does.
        """
        if getattr(self.sandbox, "container_id", None) is None:
            return "", "", {}, "the container was gone before the diff could be taken"
        try:
            patch = collect_patch(self.sandbox, self.base_dir)
        except Exception as exc:  # noqa: BLE001 -- recorded, never raised
            logger.warning("could not collect the diff for a checkpoint: %s", exc)
            return "", "", {}, f"{type(exc).__name__}: {exc}"
        return patch.text, patch.sha256, patch.to_dict(), ""
