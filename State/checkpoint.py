"""What a run looks like written down: enough to rebuild it somewhere else.

The sandbox is ephemeral by design (Principle 2) -- there is no `docker commit`,
no pause, no volume anywhere in `Sandbox/`, and adding one would weaken the
isolation boundary this whole project rests on. So a checkpoint cannot be an
image. It has to be the two things that actually constitute a run in progress:

    the conversation   the exact message list, verbatim and replayable
    the work so far    the accumulated diff against the base commit

Given those, resuming is rebuilding: fresh container from the base commit, the
diff applied on top, setup re-run (already deterministic -- every grading pass
does it from scratch today), conversation rehydrated, loop continued. Nothing is
recovered from the dead container, so nothing depends on it having survived.

Pure data. No import of Agent, Sandbox or Evaluation, so this module can be read
back by anything -- including a future tool that never runs an agent at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

#: Bumped when the meaning of a field changes, mirroring EVALUATOR_VERSION in
#: Evaluation/result.py. A stored checkpoint from a different version is refused
#: rather than resumed under rules it was not written under.
#:
#: 2: `retries` added. A v1 record would read back as "no retries", which is a
#: claim about the environment it never actually made -- and resuming from one
#: would silently reset a count the recovered run then continues from zero.
CHECKPOINT_VERSION = 2

#: Status of a checkpoint taken mid-run. Every other value is an Agent.loop
#: StopReason -- the two share one field so that "still going" and "ended, here
#: is why" are answered in the same place rather than by two flags that can
#: disagree.
IN_PROGRESS = "in_progress"

#: Stop reasons after which there is nothing to continue: the agent said it was
#: done. Kept as strings so this module stays free of the Agent package; the
#: selftest asserts the two never drift apart.
AGENT_DECIDED = ("finished", "finished_implicit")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_checkpoint_id() -> str:
    return uuid4().hex[:12]


@dataclass
class Checkpoint:
    task_id: str
    step: int = 0
    status: str = IN_PROGRESS
    checkpoint_id: str = field(default_factory=new_checkpoint_id)
    #: The checkpoint this one continues from, so a resumed run can be walked
    #: back to the run it recovered -- which is the whole claim the kill/resume
    #: experiment makes, and it must be checkable from the record alone.
    parent_checkpoint_id: str = ""
    created_at: str = field(default_factory=utc_now)

    model: str = ""
    provider: str = ""
    #: The AgentConfig this ran under. Resuming with different budgets is
    #: allowed, but the record must show it happened.
    agent_config: dict = field(default_factory=dict)
    #: Same shape as EvaluationResult.provenance: repository, commit, image.
    provenance: dict = field(default_factory=dict)

    messages: List[dict] = field(default_factory=list)
    tool_calls: List[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    cost_usd: Optional[float] = None
    summary: str = ""
    error: str = ""
    #: Transient failures the harness absorbed before this point (Recovery
    #: RetryAttempt dicts). Carried across a resume so the recovered run's
    #: record covers the whole attempt, not just the part after the crash.
    retries: List[dict] = field(default_factory=list)

    patch_sha256: str = ""
    patch_meta: dict = field(default_factory=dict)
    #: Why the diff could not be taken, when it could not. A checkpoint with this
    #: set records what the agent said but not what it wrote, so resuming from it
    #: would silently discard work -- `resumable` refuses it for that reason.
    patch_error: str = ""

    checkpoint_version: int = CHECKPOINT_VERSION

    #: The diff itself. Stored beside the JSON as a `.patch` sidecar rather than
    #: inline, so the records stay small enough to grep and diff by eye; the
    #: store reads it back into this field.
    patch_text: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status != IN_PROGRESS

    @property
    def resumable(self) -> str:
        """Empty if this can be continued, else why it cannot."""
        if self.patch_error:
            return f"the work was never captured: {self.patch_error}"
        if self.status in AGENT_DECIDED:
            return f"the run already ended on its own terms ({self.status})"
        return ""

    def resume_state(self) -> dict:
        """What AgentLoop needs to pick up where this left off.

        A plain dict, matching `AgentLoop(resume_from=...)`: the loop must not
        learn the checkpoint format, and this module must not learn the loop's
        types. The seam is five well-known keys and nothing else.
        """
        return {
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "turns": self.step,
            "retries": self.retries,
        }

    def filename_stem(self) -> str:
        """Zero-padded so a plain directory listing is in run order."""
        return f"{self.step:06d}-{self.checkpoint_id}"

    def to_dict(self) -> dict:
        # patch_text is deliberately absent: it lives in the sidecar file.
        return {
            "checkpoint_version": self.checkpoint_version,
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "task_id": self.task_id,
            "step": self.step,
            "status": self.status,
            "created_at": self.created_at,
            "model": self.model,
            "provider": self.provider,
            "agent_config": self.agent_config,
            "provenance": self.provenance,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "summary": self.summary,
            "error": self.error,
            "retries": self.retries,
            "patch_sha256": self.patch_sha256,
            "patch_meta": self.patch_meta,
            "patch_error": self.patch_error,
            "tool_calls": self.tool_calls,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict, patch_text: str = "") -> "Checkpoint":
        data = data or {}
        return cls(
            task_id=str(data.get("task_id", "")),
            step=int(data.get("step", 0)),
            status=str(data.get("status", IN_PROGRESS)),
            checkpoint_id=str(data.get("checkpoint_id") or new_checkpoint_id()),
            parent_checkpoint_id=str(data.get("parent_checkpoint_id", "")),
            created_at=str(data.get("created_at", "")),
            model=str(data.get("model", "")),
            provider=str(data.get("provider", "")),
            agent_config=data.get("agent_config") or {},
            provenance=data.get("provenance") or {},
            messages=data.get("messages") or [],
            tool_calls=data.get("tool_calls") or [],
            usage=data.get("usage") or {},
            cost_usd=data.get("cost_usd"),
            summary=str(data.get("summary", "")),
            error=str(data.get("error", "")),
            retries=data.get("retries") or [],
            patch_sha256=str(data.get("patch_sha256", "")),
            patch_meta=data.get("patch_meta") or {},
            patch_error=str(data.get("patch_error", "")),
            checkpoint_version=int(data.get("checkpoint_version", CHECKPOINT_VERSION)),
            patch_text=patch_text,
        )

    def summary_line(self) -> str:
        line = f"step {self.step:>3}  {self.status:<18} {self.checkpoint_id}"
        if self.patch_meta.get("files"):
            line += f"  {len(self.patch_meta['files'])} file(s) changed"
        if self.retries:
            line += f"  {len(self.retries)} retried"
        if self.patch_error:
            line += f"  !! {self.patch_error[:60]}"
        return line
