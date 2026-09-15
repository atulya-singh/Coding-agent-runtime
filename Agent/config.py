"""Agent configuration: how the loop talks to a model, not what it's solving.

Kept separate from Tasks.schema.Task on purpose -- a Task describes what to
solve and how it's graded; AgentConfig describes the model and the budgets the
loop runs under. The two only meet at run time (a future Agent.run.run_task
takes both), so a task's grading contract never depends on which model or
budget attempted it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .pricing import ModelPricing

DEFAULT_MODEL = "claude-opus-5"
CONFIG_FILENAME = "agent_config.yaml"


#: Adaptive thinking is the only on-mode on the current models; `budget_tokens`
#: is rejected by them. "off" omits the parameter's on-mode entirely.
THINKING_MODES = ("adaptive", "off")
#: output_config.effort. Unsupported on Haiku 4.5, which rejects the field.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    # Per-request output cap passed to the Messages API. A coding agent writes
    # whole files in one block, and a turn cut off at max_tokens is thrown away
    # (its last tool call may be truncated), so this is not a place to economise.
    max_tokens: int = 16000
    # How hard the model thinks before answering, and how deeply. Both are real
    # experimental variables -- plan.md Phase 13 compares reasoning strategies --
    # so they are recorded in config rather than hardcoded at the call site.
    thinking: str = "adaptive"
    effort: str = "high"
    # Hard stop on loop iterations. Also the denominator Phase 5's kill/resume
    # experiment uses for "10% of expected execution" -- see plan.md Phase 5.
    max_turns: int = 40
    # Cumulative budgets for the whole run, independent of max_turns (a handful
    # of long turns can exhaust either well before max_turns is reached).
    # None means unbounded.
    max_total_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    # Per-request timeout, in seconds.
    request_timeout: float = 120.0
    # Token rates per model, keyed by model id. Not hardcoded in pricing.py: see
    # that module for why. A model missing here reports an unknown cost rather
    # than a zero one.
    pricing: Dict[str, ModelPricing] = field(default_factory=dict)

    def pricing_for_model(self) -> Optional[ModelPricing]:
        return self.pricing.get(self.model)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": self.thinking,
            "effort": self.effort,
            "max_turns": self.max_turns,
            "max_total_tokens": self.max_total_tokens,
            "max_cost_usd": self.max_cost_usd,
            "request_timeout": self.request_timeout,
            "pricing": {name: rates.to_dict() for name, rates in self.pricing.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfig":
        data = data or {}
        return cls(
            model=str(data.get("model", DEFAULT_MODEL)),
            max_tokens=int(data.get("max_tokens", 16000)),
            thinking=str(data.get("thinking", "adaptive")),
            effort=str(data.get("effort", "high")),
            max_turns=int(data.get("max_turns", 40)),
            max_total_tokens=(
                int(data["max_total_tokens"])
                if data.get("max_total_tokens") is not None
                else None
            ),
            max_cost_usd=(
                float(data["max_cost_usd"]) if data.get("max_cost_usd") is not None else None
            ),
            request_timeout=float(data.get("request_timeout", 120.0)),
            pricing={
                name: ModelPricing.from_dict(rates)
                for name, rates in (data.get("pricing") or {}).items()
            },
        )


def validate_config(config: AgentConfig) -> List[str]:
    """Human-readable errors, mirroring Tasks.schema.validate_task's shape."""
    errors: List[str] = []
    if not config.model:
        errors.append("model must be set")
    if config.max_tokens <= 0:
        errors.append(f"max_tokens must be positive, got {config.max_tokens}")
    if config.thinking not in THINKING_MODES:
        errors.append(
            f"thinking must be one of {', '.join(THINKING_MODES)}, got {config.thinking!r}"
        )
    if config.effort and config.effort not in EFFORT_LEVELS:
        errors.append(
            f"effort must be one of {', '.join(EFFORT_LEVELS)}, got {config.effort!r}"
        )
    if config.max_turns <= 0:
        errors.append(f"max_turns must be positive, got {config.max_turns}")
    if config.max_total_tokens is not None and config.max_total_tokens <= 0:
        errors.append(
            f"max_total_tokens must be positive when set, got {config.max_total_tokens}"
        )
    if config.max_cost_usd is not None:
        if config.max_cost_usd <= 0:
            errors.append(f"max_cost_usd must be positive when set, got {config.max_cost_usd}")
        elif config.pricing_for_model() is None:
            # The budget would never trigger: cost is unknown without rates.
            errors.append(
                f"max_cost_usd is set but no pricing is configured for model "
                f"'{config.model}'; add it under `pricing` or unset the budget"
            )
    if config.request_timeout <= 0:
        errors.append(f"request_timeout must be positive, got {config.request_timeout}")
    return errors


def load_config(path: Path) -> AgentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no agent config at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AgentConfig.from_dict(data)


def save_config(config: AgentConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
