"""Turning token counts into dollars.

Rates live in the config file, not in this module, for two reasons: they change
on a schedule that has nothing to do with this code, and a wrong-but-plausible
hardcoded number would silently corrupt the cost metrics plan.md Phase 8 is
built to report. A model with no rates configured reports `None` rather than
zero -- an unknown cost and a free call are not the same measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PER_MTOK = 1_000_000


@dataclass
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float
    # Default to the plain input rate when a model's cache rates are unknown, so
    # a cached run is never reported as cheaper than it might be.
    cache_write_per_mtok: Optional[float] = None
    cache_read_per_mtok: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ModelPricing":
        return cls(
            input_per_mtok=float(data["input_per_mtok"]),
            output_per_mtok=float(data["output_per_mtok"]),
            cache_write_per_mtok=(
                float(data["cache_write_per_mtok"])
                if data.get("cache_write_per_mtok") is not None
                else None
            ),
            cache_read_per_mtok=(
                float(data["cache_read_per_mtok"])
                if data.get("cache_read_per_mtok") is not None
                else None
            ),
        )

    def to_dict(self) -> dict:
        return {
            "input_per_mtok": self.input_per_mtok,
            "output_per_mtok": self.output_per_mtok,
            "cache_write_per_mtok": self.cache_write_per_mtok,
            "cache_read_per_mtok": self.cache_read_per_mtok,
        }


def cost_usd(usage: dict, pricing: Optional[ModelPricing]) -> Optional[float]:
    """Cost of `usage`, or None when the model's rates are not configured."""
    if pricing is None:
        return None
    cache_write = (
        pricing.cache_write_per_mtok
        if pricing.cache_write_per_mtok is not None
        else pricing.input_per_mtok
    )
    cache_read = (
        pricing.cache_read_per_mtok
        if pricing.cache_read_per_mtok is not None
        else pricing.input_per_mtok
    )
    total = (
        usage.get("input_tokens", 0) * pricing.input_per_mtok
        + usage.get("output_tokens", 0) * pricing.output_per_mtok
        + usage.get("cache_creation_input_tokens", 0) * cache_write
        + usage.get("cache_read_input_tokens", 0) * cache_read
    )
    return total / PER_MTOK
