from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .secrets import assert_no_secrets

# Defaults mirror the example policy in plan.md Phase 2.
DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_CPUS = 2.0
DEFAULT_MEMORY_GB = 4.0
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_PIDS_LIMIT = 100
DEFAULT_DISK_GB = 10.0


@dataclass
class SandboxConfig:
    image: str = DEFAULT_IMAGE
    cpus: float = DEFAULT_CPUS
    memory_gb: float = DEFAULT_MEMORY_GB
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    pids_limit: int = DEFAULT_PIDS_LIMIT
    disk_gb: Optional[float] = DEFAULT_DISK_GB
    network_enabled: bool = False
    # Non-secret config only -- enforced below. Never put API keys/tokens here.
    env: dict = field(default_factory=dict)
    # Off by default: read-only root + sized tmpfs gives real disk-quota enforcement
    # but its tmpfs is memory-backed, so it can contend with `memory_gb`. Opt in.
    read_only_root: bool = False

    def __post_init__(self) -> None:
        assert_no_secrets(self.env)
        if self.cpus <= 0:
            raise ValueError("cpus must be > 0")
        if self.memory_gb <= 0:
            raise ValueError("memory_gb must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.pids_limit <= 0:
            raise ValueError("pids_limit must be > 0")
