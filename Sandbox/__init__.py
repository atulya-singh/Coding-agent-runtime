"""Phase 2: Docker-based sandbox. Lifecycle: CREATE -> INITIALIZE -> RUN -> COLLECT -> DESTROY.

Security: sandboxes never receive API keys or auth tokens. Host-side code fetches
credentials via SecretBroker and uses them outside the container; see secrets.py.
"""
from .config import SandboxConfig
from .manager import Sandbox, SandboxResult, SandboxState
from .secrets import SandboxSecurityError, SecretBroker, assert_no_secrets

__all__ = [
    "Sandbox",
    "SandboxConfig",
    "SandboxResult",
    "SandboxState",
    "SandboxSecurityError",
    "SecretBroker",
    "assert_no_secrets",
]
