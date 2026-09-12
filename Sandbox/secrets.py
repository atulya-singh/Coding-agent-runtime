from __future__ import annotations

import os
import re
from typing import Callable, Optional

# Defense-in-depth denylist. The primary control is architectural: no code path in
# this package forwards host environment variables or credentials into a container.
_SECRET_NAME_PATTERN = re.compile(
    r"(API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE[_-]?KEY|^AUTH|_AUTH$)",
    re.IGNORECASE,
)


class SandboxSecurityError(Exception):
    pass


def assert_no_secrets(env: Optional[dict]) -> None:
    """Raise if `env` contains a key that looks like a credential."""
    for key in env or {}:
        if _SECRET_NAME_PATTERN.search(key):
            raise SandboxSecurityError(
                f"Refusing to inject '{key}' into the sandbox: name looks like a credential. "
                "Fetch secrets host-side via SecretBroker and keep them out of the container."
            )


class SecretBroker:
    """Host-side-only credential access. Never pass its output into a Sandbox.

    Real secret storage (Vault, AWS Secrets Manager, a KMS-backed API, ...) belongs
    behind `fetch_fn`; the default just reads host process env so callers have one
    seam to swap in a real backend later without changing call sites.
    """

    def __init__(self, fetch_fn: Optional[Callable[[str], Optional[str]]] = None):
        self._fetch_fn = fetch_fn or (lambda name: os.environ.get(name))

    def get_secret(self, name: str) -> str:
        value = self._fetch_fn(name)
        if value is None:
            raise KeyError(f"Secret '{name}' is not available")
        return value
