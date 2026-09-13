from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING, Optional

from Execution.tools.base import ToolTimeoutError

from .secrets import assert_no_secrets

if TYPE_CHECKING:
    from .config import SandboxConfig

logger = logging.getLogger("sandbox.docker")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

DEFAULT_DOCKER_TIMEOUT = 30.0


def _docker(
    args: list[str], timeout: float = DEFAULT_DOCKER_TIMEOUT, stdin: Optional[str] = None
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", *args], input=stdin, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolTimeoutError(f"docker {' '.join(args)} timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("docker is not installed or not on PATH") from exc


def image_exists(image: str) -> bool:
    return _docker(["image", "inspect", image]).returncode == 0


def pull_image(image: str, timeout: float = 300.0) -> None:
    proc = _docker(["pull", image], timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"docker pull failed: {proc.stderr.strip()}")


def _base_create_args(name: str, config: "SandboxConfig", workdir: str) -> list[str]:
    # Re-checked here (not just in SandboxConfig.__post_init__) because config.env
    # is a mutable dict that could be edited after construction.
    assert_no_secrets(config.env)
    args = [
        "create",
        "--name", name,
        "--cpus", str(config.cpus),
        "--memory", f"{config.memory_gb}g",
        "--memory-swap", f"{config.memory_gb}g",
        "--pids-limit", str(config.pids_limit),
        "--network", "bridge" if config.network_enabled else "none",
        "--workdir", workdir,
    ]
    for key, value in config.env.items():
        args += ["-e", f"{key}={value}"]
    return args


def build_create_args(name: str, config: "SandboxConfig", workdir: str, enforce_disk_quota: bool = True) -> list[str]:
    args = _base_create_args(name, config, workdir)

    if config.read_only_root:
        tmp_size = f"{config.disk_gb}g" if config.disk_gb else "2g"
        args += [
            "--read-only",
            "--tmpfs", f"/tmp:rw,exec,size={tmp_size}",
            "--tmpfs", f"{workdir}:rw,exec,size={tmp_size}",
        ]
    elif config.disk_gb and enforce_disk_quota:
        args += ["--storage-opt", f"size={config.disk_gb}g"]

    args += [config.image, "sleep", "infinity"]
    return args


def create_container(name: str, config: "SandboxConfig", workdir: str = "/task") -> str:
    args = build_create_args(name, config, workdir)
    proc = _docker(args, timeout=60.0)

    if proc.returncode != 0 and config.disk_gb and not config.read_only_root:
        # --storage-opt requires a storage driver that supports quotas (e.g. overlay2
        # on xfs with pquota); many hosts -- notably Docker Desktop -- don't support
        # it. Retry without it rather than failing the whole sandbox, but say so.
        logger.warning(
            "docker create: --storage-opt unsupported on this host; disk quota will "
            "not be enforced for this sandbox (%s)",
            proc.stderr.strip(),
        )
        args = build_create_args(name, config, workdir, enforce_disk_quota=False)
        proc = _docker(args, timeout=60.0)

    if proc.returncode != 0:
        raise RuntimeError(f"docker create failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def start_container(container_id: str) -> None:
    proc = _docker(["start", container_id], timeout=30.0)
    if proc.returncode != 0:
        raise RuntimeError(f"docker start failed: {proc.stderr.strip()}")


def exec_in_container(
    container_id: str,
    command: str,
    cwd: Optional[str] = None,
    timeout: float = 60.0,
    stdin: Optional[str] = None,
) -> dict:
    args = ["exec"]
    if stdin is not None:
        # Keeps large payloads (a file being written, a search request) out of argv
        # and therefore out of the command string that gets logged.
        args.append("-i")
    if cwd:
        args += ["--workdir", cwd]
    args += [container_id, "sh", "-c", command]
    proc = _docker(args, timeout=timeout, stdin=stdin)
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "success": proc.returncode == 0,
    }


def copy_into_container(container_id: str, src: str, dest: str, timeout: float = 60.0) -> None:
    proc = _docker(["cp", src, f"{container_id}:{dest}"], timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"docker cp (into container) failed: {proc.stderr.strip()}")


def copy_from_container(container_id: str, src: str, dest: str, timeout: float = 60.0) -> None:
    proc = _docker(["cp", f"{container_id}:{src}", dest], timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"docker cp (from container) failed: {proc.stderr.strip()}")


def stop_container(container_id: str, timeout: float = 10.0) -> None:
    # Best-effort: destroy() may be called on an already-gone container during
    # cleanup paths, and that should not raise.
    _docker(["stop", "-t", str(int(timeout)), container_id], timeout=timeout + 15)


def remove_container(container_id: str, force: bool = True) -> None:
    args = ["rm", "-f", container_id] if force else ["rm", container_id]
    _docker(args, timeout=30.0)
