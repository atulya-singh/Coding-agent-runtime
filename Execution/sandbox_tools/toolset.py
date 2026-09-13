"""The Tool Manager: every agent tool, executed inside a Sandbox.

This is the layer plan.md puts between the agent and the sandbox manager, and it is
what satisfies the Phase 1 requirement that the model never executes anything on the
host. `Execution.tools` still holds the host-side primitives (used by the harness
itself); anything the *model* drives goes through a SandboxToolset instead.

Two execution paths, both audited through Sandbox.run_command:

  run_command / run_tests / git_diff   -> shell commands, run directly
  read_file / write_file / edit_file /
  search_code                          -> dispatched to toolhost.py in the container

The toolhost exists because shelling out to cat/sed/grep would make tool behaviour
depend on the task image's userland, and would need every path and payload escaped
into a shell string. One file, one `docker cp`, JSON over stdin/stdout instead.

Contract kept identical to the host-side tools: every call returns a ToolResult and
emits exactly one `tool_call` log event, whether it succeeded, failed, or timed out.
"""
from __future__ import annotations

import json
import posixpath
import shlex
from pathlib import Path
from typing import Any, Optional

from Sandbox.manager import REPO_DIR, TASK_ROOT, Sandbox
from Sandbox.secrets import assert_no_secrets

from ..tools.base import ToolExecutionError, ToolResult, ToolTimeoutError, log_tool_event, tool
from ..tools.execution import DEFAULT_COMMAND_TIMEOUT, DEFAULT_TEST_TIMEOUT, truncate_output

#: Bumped when the host/container wire format changes; a mismatch fails loudly at
#: install time rather than producing confusing results mid-run.
PROTOCOL_VERSION = 1

#: Deliberately outside /task: not part of the repository the agent edits, and not
#: swept up by collect_results() into the graded patch.
TOOLHOST_DIR = "/opt/agent-runtime"
TOOLHOST_PATH = f"{TOOLHOST_DIR}/toolhost.py"
TOOLHOST_SOURCE = Path(__file__).with_name("toolhost.py")

DEFAULT_FILE_TIMEOUT = 30.0
DEFAULT_SEARCH_TIMEOUT = 60.0
DEFAULT_GIT_TIMEOUT = 30.0

#: toolhost reports an error_type; map the ones with a natural Python equivalent so
#: callers can catch what they'd catch from the host-side tools.
_ERROR_TYPES = {
    "FileNotFoundError": FileNotFoundError,
    "IsADirectoryError": IsADirectoryError,
    "NotADirectoryError": NotADirectoryError,
    "PermissionError": PermissionError,
    "ValueError": ValueError,
    "UnicodeDecodeError": ValueError,
}


class SandboxToolError(ToolExecutionError):
    """A tool failed inside the sandbox, or the toolhost protocol broke."""


class SandboxToolset:
    """All seven agent tools, bound to one Sandbox.

    >>> with Sandbox(task_id="demo") as sb:
    ...     sb.initialize("/path/to/repo")
    ...     tools = SandboxToolset(sb)
    ...     tools.read_file("src/app.py")          # ToolResult
    ...     tools.call("run_tests", {"command": "pytest -q"})
    """

    def __init__(
        self,
        sandbox: Sandbox,
        workdir: str = REPO_DIR,
        root: str = TASK_ROOT,
        python: str = "python3",
    ):
        self.sandbox = sandbox
        self.workdir = posixpath.normpath(workdir)
        self.root = posixpath.normpath(root)
        self.python = python
        self._toolhost_ready = False

    # ---- tools: files ------------------------------------------------------------

    @tool("read_file")
    def read_file(self, path: str, timeout: float = DEFAULT_FILE_TIMEOUT):
        return self._toolhost("read_file", {"path": path}, timeout=timeout)

    @tool("write_file")
    def write_file(self, path: str, content: str, timeout: float = DEFAULT_FILE_TIMEOUT):
        return self._toolhost("write_file", {"path": path, "content": content}, timeout=timeout)

    @tool("edit_file")
    def edit_file(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        timeout: float = DEFAULT_FILE_TIMEOUT,
    ):
        return self._toolhost(
            "edit_file",
            {
                "path": path,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all,
            },
            timeout=timeout,
        )

    @tool("search_code")
    def search_code(
        self,
        query: str,
        path: str = ".",
        file_pattern: Optional[str] = None,
        case_sensitive: bool = False,
        max_results: int = 200,
        timeout: float = DEFAULT_SEARCH_TIMEOUT,
    ):
        return self._toolhost(
            "search_code",
            {
                "query": query,
                "path": path,
                "file_pattern": file_pattern,
                "case_sensitive": case_sensitive,
                "max_results": max_results,
            },
            timeout=timeout,
        )

    # ---- tools: execution --------------------------------------------------------

    @tool("run_command")
    def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: float = DEFAULT_COMMAND_TIMEOUT,
        env: Optional[dict] = None,
    ):
        return self._shell(command, cwd, timeout, env)

    @tool("run_tests")
    def run_tests(
        self,
        command: str = "pytest -q",
        cwd: Optional[str] = None,
        timeout: float = DEFAULT_TEST_TIMEOUT,
        env: Optional[dict] = None,
    ):
        return self._shell(command, cwd, timeout, env)

    @tool("git_diff")
    def git_diff(
        self,
        cwd: Optional[str] = None,
        staged: bool = False,
        path: Optional[str] = None,
        timeout: float = DEFAULT_GIT_TIMEOUT,
    ):
        argv = ["git", "diff"]
        if staged:
            argv.append("--staged")
        if path:
            argv += ["--", path]

        target = self._resolve(cwd) if cwd else self.workdir
        result = self._exec(" ".join(shlex.quote(part) for part in argv), target, timeout)

        if result.exit_code == 127 or "not found" in result.stderr:
            raise SandboxToolError(
                "git is not installed in the sandbox image. Add "
                "'apt-get update && apt-get install -y git' to the task's setup steps, "
                "or use an image that ships git."
            )
        if not result.success:
            raise SandboxToolError(f"git diff failed (exit {result.exit_code}): {result.stderr.strip()}")

        diff = truncate_output(result.stdout)
        return diff, self._meta(cwd=target, staged=staged, path=path, has_changes=bool(diff.strip()))

    # ---- uniform dispatch --------------------------------------------------------

    #: Tool name -> bound method. The single source of truth for "what can the agent
    #: call", used by call() and tool_specs() alike so the two can never drift.
    @property
    def tools(self) -> dict:
        return {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "search_code": self.search_code,
            "run_command": self.run_command,
            "run_tests": self.run_tests,
            "git_diff": self.git_diff,
        }

    def call(self, tool_name: str, arguments: Optional[dict] = None) -> ToolResult:
        """Invoke a tool by name. Never raises -- a bad name or bad arguments come
        back as a failed ToolResult, because this is fed directly by model output."""
        func = self.tools.get(tool_name)
        if func is None:
            error = f"unknown tool: {tool_name!r} (available: {', '.join(sorted(self.tools))})"
            log_tool_event(tool_name, False, 0.0, error, {"task_id": self.sandbox.task_id})
            return ToolResult(tool=tool_name, success=False, error=error)
        return func(**(arguments or {}))

    def tool_specs(self) -> list:
        """JSON-schema tool definitions to hand to a model's tool-calling API."""
        return [
            {
                "name": "read_file",
                "description": "Read a UTF-8 text file from the task workspace.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Path relative to the repository root, or absolute under /task."}},
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Create or overwrite a file, creating parent directories as needed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "edit_file",
                "description": (
                    "Replace an exact string in a file. old_string must be unique in the "
                    "file unless replace_all is true."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean", "default": False},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
            {
                "name": "search_code",
                "description": "Regex search across files in the workspace. Returns file/line matches.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Python regular expression."},
                        "path": {"type": "string", "default": "."},
                        "file_pattern": {"type": "string", "description": "Glob filter, e.g. '*.py'."},
                        "case_sensitive": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "default": 200},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "run_command",
                "description": "Run a shell command inside the sandbox container.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {"type": "string"},
                        "timeout": {"type": "number"},
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "run_tests",
                "description": "Run the test suite inside the sandbox container.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "default": "pytest -q"},
                        "cwd": {"type": "string"},
                        "timeout": {"type": "number"},
                    },
                },
            },
            {
                "name": "git_diff",
                "description": "Show the working-tree diff of changes made so far.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cwd": {"type": "string"},
                        "staged": {"type": "boolean", "default": False},
                        "path": {"type": "string"},
                    },
                },
            },
        ]

    # ---- toolhost plumbing -------------------------------------------------------

    def install(self, force: bool = False) -> dict:
        """Copy toolhost.py into the container and verify it answers. Idempotent."""
        if self._toolhost_ready and not force:
            return {"installed": False, "path": TOOLHOST_PATH}

        self._require_live()
        self.sandbox.put_file(str(TOOLHOST_SOURCE), TOOLHOST_PATH)
        self._toolhost_ready = True

        pong = self._toolhost_request("ping", {}, timeout=30.0, allow_reinstall=False)[0]
        if pong.get("protocol_version") != PROTOCOL_VERSION:
            raise SandboxToolError(
                f"toolhost protocol mismatch: container reports {pong.get('protocol_version')}, "
                f"host expects {PROTOCOL_VERSION}"
            )
        return {"installed": True, "path": TOOLHOST_PATH, **pong}

    def _toolhost(self, op: str, args: dict, timeout: float):
        output, metadata = self._toolhost_request(op, args, timeout=timeout)
        return output, self._meta(**metadata)

    def _toolhost_request(
        self, op: str, args: dict, timeout: float, allow_reinstall: bool = True
    ) -> tuple:
        if not self._toolhost_ready:
            self.install()

        # Validate host-side too: the container re-checks with realpath, but catching
        # an escape here means it never becomes a command at all.
        if "path" in args and args["path"] is not None:
            self._resolve(args["path"])

        self._require_live()
        request = json.dumps({"op": op, "root": self.root, "workdir": self.workdir, "args": args})
        result = self.sandbox.run_command(
            f"{self.python} {shlex.quote(TOOLHOST_PATH)}",
            cwd=self.workdir,
            timeout=timeout,
            stdin=request,
        )

        if result.timed_out:
            raise ToolTimeoutError(
                f"{op} timed out after {timeout}s (the sandbox was torn down; see run log)"
            )

        if result.exit_code != 0 or not result.stdout:
            # An agent command can delete or clobber the toolhost; reinstall once
            # before treating it as a real protocol failure.
            if allow_reinstall:
                self._toolhost_ready = False
                self.install(force=True)
                return self._toolhost_request(op, args, timeout, allow_reinstall=False)
            raise SandboxToolError(
                f"toolhost failed for op {op!r} (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[:500]}"
            )

        try:
            response = json.loads(result.stdout)
        except ValueError as exc:
            raise SandboxToolError(
                f"toolhost returned unparseable output for op {op!r}: {result.stdout[:500]}"
            ) from exc

        if not response.get("ok"):
            error = response.get("error", "unknown toolhost error")
            raise _ERROR_TYPES.get(response.get("error_type"), SandboxToolError)(error)
        return response.get("output"), response.get("metadata", {})

    # ---- shell plumbing ----------------------------------------------------------

    def _shell(self, command: str, cwd: Optional[str], timeout: float, env: Optional[dict]):
        # Same rule as SandboxConfig.env: credentials never cross into the container.
        assert_no_secrets(env)
        target = self._resolve(cwd) if cwd else self.workdir

        full_command = command
        if env:
            prefix = " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
            full_command = f"env {prefix} {command}"

        result = self._exec(full_command, target, timeout)

        # A non-zero exit is a normal observation for the agent, not a tool failure:
        # the tool did its job. Only a timeout or a broken sandbox fails the call.
        output = {
            "command": command,
            "exit_code": result.exit_code,
            "stdout": truncate_output(result.stdout),
            "stderr": truncate_output(result.stderr),
            "success": result.success,
        }
        return output, self._meta(cwd=target, command=command, exit_code=result.exit_code)

    def _exec(self, command: str, cwd: str, timeout: float):
        self._require_live()
        result = self.sandbox.run_command(command, cwd=cwd, timeout=timeout)
        if result.timed_out:
            # Sandbox._run tears the container down on timeout, since a killed
            # `docker exec` leaves the process inside still running.
            raise ToolTimeoutError(
                f"command timed out after {timeout}s (the sandbox was torn down): {command}"
            )
        return result

    # ---- shared helpers ----------------------------------------------------------

    def _resolve(self, path: str) -> str:
        """Make `path` absolute against the workdir and refuse anything outside root."""
        if not isinstance(path, str) or not path:
            raise ValueError("path must be a non-empty string")
        absolute = path if posixpath.isabs(path) else posixpath.join(self.workdir, path)
        normalized = posixpath.normpath(absolute)
        if normalized != self.root and not normalized.startswith(self.root + "/"):
            raise PermissionError(f"path escapes the sandbox task root {self.root}: {path}")
        return normalized

    def _require_live(self) -> None:
        if self.sandbox.container_id is None:
            raise SandboxToolError(
                f"sandbox {self.sandbox.task_id} is not running (state={self.sandbox.state.value}); "
                "create it before calling tools"
            )

    def _meta(self, **fields) -> dict:
        # task_id ties every tool_call event back to the sandbox_run events it caused.
        return {"task_id": self.sandbox.task_id, "sandboxed": True, **fields}


def toolset_for(sandbox: Sandbox, **kwargs) -> SandboxToolset:
    """Convenience constructor mirroring how a worker would build one per task."""
    return SandboxToolset(sandbox, **kwargs)
