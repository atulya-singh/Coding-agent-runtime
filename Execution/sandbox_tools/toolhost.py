"""Runs INSIDE the sandbox container. Never imported by host code at runtime.

The host copies this single file into the container and invokes it as:

    python3 <path>            # JSON request on stdin, JSON response on stdout

Constraints that shape this file:
  * stdlib only, and no imports from this project -- the container has nothing
    installed but a Python interpreter.
  * single file -- it is delivered with one `docker cp`.
  * stdout carries the response and nothing else; diagnostics go to stderr.
  * file ops are implemented here rather than shelled out to cat/sed/grep so that
    behaviour does not vary with whatever userland a task's image happens to ship
    (python:3.11-slim, for instance, has no git and no ripgrep).

Response envelope, always exit code 0:
    {"ok": true,  "output": <any>, "metadata": {...}}
    {"ok": false, "error": "...", "error_type": "FileNotFoundError"}
A non-zero exit or unparseable stdout means the host's protocol assumption broke
(interpreter missing, file truncated) rather than a tool-level failure.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import sys

PROTOCOL_VERSION = 1

MAX_READ_BYTES = 5 * 1024 * 1024
MAX_SEARCH_FILE_BYTES = 2 * 1024 * 1024
MAX_LINE_CHARS = 500

SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".tox", "node_modules", ".venv", "venv", "dist", "build",
        ".eggs", ".idea", ".vscode",
    }
)


class ToolHostError(Exception):
    """Carries an explicit error_type so the host can re-raise a matching exception."""

    def __init__(self, message: str, error_type: str = "ToolExecutionError"):
        super().__init__(message)
        self.error_type = error_type


# ---- path policy ---------------------------------------------------------------


def resolve_path(path: str, root: str, workdir: str) -> str:
    """Resolve `path` to an absolute real path, refusing anything outside `root`.

    Relative paths resolve against `workdir`. Symlinks are followed before the
    check, so a symlink planted inside the workspace cannot be used to reach the
    container's system files. (Only in-container files are at stake either way --
    the host filesystem is not mounted -- but keeping the agent inside /task means
    a confused agent corrupts its own workspace instead of its interpreter.)
    """
    if not isinstance(path, str) or not path:
        raise ToolHostError("path must be a non-empty string", "ValueError")

    candidate = path if os.path.isabs(path) else os.path.join(workdir, path)
    candidate = os.path.normpath(candidate)

    # Symlinks only exist on components that exist, so resolve the deepest existing
    # ancestor and re-attach the (necessarily symlink-free) remainder.
    probe = candidate
    while probe not in ("/", "") and not os.path.lexists(probe):
        probe = os.path.dirname(probe)
    real = os.path.realpath(probe)
    if probe != candidate:
        real = os.path.normpath(os.path.join(real, os.path.relpath(candidate, probe)))

    root_real = os.path.realpath(root)
    if real != root_real and not real.startswith(root_real + os.sep):
        raise ToolHostError(
            f"path escapes the sandbox task root {root_real}: {path}", "PermissionError"
        )
    return real


# ---- operations ------------------------------------------------------------------


def op_ping(args: dict, root: str, workdir: str) -> tuple:
    return (
        {"protocol_version": PROTOCOL_VERSION, "python": sys.version.split()[0]},
        {"root": root, "workdir": workdir},
    )


def op_read_file(args: dict, root: str, workdir: str) -> tuple:
    target = resolve_path(args["path"], root, workdir)
    if not os.path.exists(target):
        raise ToolHostError(f"No such file: {args['path']}", "FileNotFoundError")
    if os.path.isdir(target):
        raise ToolHostError(f"Not a file: {args['path']}", "IsADirectoryError")
    size = os.path.getsize(target)
    if size > MAX_READ_BYTES:
        raise ToolHostError(
            f"File too large to read ({size} bytes > {MAX_READ_BYTES}): {args['path']}",
            "ValueError",
        )
    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read()
    return content, {"path": target, "size_bytes": len(content.encode("utf-8"))}


def op_write_file(args: dict, root: str, workdir: str) -> tuple:
    target = resolve_path(args["path"], root, workdir)
    content = args.get("content", "")
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content)
    return (
        {"path": target, "bytes_written": len(content.encode("utf-8"))},
        {"path": target},
    )


def op_edit_file(args: dict, root: str, workdir: str) -> tuple:
    target = resolve_path(args["path"], root, workdir)
    old_string = args["old_string"]
    new_string = args["new_string"]
    replace_all = bool(args.get("replace_all", False))

    if not os.path.exists(target):
        raise ToolHostError(f"No such file: {args['path']}", "FileNotFoundError")
    if not old_string:
        raise ToolHostError("old_string must not be empty", "ValueError")

    with open(target, "r", encoding="utf-8", errors="replace") as handle:
        original = handle.read()

    occurrences = original.count(old_string)
    if occurrences == 0:
        raise ToolHostError(f"old_string not found in {args['path']}", "ValueError")
    # Same uniqueness rule as the host-side tool: an under-specified old_string
    # would otherwise silently edit the wrong occurrence.
    if occurrences > 1 and not replace_all:
        raise ToolHostError(
            f"old_string is not unique in {args['path']} ({occurrences} matches); "
            "add more surrounding context or set replace_all=True",
            "ValueError",
        )

    count = occurrences if replace_all else 1
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(original.replace(old_string, new_string, count))
    return (
        {"path": target, "occurrences_replaced": count},
        {"path": target},
    )


def _match_pattern(rel_path: str, name: str, file_pattern: str) -> bool:
    return fnmatch.fnmatch(name, file_pattern) or fnmatch.fnmatch(rel_path, file_pattern)


def op_search_code(args: dict, root: str, workdir: str) -> tuple:
    base = resolve_path(args.get("path") or ".", root, workdir)
    if not os.path.exists(base):
        raise ToolHostError(f"Search path does not exist: {args.get('path')}", "FileNotFoundError")

    query = args["query"]
    file_pattern = args.get("file_pattern")
    max_results = int(args.get("max_results", 200))
    flags = 0 if args.get("case_sensitive") else re.IGNORECASE
    try:
        pattern = re.compile(query, flags)
    except re.error as exc:
        raise ToolHostError(f"invalid search regex {query!r}: {exc}", "ValueError") from exc

    matches: list = []
    truncated = False

    def report(path: str) -> str:
        # Paths are reported relative to the workdir so they can be handed straight
        # back to read_file/edit_file without translation.
        rel = os.path.relpath(path, workdir)
        return path if rel.startswith("..") else rel

    def scan(file_path: str) -> bool:
        try:
            if os.path.getsize(file_path) > MAX_SEARCH_FILE_BYTES:
                return False
            with open(file_path, "rb") as raw:
                if b"\0" in raw.read(8192):  # binary
                    return False
            with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if pattern.search(line):
                        text = line.rstrip("\n")
                        if len(text) > MAX_LINE_CHARS:
                            text = text[:MAX_LINE_CHARS] + "...[line truncated]"
                        matches.append(
                            {"file": report(file_path), "line_number": line_number, "line": text}
                        )
                        if len(matches) >= max_results:
                            return True
        except OSError:
            return False  # unreadable file: skip rather than abort the whole search
        return False

    if os.path.isfile(base):
        truncated = scan(base)
    else:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                file_path = os.path.join(dirpath, name)
                if file_pattern and not _match_pattern(
                    os.path.relpath(file_path, base), name, file_pattern
                ):
                    continue
                if scan(file_path):
                    truncated = True
                    break
            if truncated:
                break

    return matches, {
        "query": query,
        "path": base,
        "match_count": len(matches),
        "truncated": truncated,
    }


OPS = {
    "ping": op_ping,
    "read_file": op_read_file,
    "write_file": op_write_file,
    "edit_file": op_edit_file,
    "search_code": op_search_code,
}


def handle(request: dict) -> dict:
    op_name = request.get("op")
    handler = OPS.get(op_name)
    if handler is None:
        return {
            "ok": False,
            "error": f"unknown op: {op_name!r} (known: {', '.join(sorted(OPS))})",
            "error_type": "ValueError",
        }

    root = request.get("root") or "/task"
    workdir = request.get("workdir") or root
    args = request.get("args") or {}

    try:
        output, metadata = handler(args, root, workdir)
    except ToolHostError as exc:
        return {"ok": False, "error": str(exc), "error_type": exc.error_type}
    except KeyError as exc:
        return {"ok": False, "error": f"missing required argument: {exc}", "error_type": "ValueError"}
    except Exception as exc:  # never let a crash look like a protocol failure
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__}
    return {"ok": True, "output": output, "metadata": metadata}


def main() -> int:
    raw = sys.stdin.read()
    try:
        request = json.loads(raw)
    except ValueError as exc:
        print(f"toolhost: could not parse request: {exc}", file=sys.stderr)
        return 3
    sys.stdout.write(json.dumps(handle(request)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
