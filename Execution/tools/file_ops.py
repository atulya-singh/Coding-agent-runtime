from __future__ import annotations

from pathlib import Path

from .base import run_with_timeout, tool

MAX_READ_BYTES = 5 * 1024 * 1024


def _read(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Not a file: {path}")
    if file_path.stat().st_size > MAX_READ_BYTES:
        raise ValueError(f"File too large to read: {path}")
    return file_path.read_text(encoding="utf-8", errors="replace")


@tool("read_file")
def read_file(path: str, timeout: float = 10.0):
    content = run_with_timeout(_read, args=(path,), timeout=timeout)
    return content, {"path": path, "size_bytes": len(content.encode("utf-8"))}


def _write(path: str, content: str) -> dict:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return {"path": path, "bytes_written": len(content.encode("utf-8"))}


@tool("write_file")
def write_file(path: str, content: str, timeout: float = 10.0):
    result = run_with_timeout(_write, args=(path, content), timeout=timeout)
    return result, {"path": path}


def _edit(path: str, old_string: str, new_string: str, replace_all: bool) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    original = file_path.read_text(encoding="utf-8", errors="replace")
    occurrences = original.count(old_string)

    if occurrences == 0:
        raise ValueError(f"old_string not found in {path}")
    # Require uniqueness unless replace_all: prevents editing the wrong occurrence
    # when an agent's old_string under-specifies the intended location.
    if occurrences > 1 and not replace_all:
        raise ValueError(
            f"old_string is not unique in {path} ({occurrences} matches); "
            "add more surrounding context or set replace_all=True"
        )

    count = occurrences if replace_all else 1
    updated = original.replace(old_string, new_string, count)
    file_path.write_text(updated, encoding="utf-8")
    return {"path": path, "occurrences_replaced": count}


@tool("edit_file")
def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False, timeout: float = 10.0):
    result = run_with_timeout(_edit, args=(path, old_string, new_string, replace_all), timeout=timeout)
    return result, {"path": path}
