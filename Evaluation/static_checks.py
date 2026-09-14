"""Static checks, run on the patch rather than on the container.

plan.md puts a static-check stage in the pipeline but does not say what it should
check. The useful answer for a coding agent is: things that are wrong on the face
of the diff, regardless of whether the tests happen to pass. Those checks belong
on the patch itself -- they are deterministic, need no container time, and their
findings point at a file and a line, which a test exit code never does.

Every check here is a fact about the text, not a judgement about style. There is
no linter to install (which would need network in the sandbox), no configuration
to drift, and no opinion about formatting -- a "solution" is not marked down for
looking unusual, only for containing something that is objectively debris.

Tasks can add their own in-container commands via `static_checks:` in task.yaml;
those run alongside these and any non-zero exit fails the stage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

ERROR = "error"
WARNING = "warning"

#: Both halves of a conflict marker. "=======" is deliberately not matched: it is
#: a legitimate line in reStructuredText, Markdown and plenty of banners.
_CONFLICT = re.compile(r"^(?:<{7}|>{7})(?:\s|$)")

#: Interactive debuggers left in a patch. Matched as statements -- an argument
#: list or a line end must follow -- so a docstring mentioning breakpoints, or a
#: variable named `breakpoints`, does not trip it.
_DEBUGGER = re.compile(
    r"(?:^|[\s;:])(?:"
    r"breakpoint\s*\(|"
    r"(?:pdb|ipdb|pudb)\s*\.\s*set_trace\s*\(|"
    r"import\s+(?:pdb|ipdb|pudb)\b"
    r")"
)

_FILE_HEADER = re.compile(r"^\+\+\+ (?:b/)?(.+?)(?:\t.*)?$")
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    path: str = ""
    line: int = 0

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "path": self.path,
            "line": self.line,
        }

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.path else "patch"
        return f"[{self.severity}] {where}: {self.message}"


def added_lines(patch_text: str) -> Iterator[Tuple[str, int, str]]:
    """Yield (path, line number in the new file, text) for every added line.

    Only added lines are inspected. Flagging a conflict marker or a stray
    debugger that was already in the repository would blame the agent for the
    task's own starting state.
    """
    path = ""
    line_number = 0
    in_hunk = False

    for raw in patch_text.splitlines():
        header = _FILE_HEADER.match(raw)
        if header and not raw.startswith("+++ +"):
            path = header.group(1)
            if path == "/dev/null":
                path = ""
            in_hunk = False
            continue

        hunk = _HUNK_HEADER.match(raw)
        if hunk:
            line_number = int(hunk.group(1))
            in_hunk = True
            continue

        if not in_hunk or not path:
            continue

        if raw.startswith("+"):
            yield path, line_number, raw[1:]
            line_number += 1
        elif raw.startswith("-") or raw.startswith("\\"):
            # A deletion consumes no line in the new file; "\ No newline at end
            # of file" is a note, not content.
            continue
        elif raw.startswith(" "):
            line_number += 1
        else:
            # Anything else (a new `diff --git`, binary-patch payload) ends the hunk.
            in_hunk = False


def inspect_patch(patch, protected_paths: Sequence[str] = ()) -> List[Finding]:
    """Run every patch-level check. `protected_paths` are files that grading will
    overwrite, so edits to them cannot change the verdict."""
    from .patch import unsafe_paths  # local import: patch.py imports nothing from here

    findings: List[Finding] = []

    for path in unsafe_paths(patch):
        findings.append(
            Finding(
                "unsafe_path",
                ERROR,
                "patch writes outside the repository tree",
                path=path,
            )
        )

    if patch.is_empty:
        findings.append(
            Finding("empty_patch", WARNING, "the agent produced no change at all")
        )

    protected = set(protected_paths)
    for path in patch.files:
        if path in protected:
            findings.append(
                Finding(
                    "modified_graded_test_file",
                    WARNING,
                    "grading restores this file from the task definition, so the "
                    "edit has no effect on the result",
                    path=path,
                )
            )

    for path, line_number, text in added_lines(patch.text):
        if _CONFLICT.match(text):
            findings.append(
                Finding(
                    "conflict_marker",
                    ERROR,
                    "unresolved merge-conflict marker in an added line",
                    path=path,
                    line=line_number,
                )
            )
        if _DEBUGGER.search(text):
            findings.append(
                Finding(
                    "debugger_statement",
                    ERROR,
                    f"interactive debugger left in the change: {text.strip()[:80]}",
                    path=path,
                    line=line_number,
                )
            )

    return findings


def errors(findings: Iterable[Finding]) -> List[Finding]:
    return [finding for finding in findings if finding.severity == ERROR]
