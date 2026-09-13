"""Live end-to-end check: every tool, against a real container.

    python -m Execution.sandbox_tools [--keep-logs DIR] [--no-git]

Creates a throwaway sandbox with a small generated repository, exercises all seven
tools plus the path policy and the dispatcher, and always destroys the container.
Exits non-zero if any check fails, so it can be wired into CI later.
"""
from __future__ import annotations

import argparse
import logging
import sys

from Sandbox import Sandbox, SandboxConfig

from .toolset import SandboxToolset

FIXTURE_FILES = {
    "app.py": (
        "def greet(name):\n"
        '    return "hello " + name\n'
        "\n"
        "\n"
        "def farewell(name):\n"
        '    return "bye " + name\n'
    ),
    "test_app.py": (
        "from app import farewell, greet\n"
        "\n"
        "\n"
        "def test_greet():\n"
        '    assert greet("world") == "hello world"\n'
        "\n"
        "\n"
        "def test_farewell():\n"
        '    assert farewell("world") == "bye world"\n'
    ),
    "notes/readme.txt": "scratch notes\n",
}


class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list = []

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  PASS  {label}")
        else:
            self.failures.append(label)
            print(f"  FAIL  {label}{(' -- ' + detail) if detail else ''}")

    def report(self) -> int:
        total = self.passed + len(self.failures)
        print(f"\n{self.passed}/{total} checks passed")
        if self.failures:
            print("failed: " + ", ".join(self.failures))
            return 1
        print("all sandboxed tools verified")
        return 0


def build_fixture_repo(tools: SandboxToolset, checks: Checks) -> None:
    """Seed the workspace using write_file, which also exercises the tool itself."""
    for path, content in FIXTURE_FILES.items():
        result = tools.write_file(path, content)
        checks.check(f"write_file {path}", result.success, str(result.error))


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m Execution.sandbox_tools")
    parser.add_argument("--keep-logs", metavar="DIR", help="collect sandbox logs to DIR before teardown")
    parser.add_argument("--no-git", action="store_true", help="skip installing git and the git_diff check")
    parser.add_argument("--quiet", action="store_true", help="silence sandbox/tool JSON log lines")
    args = parser.parse_args()

    if args.quiet:
        for name in ("sandbox", "sandbox.docker", "agent.tools"):
            logging.getLogger(name).setLevel(logging.CRITICAL)

    checks = Checks()
    # Network is on because the smoke test installs pytest (and git) the way a real
    # task's setup step would; the tools themselves need no network.
    config = SandboxConfig(network_enabled=True, timeout_seconds=600)

    with Sandbox(task_id="toolset-smoke", config=config) as sandbox:
        tools = SandboxToolset(sandbox)

        print("\n[install]")
        info = tools.install()
        checks.check("toolhost installed", info.get("installed") is True, str(info))
        checks.check("container python reported", bool(info.get("python")), str(info))

        print("\n[write_file]")
        build_fixture_repo(tools, checks)

        print("\n[read_file]")
        read = tools.read_file("app.py")
        checks.check("read_file returns content", read.success and "def greet" in read.output, str(read.error))
        missing = tools.read_file("does_not_exist.py")
        checks.check(
            "read_file missing file fails cleanly",
            not missing.success and "No such file" in (missing.error or ""),
            str(missing.error),
        )

        print("\n[edit_file]")
        edited = tools.edit_file("app.py", '"hello " + name', 'f"hello {name}!"')
        checks.check("edit_file replaces once", edited.success and edited.output["occurrences_replaced"] == 1, str(edited.error))
        ambiguous = tools.edit_file("app.py", "name", "renamed")
        checks.check(
            "edit_file rejects ambiguous old_string",
            not ambiguous.success and "not unique" in (ambiguous.error or ""),
            str(ambiguous.error),
        )
        after = tools.read_file("app.py")
        checks.check("edit persisted in container", 'f"hello {name}!"' in after.output)

        print("\n[search_code]")
        found = tools.search_code(r"def \w+", file_pattern="*.py")
        checks.check("search_code finds defs", found.success and len(found.output) >= 4, str(found.error))
        checks.check(
            "search_code returns workdir-relative paths",
            all(not m["file"].startswith("/") for m in found.output),
            str(found.output[:2]),
        )
        filtered = tools.search_code("scratch", file_pattern="*.txt")
        checks.check("search_code honours file_pattern", filtered.success and len(filtered.output) == 1, str(filtered.output))
        none = tools.search_code("zzz_no_such_symbol")
        checks.check("search_code empty result is a success", none.success and none.output == [])

        print("\n[run_command]")
        ok = tools.run_command("echo sandboxed && pwd")
        checks.check("run_command succeeds", ok.success and "sandboxed" in ok.output["stdout"], str(ok.error))
        checks.check("run_command runs in the repo dir", "/task/repository" in ok.output["stdout"], ok.output["stdout"])
        failing = tools.run_command("exit 3")
        checks.check(
            "non-zero exit is reported, not raised",
            failing.success and failing.output["exit_code"] == 3 and failing.output["success"] is False,
            str(failing.output),
        )
        isolated = tools.run_command("ls /task")
        checks.check("host filesystem is not visible", "Users" not in isolated.output["stdout"], isolated.output["stdout"])

        print("\n[run_tests]")
        tools.run_command("pip install --no-input --quiet pytest==8.3.3", timeout=300)
        tests = tools.run_tests("python -m pytest -q", timeout=300)
        checks.check(
            "run_tests reports the failure the edit introduced",
            tests.success and tests.output["exit_code"] != 0 and "1 failed" in tests.output["stdout"],
            tests.output["stdout"][-300:],
        )
        tools.edit_file("app.py", 'f"hello {name}!"', '"hello " + name')
        fixed = tools.run_tests("python -m pytest -q", timeout=300)
        checks.check(
            "run_tests passes once reverted",
            fixed.success and fixed.output["exit_code"] == 0,
            fixed.output["stdout"][-300:],
        )

        print("\n[git_diff]")
        if args.no_git:
            print("  SKIP  git_diff (--no-git)")
        else:
            tools.run_command("apt-get update -qq && apt-get install -y -qq git", timeout=300)
            tools.run_command("git init -q . && git add -A && git -c user.email=a@b -c user.name=c commit -qm base", timeout=120)
            clean = tools.git_diff()
            checks.check("git_diff clean tree is empty", clean.success and clean.output == "", str(clean.error))
            tools.edit_file("app.py", '"bye " + name', '"goodbye " + name')
            diff = tools.git_diff()
            checks.check(
                "git_diff shows the tool's edit",
                diff.success and "goodbye" in diff.output and diff.metadata["has_changes"],
                str(diff.error),
            )

        print("\n[path policy]")
        for label, result in [
            ("absolute escape", tools.read_file("/etc/hostname")),
            ("traversal escape", tools.read_file("../../etc/hostname")),
        ]:
            checks.check(
                f"read_file blocks {label}",
                not result.success and "escapes the sandbox task root" in (result.error or ""),
                str(result.error),
            )
        tools.run_command("ln -sf /etc/hostname /task/repository/link_out")
        symlinked = tools.read_file("link_out")
        checks.check(
            "read_file blocks symlink escape",
            not symlinked.success and "escapes the sandbox task root" in (symlinked.error or ""),
            str(symlinked.error),
        )

        print("\n[dispatcher]")
        dispatched = tools.call("read_file", {"path": "app.py"})
        checks.check("call() dispatches by name", dispatched.success and "def greet" in dispatched.output)
        unknown = tools.call("rm_rf", {})
        checks.check("call() rejects unknown tools", not unknown.success and "unknown tool" in (unknown.error or ""))
        bad_args = tools.call("read_file", {"nope": 1})
        checks.check("call() rejects bad arguments", not bad_args.success, str(bad_args.error))
        checks.check("tool_specs covers every tool", {s["name"] for s in tools.tool_specs()} == set(tools.tools))

        print("\n[recovery]")
        tools.run_command(f"rm -f {'/opt/agent-runtime/toolhost.py'}")
        recovered = tools.read_file("app.py")
        checks.check("toolhost reinstalls after deletion", recovered.success, str(recovered.error))

        print("\n[result shape]")
        sample = tools.read_file("app.py")
        checks.check(
            "ToolResult fields intact",
            sample.tool == "read_file" and sample.duration_ms > 0 and sample.metadata["task_id"] == "toolset-smoke",
            str(sample.to_dict().keys()),
        )
        checks.check("results are JSON-serialisable", isinstance(sample.to_dict(), dict))

        if args.keep_logs:
            out = sandbox.collect_results(args.keep_logs)
            print(f"\nlogs collected to {out}")

    return checks.report()


if __name__ == "__main__":
    sys.exit(main())
