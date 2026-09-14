# Coding-agent-runtime

Building out this project in different stages:

1. Agent Execution : Provide the agent with capabilites to read, write, edit files. Also provide capabilites to run tests, git diff, and search up specific code.
2. Sandbox : Disposable Docker container per task, with CPU/memory/pid/timeout limits, network off by default, and no credentials ever passed inside.
3. Task dataset : Tasks mined from real upstream commits across all seven categories, each verified in a sandbox to be unsolved before the fix and solved after it.

## Running tools in the sandbox

`Execution.tools` runs on the host and is for harness code only. Everything the
agent drives goes through `SandboxToolset`, which offers the same seven tools with
the same result and logging contract, executed inside the container:

```python
from Sandbox import Sandbox, SandboxConfig
from Execution.sandbox_tools import SandboxToolset

with Sandbox(task_id="demo", config=SandboxConfig()) as sandbox:
    sandbox.initialize("/path/to/repo")
    tools = SandboxToolset(sandbox)

    tools.read_file("src/app.py")                      # -> ToolResult
    tools.edit_file("src/app.py", "old", "new")
    tools.run_tests("pytest -q")
    tools.call("search_code", {"query": r"def \w+"})   # dispatch by name
    tools.tool_specs()                                 # schemas for a model API
```

Verify the whole toolset against a real container:

```
python -m Execution.sandbox_tools            # 32 checks, tears the container down
python -m Execution.sandbox_tools --no-git   # skip the git install + git_diff check
```

## The task dataset

Nine tasks, all mined from real psf/requests commits, covering all seven
categories. Each task pins `commit` to the *parent* of the real fix, so the
agent starts from the genuine pre-change state.

| category | task | graded by |
| --- | --- | --- |
| bug_fix | 001 netrc empty default, 002 content-type param, 003 JSONDecodeError pickle | hidden tests |
| feature | 004 header name validation | hidden tests |
| refactor | 005 resolve_proxies extraction | hidden (structural + behavioral) + regression |
| testing | 006 LookupDict attribute tests | mutation testing |
| performance | 007 proxy bypass short-circuit | benchmark ratio + regression |
| api_change | 008 pool key attributes hook | hidden tests + regression |
| dependency_migration | 009 chardet to charset_normalizer | hidden tests |

A category is not a label: it selects the grading contract, and
`CATEGORY_CONTRACTS` in `Tasks/schema.py` refuses a task that doesn't declare
one. A `performance` task cannot ship without a benchmark, a `testing` task
cannot ship without mutations, and neither a `refactor` nor an `api_change` can
ship without the regression command that pins what must not change.

Check that every task file is well-formed:

```
python -m Tasks
```

Check the thing that actually matters -- that each task is real and solvable, and
that its grading criterion separates the pre-change state from the upstream fix:

```
python -m Tasks.verify            # every task, in a real sandbox
python -m Tasks.verify <task_id>  # just one
```

For each task this runs the base commit in a container, proves the criterion is
*not* already satisfied, applies the real upstream fix file by file, and proves
it then is. For `performance` that means measuring the benchmark on both sides
of the threshold; for `testing`, proving every mutation survives the old tests
and dies to the new ones.
