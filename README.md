# Coding-agent-runtime

Building out this project in different stages:

1. Agent Execution : Provide the agent with capabilites to read, write, edit files. Also provide capabilites to run tests, git diff, and search up specific code.
2. Sandbox : Disposable Docker container per task, with CPU/memory/pid/timeout limits, network off by default, and no credentials ever passed inside.
3. Task dataset : Tasks mined from real upstream bug-fix commits, each verified to fail before the fix and pass after it.

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

## Validate the task dataset

```
python -m Tasks
```
