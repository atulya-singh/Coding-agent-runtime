# Coding-agent-runtime

Building out this project in different stages:

1. Agent Execution : Provide the agent with capabilites to read, write, edit files. Also provide capabilites to run tests, git diff, and search up specific code.
2. Sandbox : Disposable Docker container per task, with CPU/memory/pid/timeout limits, network off by default, and no credentials ever passed inside.
3. Task dataset : Tasks mined from real upstream commits across all seven categories, each verified in a sandbox to be unsolved before the fix and solved after it.
4. Objective evaluation : Grade an agent by replaying its patch onto a clean checkout and running build, tests, benchmarks, mutations and static checks -- never by asking a model whether the answer looks right.

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

## Objective evaluation

`Evaluation` grades an agent's work. Nothing it decides comes from a model's
opinion: a verdict is tests, a build, a measured benchmark, killed mutations and
static checks.

```python
from Evaluation import Evaluator, collect_patch

patch  = collect_patch(sandbox, base_dir)          # what the agent changed
result = Evaluator(task, task_dir).evaluate(patch)

result.outcome                # solved | partial | failed | build_failed | patch_failed | infrastructure_error
result.hidden_test_pass_rate  # 0.0-1.0
result.to_dict()              # the full reproducible record
```

The agent's own container is never graded -- it holds whatever the agent
installed, deleted or cached. The patch is replayed onto a pristine export of the
task's base commit in a fresh container, so the verdict depends on the diff and
nothing else:

```
apply patch -> setup -> build -> public tests -> [restore graded files]
            -> hidden tests -> regression tests -> benchmark -> mutations
            -> static checks -> result
```

Three rules the pipeline holds to:

- **Hidden tests are restored, not trusted.** Files the task lists are copied over
  the agent's copies before grading, so editing a test is not a way to pass. For
  `testing` tasks, where writing tests *is* the job, grading is by mutation and
  the agent's file is left alone.
- **Gates are separated from achievements.** Hidden tests, a benchmark threshold
  and killed mutations say the task was solved; the patch applying, the build
  working and the regression suite still passing only say nothing was wrecked.
  Only achievements earn partial credit -- otherwise an agent that changed
  nothing would score 50% on a refactor task because tests it never touched still
  pass.
- **Partial credit starts at zero.** A hidden suite also covers the cases around
  the change, so it is normal for most of it to pass before the agent starts.
  Credit is for closing the gap from that baseline to 1.0, not for the raw rate;
  each task records its `baseline_hidden_pass_rate`, and `python -m Tasks.verify`
  re-measures it so it cannot drift. The Hidden Test Pass Rate metric still
  reports the raw counts.
- **A broken runtime is not a failed agent.** Docker dying or a registry outage
  produces `infrastructure_error`, which is reported separately and excluded from
  the capability metrics rather than counted against the model (Principle 5).

Static checks run on the patch rather than in the container: unresolved conflict
markers, interactive debuggers left in added lines, and any write outside the
repository. They are facts about the diff, need no linter installed, and point at
a file and a line, which an exit code never does.

Grade a patch, or bracket the pipeline with the two patches whose correct grade is
already known:

```
python -m Evaluation <task_id> --patch fix.diff
python -m Evaluation --all --reference --json results.json   # the real upstream fix
python -m Evaluation --all --empty                           # an agent that did nothing
```

Check the grader itself before trusting a number out of it:

```
python -m Evaluation.selftest             # offline checks, then every task
python -m Evaluation.selftest --offline   # no Docker, under a second
```

The reference patch must grade `solved` on every task and the empty patch must
grade unsolved with no partial credit. If the first breaks, no agent could ever
pass; if the second breaks, every number the pipeline reports is inflated.
