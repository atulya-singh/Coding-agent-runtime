# Coding-agent-runtime

Building out this project in different stages:

1. Agent Execution : Provide the agent with capabilites to read, write, edit files. Also provide capabilites to run tests, git diff, and search up specific code.
2. Sandbox : Disposable Docker container per task, with CPU/memory/pid/timeout limits, network off by default, and no credentials ever passed inside.
3. Task dataset : Tasks mined from real upstream commits across all seven categories, each verified in a sandbox to be unsolved before the fix and solved after it.
4. Objective evaluation : Grade an agent by replaying its patch onto a clean checkout and running build, tests, benchmarks, mutations and static checks -- never by asking a model whether the answer looks right.
5. Agent loop : Drive a real model through the sandboxed tools -- generate, execute, feed the result back -- until it reports done or hits a turn, token or cost budget.

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

## The agent loop

`Agent` is the part that actually calls a model. One turn is: send the
conversation, run whatever tools the model asked for inside the sandbox, send
the results back.

```
generate -> tool_use blocks -> execute in the sandbox -> tool_result blocks -> generate
```

```python
from Agent import run_task, RunConfig

run = run_task(task, task_dir, config=RunConfig(grade=True))

run.agent.stop_reason   # finished | finished_implicit | max_turns | budget |
                        # max_output_tokens | model_error | refusal | sandbox_gone
run.agent.tool_calls    # structured history, one entry per dispatched tool
run.agent.usage         # cumulative tokens; .cost_usd when the model has configured rates
run.patch               # what the agent changed, as a diff against the base commit
run.evaluation.outcome  # the Phase 4 grade, from a separate clean container
```

`run_task` assembles what a real attempt needs around the loop: a pristine
export of the base commit, a container, the task's setup steps, the toolhost,
and the diff at the end. Each of those is a different kind of failure, and the
record keeps them apart -- a container that would not build is not an agent that
could not solve the task.

From the command line:

```
python -m Agent requests-001-netrc-empty-default --grade
python -m Agent requests-001-netrc-empty-default --dry-run   # spends nothing
```

Defaults -- model, thinking, effort, budgets, token rates -- come from
`Agent/config.yaml`, so what a run was configured with is a file you can read
rather than an argument someone remembered to pass. The API key is fetched
host-side through `SecretBroker` and never enters a container; a free
token-count call checks it, and the model id, before any container is built.

There is no separate planning call and no separate verification call, on
purpose. Planning-vs-direct-execution is one of the experiments this project
exists to measure, so building it in would foreclose the comparison; and asking
a model to judge whether a tool did what it asked adds an opinion where there is
already a fact -- the tool's own result, and the tests the model can run itself.
Principle 1 applies inside the loop, not only at grading time.

Every way a run can end is a named reason rather than a bare success flag. "The
model said it was done", "it ran out of turns" and "the API returned 503" are
three different events, and Principle 5 turns on not averaging them together.

Three details that only look like details:

- **`finish` is a tool the loop answers itself.** The model needs a way to say
  "done" that is distinguishable from having nothing more to say. It is never
  dispatched to the sandbox, and a toolset that defines its own `finish` is
  rejected rather than silently shadowed.
- **Cost rates are configuration, not code.** A stale hardcoded price would
  quietly corrupt every cost number the project reports, so a model with no
  configured rates reports an unknown cost rather than zero, and a cost budget
  without rates is rejected as unenforceable. The rates shipped in
  `Agent/config.yaml` are the published first-party ones, dated in the file.
- **Three endings are not the agent's doing and are not counted as its
  failures.** The API failing, the container tearing itself down after a command
  timed out, and a safety classifier declining the request each stop the run
  under their own name. Without that, a run that hit a 503 on turn one and a run
  that tried for forty turns and failed would look identical in the results.

Check the loop without spending a token -- every exit path, budget and
malformed turn, against a scripted model and no container:

```
python -m Agent.selftest --offline
```

Without `--offline` it then runs the whole thing for real -- container, setup,
toolhost, tool calls, diff, grade -- with a scripted agent replaying a task's
known upstream fix. That still needs no API key: if it does not grade as solved,
the wiring is wrong, because the patch is correct by construction.

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
