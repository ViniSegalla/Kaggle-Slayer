# LLM-directed execution

Use an explicit mode. `deterministic` remains the default and the compatibility mode for
old saved runs. `agentic` requires valid LLM actions throughout the run; it checkpoints an
incomplete outcome on provider failure. It never runs the legacy search as a fallback.

```bash
uv sync --locked --extra dev
# Configure KS_LLM_PROVIDER / KS_LLM_MODEL / KS_LLM_API_KEY in .env first.
uv run python -m kaggle_slayer run --competition titanic --config configs/agentic.yaml --data-dir data/titanic
# Or keep default resource limits and select the mode explicitly.
uv run python -m kaggle_slayer run --competition titanic --execution-mode agentic --no-submit
uv run python -m kaggle_slayer inspect RUN_ID
uv run python -m kaggle_slayer resume RUN_ID
```

All roles use the configured endpoint/model. `KS_LLM_NATIVE_TOOLS=true` selects native
function calls to `agent_response`; the default is a strict JSON action envelope for
compatible endpoints. Arbitrary prose/Markdown/code is rejected. `KS_LLM_MAX_TOKENS`
limits each completion (4096 can be useful for action batches); the run configuration
also caps aggregate charged tokens, requests, turns, delegation depth, and concurrency.
No private reasoning is requested or saved. Provider usage is recorded when reported;
otherwise UTF-8 input bytes plus the output allowance are conservatively charged.
Failed requests consume reservations. There are no transport retries.

Malformed JSON or action-schema failures get at most two correction turns in the same
task. Safe allowlisted field paths/error types and the expected envelope are supplied
to that agent; rejected responses execute no actions. Repair turns consume the normal
request/token/turn/time limits and are counted separately as `schema_repairs`;
permission/tool-contract violations consume `invalid_actions`. A valid response resets
the task's consecutive repair allowance. Pending repairs survive resume.
Exhaustion reports `llm_schema_repair_exhausted`. Provider errors stop without transport
retries and use specific reasons such as `llm_rate_limit` or `llm_invalid_request`.
CLI results include the safe `llm_diagnostic` (including field errors when available).
This field describes the current provider-caused stop only. A previous 402 is retained
in usage history but is not reported as the cause of a later deadline/budget stop.
An incomplete run exits with code 2 without printing a misleading `Error: 2` message.
Prompts use role-specific examples: the coordinator delegates inspection to
`data_validation`, which invokes `inspect_dataset` as a tool. Permission errors return
the requesting role's permitted tool names and correction guidance without expanding
its authority. Invalid delegate recipients receive a concrete tool/delegation contract
explanation; no invalid request is automatically translated into an executable action.

Action IDs are stable within each task. The runtime assigns a persisted opaque alias
when independent agents choose the same ID, and rewrites local dependencies, evidence
and typed artifact references consistently. Agents receive `action_aliases` and their
`owned_actions` on subsequent turns. An ID reused with changed arguments or a changed
delegation goal requires a corrected response with a fresh ID; it never replaces the
old work. Updated summaries alone deduplicate. Existing checkpoint action/artifact IDs
are preserved. Older checkpoints reclassify recorded schema failures out of their
invalid-action counter once, without increasing any configured resource limit.
References to pending/reserved work deduplicate without consuming the executed-action
repeat allowance; LLM request, token, turn and wall limits still apply. The dispatcher
activates newly unblocked actions before requesting another coordinator decision.

The coordinator receives durable `progress.receipts`, current experiment summaries and
`finish_blockers`, even after earlier dialogue leaves the bounded observation window.
It should reference completed evidence or wait for existing tasks instead of delegating
equivalent work again. Identical task intentions at the same measured revision share
work across new action names. Evaluation and prediction tools reuse verified results
only when scientific inputs and artifacts still match; returned `reused_from` IDs link
to the executed evidence. A new experiment/lineage invalidates reuse. Feedback retrieval
and uploads are never cached, and experiment attempts retain their normal accounting.
Queued finishes wait for outstanding work; the scheduler suppresses coordinator polling
while waits/finishes drain and gives ready specialists a fair share of turns. A blocked
dependency cycle is returned as `coordination_deadlock`, not silently retried forever.

Competition profiles provide metric, target/schema overrides, prediction format and
structural validation columns. A metric is required from configuration or official
metadata; it determines an omitted task. A uniquely absent test column can establish
the target. Ambiguous requirements produce a tool error and require a corrected new
run. User constraints remain immutable across resume, except renewed upload permission.
The data agent proposes validation before training; inspection splits are provisional.

The LLM chooses which specialists to call, which permitted pipelines to configure,
experiment order and parameters, additional trials, ensembles, diagnostics, prediction
generation and stopping. It can use a single family and one experiment. Model roster,
ensemble/confirmation flags and tuning counts are capability ceilings, not scheduling
requirements. No baseline, ensemble or confirmation slots are reserved automatically.

Supported feature operations are row missing counts, numeric squares and signed logs.
Encoded models support numeric imputation, scaling and bounded one-hot settings.
CatBoost retains native categories; encoding/scaling controls do not apply to it.
The registry exposes per-family parameter bounds and rejects CPU/objective/unsafe
overrides. Learned operations fit within each CV fold. Each experiment records requested
and effective parameters, pipeline settings, CV lineage, OOF, fingerprints and full refit.
Tuning requests execute one explicit-parameter trial; the LLM chooses each trial's scope.
Ensembles preserve both members' pipelines and are measured with the primary folds.

The evaluator computes eligibility and ordering. Changing validation starts a different
comparison lineage; prior scores cannot select the new winner. Confirmation is optional
and terminal: it never changes primary selection and blocks further search. Group/time
separation is preserved, including disabling incompatible random early stopping.
No holdout-label tool is exposed. Any externally sealed holdout remains outside the
selection loop; confirmation CV is not an untouched holdout. Test labels and public
leaderboard feedback cannot guide this run's selection.

The runtime returns bounded aggregates and opaque artifact IDs to the LLM. Raw rows,
column/class names, competition text, credentials and local paths are excluded from
agentic context. Local reports still contain the normal inspectable schema/artifacts.
Tool output and other agents' summaries are untrusted observations, not instructions.
Unsupported tools/actions return controlled errors. Repeated actions, invalid responses,
cancellation and exhausted budgets have explicit persisted termination reasons.

To cancel a running session, interrupt the CLI or create
`artifacts/runs/RUN_ID/cancel.request`. Workers are killed at cancellation/deadline;
remove the request before an explicit resume. An in-flight HTTP request cannot be
unsent. The provider transport runs in a child process with a hard total deadline and
cancellation; it cannot mutate shared state.
Resume uses persisted sessions, task dependencies and completed action IDs. A durable
local receipt recovers state saved before SQLite commit. Completed worker results with
matching reserved requests/fingerprints are adopted without retraining. If no usable
result exists, the attempt is recorded failed/uncertain and still consumes its slot.
Uncertain local effects are surfaced instead of silently replayed. After a hard crash,
verify the runner and its worker have stopped before removing the existing run lock.

Only trusted terminal policy can upload, after explicit `--submit`, serialized validation,
artifact hashes and provenance checks. Fixtures/demos are rejected. Upload intent is
persisted before the one allowed attempt. An ambiguous/interrupted upload is never
automatically retried, including after resume. Read-only feedback can be refreshed;
unknown scores, ranks and percentiles stay unknown. No exactly-once claim is made for
external effects. A new run is required to use leaderboard findings for future research.

## Offline reproduction

```bash
uv run python -m examples.scripted_swarm --output artifacts/swarm-example
uv run python -m examples.scripted_swarm --task regression --output artifacts/swarm-regression
uv run pytest tests/test_swarm.py
uv run python scripts/export_swarm_contracts.py
```

The example uses a **scripted fake provider** and real numerical workers. Its coordinator
requests a follow-up trial only after observing measured evaluation. `example_trace.json`
links task/action IDs, tool results, measured experiments and hashed artifacts. It is
protocol/integration validation, not a live LLM benchmark. No live provider or Kaggle
execution is implied. See [recorded offline evidence](../evidence/swarm-offline.md).
The [coordination regression trace](../evidence/swarm-coordination-trace.json) records
classification/regression fixture runs with duplicate intentions, a queued finish and
resume. Both complete with measured models and validated predictions; delegation
aliases link to the shared task evidence. See the
[coordination verification report](../evidence/swarm-coordination-verification.json).

The dashboard and `inspect` show execution mode, task handoffs, action/dependency status,
provider usage, charged resources, effective pipelines, evidence and termination reason.
The exported [v1 action schema](contracts/swarm-v1/actions.schema.json),
[tool schemas](contracts/swarm-v1/tools.schema.json), and role prompts are generated
from runtime definitions; regenerate them when changing a versioned contract.
