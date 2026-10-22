# Agents and communication

Agentic mode makes actual LLM calls for distinct coordinator, data/validation,
pipeline/modeling, evaluation and submission sessions. Each session has a persisted
task ID, scoped context and role tool allowlist. The coordinator can delegate only the
work needed, request independent concurrent work, revisit a specialist after new results,
and finish. Specialists return typed proposals/actions and evidence-backed summaries.
The scheduler alone writes shared state and reserves budgets; numerical workers return
measured artifacts. See [agentic contracts and usage](agentic-usage.md) and the versioned
prompts in `agents/swarm.py` / `docs/contracts/swarm-v1`.

The following table documents the retained **deterministic compatibility mode**.

| Agent | Inputs | Tools and output | Decision responsibility |
|---|---|---|---|
| Competition Planner | Profile, quality report, evaluator feedback, budgets | Model registry, optional structured LLM advice, Optuna ask/tell; emits `Plan` and `Candidate` queue | Baseline priorities and evidence-based tuning family |
| Data Engineer | Competition config, official API/local CSV directory | Metadata/download adapter, schema audit, feature/target identification, EDA, fixed split indices; emits `Dataset` and serializable report | Refuse ambiguous schemas, incompatible metrics and exact target copies |
| Modeler | Candidate, resolved competition/config, remaining budget | Isolated CV worker, fold-fit preprocessing, early stopping, full-data refit; emits `Experiment` | Train exactly the queued family/parameters; preserve errors and timings |
| Evaluator | Completed experiments, target metric, fitted checkpoints | Metric-direction ranking, train/CV gap checks, prediction generation, template validator; emits evaluation and submission reports | Reject weaker candidates, select a validated checkpoint, request tuning |
| Leaderboard Reviewer | Run-specific submission reference/description, actual API status | Official submission history; emits real public score if available, validation discrepancy and next-run recommendation | Keep unavailable rank/percentile unknown and avoid public-leaderboard optimization |

The deterministic mode is a real policy-driven team: each role uses different tools, consumes earlier outputs, and changes later actions. It does not pretend to be five independent language models. With an LLM configured, the planner receives structured strategy and priorities; the rest of the team still enforces measurable ML and submission contracts.

Every major action emits a `Decision` with `agent`, UTC timestamp, input context, selected action, concise rationale, invoked tool, summarized result, artifact paths, next action and severity. SQLite and JSONL record public decision summaries and tool traces, not hidden chain-of-thought. Queue entries link tuning trials to their parent experiment. Out-of-fold predictions retain the exact validation evidence used by the evaluator.

Errors are observable events. Per-model errors do not stop other scheduled models; all-model failure stops the run with its checkpoint intact. Invalid configuration/data stops before training. The planner's next iteration is bounded by both experiment and iteration counts. A completed run can be inspected repeatedly without retraining or uploading again.
