# Offline baselines and compatibility mode

The main workflow uses an LLM coordinator and specialists. Deterministic execution remains available for offline tests, fixed baseline comparisons and existing saved runs. It is not the default CLI workflow.

The provider and execution mode are independent. Set both explicitly for offline CSV workflows, preferably in the current shell so your working agentic `.env` stays configured:

```powershell
$env:KS_LLM_PROVIDER = 'deterministic'
uv run kaggle-slayer run --competition titanic --data-dir data/titanic --execution-mode deterministic --no-submit
```

Start a fresh shell afterward to return to the provider configured in `.env`, or remove the shell override. Changing the provider alone does not change execution mode. The library settings retain a deterministic default for compatibility; the `.env.example` template selects OpenRouter for the normal agentic setup.

## Deterministic mode and legacy initial advice

`KS_LLM_PROVIDER=deterministic` disables LLM requests. Pair it with `--execution-mode deterministic` for local CSV workflows without a provider key or network access. The deterministic agents still perform specialized work, including feedback-driven tuning.

```bash
# With KS_LLM_PROVIDER=deterministic in .env and existing local CSVs:
uv run kaggle-slayer run --competition titanic --data-dir data/titanic --execution-mode deterministic --no-submit
```

To configure an endpoint in `.env`:

```dotenv
KS_LLM_PROVIDER=openrouter
KS_LLM_API_KEY=your-own-key
KS_LLM_MODEL=your-selected-model-id
```

Providers: `openrouter`, `groq`, `openai-compatible`, `local`, or `deterministic`. For a local compatible server, use `KS_LLM_PROVIDER=local`, `KS_LLM_BASE_URL=http://localhost:11434/v1` and a model installed on that server. No local model download is automatic. In deterministic mode, invalid/unavailable initial advice falls back to deterministic planning with a warning. Agentic mode never falls back. The legacy initial request sends bounded schema/quality and target aggregates, validation settings, budgets and model capabilities. Column names and target statistics may still be sensitive in that legacy request; agentic context uses opaque feature IDs. Raw rows, category values, class label names, credentials and local paths are excluded. Suggestions cannot change metrics, budgets or submission authorization.

The versioned [initial planner prompt](../src/kaggle_slayer/llm/prompts.py) belongs to deterministic mode's optional legacy advice. It requests a brief rationale, allowed-model priorities and up to two dataset risks before training; Python retains the experiment policy. The agentic team instead uses the [role-specific action contracts](contracts/swarm-v1) throughout execution.

To test one provider request using the legacy planning schema, without downloading data or training models:

```bash
uv run kaggle-slayer llm-check
```

This sends **one** request with synthetic context using the configured provider/model and output budget. It checks provider connectivity and legacy advice parsing; it does not demonstrate the full agentic action protocol or workflow. `KS_LLM_MAX_TOKENS` accepts 128–32768: the code default is 700, while the supplied `.env.example` overrides it to 32768. Choose a value supported by your endpoint and within the run's aggregate budget. This is a per-request output limit, not a monetary budget. The check may consume provider credits or free-tier quota. It never retries, changes models, or contacts Kaggle. A failed check exits with code 1; a deterministic provider reports `disabled` and makes no request. The ordinary `doctor` command does not test LLM requests.

Failures now identify configuration problems, HTTP authentication/credit/rate-limit errors, unsupported JSON mode, timeouts, invalid JSON, truncated output or planning-schema mismatches. Safe diagnostics include provider/model, HTTP status, a category and corrective action. Model runs save these under `metadata.llm_diagnostic` and in an agent decision, visible in the dashboard. Credentials, response bodies, headers, exception text and failed model-output values are excluded. See [LLM troubleshooting](llm-troubleshooting.md).

## Search policy and configuration

| Setting | Agentic CLI default | Deterministic default |
|---|---|---|
| Run profile | `configs/agentic.yaml` | `configs/default.yaml` |
| Experiment budget | 8 | 12 |
| Per-experiment timeout | 180 seconds | 300 seconds |
| Validation / CPU / total time | 5 folds / 2 threads / 30 minutes | 5 folds / 2 threads / 30 minutes |
| Model roster | Six allowed families, including histogram gradient boosting | Five baseline families, excluding histogram gradient boosting |
| Experiment scheduling | LLM-selected actions; tuning, ensembles and confirmations are optional capabilities | Normally five baselines, four tuning trials, one ensemble and two finalist confirmations |

The agentic profile additionally caps requests/turns at 80, aggregate charged tokens at 1,000,000 and concurrent agent calls at two. There is no mandatory baseline sequence or reserved optional-work queue in this mode. An eight-experiment budget is a ceiling, not a requirement to train eight models. Edit `models` in the selected run profile or pass an explicit `--config`; the agentic profile overrides the roster in `configs/default.yaml`.

When passing a custom `--config`, set `execution_mode: agentic` in that file or pass `--execution-mode agentic` to retain agentic execution. An explicit profile without an execution mode uses the schema's deterministic default.

In deterministic mode, the planner explores leading families before concentrating remaining trials on the winner. `--max-experiments 5` funds its five baselines. Failed training attempts consume budget; time limits may stop either mode early. Saved runs retain their recorded configuration.

The deterministic team uses a Competition Planner, Data Engineer, Modeler, Evaluator and Leaderboard Reviewer with distinct tools and policies. Its reviewer can write `next_run_plan.json` after public feedback or refresh; the current agentic loop does not reconvene after upload.

## Cached-data baseline

With official Home Data CSVs already downloaded and competition rules accepted, run an offline baseline with both provider and execution mode set explicitly:

```powershell
$env:KS_LLM_PROVIDER = 'deterministic'
uv run kaggle-slayer run --competition home-data-for-ml-course --data-dir data/home-data-for-ml-course --execution-mode deterministic --no-submit
```

The bundled profile supplies the target and MAE metric. New results and model checkpoints remain under the ignored local artifact directory.

## Development datasets and demonstrations

These scripts select deterministic execution directly. Set the provider override in the current PowerShell session to disable their optional legacy LLM advice request; a configured remote provider can otherwise make a request even in deterministic execution.

```powershell
$env:KS_LLM_PROVIDER = 'deterministic'
# Offline synthetic examples.
uv run python scripts/demo.py
uv run python scripts/demo.py --task regression

# Real public observations; public internet, no Kaggle credentials.
uv run python scripts/public_titanic_demo.py
```

The public demo downloads from the [seaborn example data repository](https://github.com/mwaskom/seaborn-data), creates a local stratified 80/20 holdout and evaluates it once after selection. It records the downloaded source hash but currently uses a mutable `master` URL rather than a pinned revision or expected hash. Future downloads may therefore differ from the historical evidence. **This is not an official Kaggle download, test set, submission or leaderboard result.** Demo data is marked in `provenance.json` and prohibited from upload.

These synthetic/public-data demonstrations are development evidence. The primary submission demonstration is the [recorded Home Data agentic run](../demo/home-data-for-ml-course-8ed561cbed5e/README.md).
