# LLM diagnostics and troubleshooting

## Agentic runs

The CLI uses `configs/agentic.yaml` when `--config` is omitted. With a custom config,
set `execution_mode: agentic` or pass `--execution-mode agentic`. Configure
`KS_LLM_PROVIDER`, `KS_LLM_MODEL` and the endpoint's credentials in `.env` before running.
Agentic execution requires live LLM actions throughout the session. A provider failure
leaves an incomplete checkpoint; there is no deterministic fallback.

Start diagnosis with the run's saved state:

```bash
uv run kaggle-slayer inspect RUN_ID
```

Inspect `swarm.termination_reason`, `swarm.usage` and `swarm.actions`. The run/resume
command also prints `llm_diagnostic` when a provider error explains the current stop.
Historical failed requests remain in usage history, so an earlier error is not
necessarily the cause of the latest termination. An incomplete run exits with code 2.

| Symptom | Action |
|---|---|
| `llm_authentication`, `llm_credits`, `llm_rate_limit`, or a provider/network error | Correct credentials, quota or connectivity, then resume if saved budgets remain. See the endpoint codes below. |
| `llm_output_truncated` | Increase per-response `KS_LLM_MAX_TOKENS` within the endpoint's supported limits. It accepts 128–32768; the code default is 700 and `.env.example` sets 32768. |
| `llm_schema_repair_exhausted` | Inspect schema-repair observations and endpoint compatibility. The runtime allows at most two consecutive schema correction turns per task; each consumes normal run budgets. |
| Native function calls are rejected | `KS_LLM_NATIVE_TOOLS=true` requires native function-call support. The default `false` uses a validated JSON action envelope. |
| Request, token, turn or wall-time budget exhausted | Start a new run with revised limits. Resume retains consumed budgets and saved constraints. |

The runtime does not retry transport failures or switch providers/models automatically.
Schema correction turns are distinct from transport retries. Arbitrary prose, Markdown
and code are rejected; only validated role-specific actions can execute. Increasing
`KS_LLM_MAX_TOKENS` changes the per-response allowance, not the saved aggregate run limits.
See [agentic usage and recovery](agentic-usage.md) for action contracts, privacy boundaries
and the resume-versus-new-run table.

## Endpoint diagnostic: `llm-check`

```bash
uv run kaggle-slayer llm-check
```

This command sends one synthetic request through the **legacy initial-advice schema**.
It checks the configured model, credentials, JSON response support, network timeout and
output-token allowance. It never trains or uploads, retries, or switches model/provider.
It may consume credits/free-tier quota. The `deterministic` provider makes no request.
A successful check is an endpoint diagnostic, **not a full agentic integration test**:
it does not exercise role-specific actions, native function calls or the scheduler.

Success reports `status=success`, `code=ok`, `request_attempts=1` and HTTP status 200.
Failure reports `status=failed`, exits nonzero, and includes a controlled explanation
and corrective action. Configuration errors can fail before any network request.

| Diagnostic code | Meaning / action |
|---|---|
| `configuration`, `missing_api_key`, `invalid_endpoint` | Check provider, model, key and API base URL. |
| `authentication` | HTTP 401: check whether the API key is valid/enabled. |
| `credits` | HTTP/API 402: check balance and key spending limits. |
| `permission_or_budget` | HTTP/API 403: check model permissions, account/key restrictions and provider data policies. |
| `rate_limit` | HTTP/API 429: wait for quota to reset. |
| `unsupported_format` | The endpoint rejected the requested response format. For JSON mode, check support for `response_format=json_object` and routing settings. |
| `invalid_model`, `not_found`, `no_available_provider` | Check the exact model ID, API base, availability and routing/data policies. |
| `timeout`, `provider_timeout` | Check connectivity/latency and `KS_LLM_TIMEOUT`, a network-operation timeout rather than a total-run limit. Agentic requests also obey the remaining wall-time budget. |
| `connection` | Check endpoint, proxy, TLS and connectivity. |
| `provider_unavailable`, `provider_error` | The API/upstream failed, possibly in an HTTP 200 body. Check the provider's activity page. |
| `invalid_response_json`, `invalid_response` | The endpoint/proxy returned non-JSON or an incompatible completion envelope. |
| `empty_response`, `refusal` | No usable completion or a model refusal. Refusal text is excluded from diagnostics. |
| `output_truncated` | Output exhausted its allowance; reasoning may consume that budget. Increase `KS_LLM_MAX_TOKENS` within supported limits. |
| `invalid_advice_json`, `schema_mismatch` | Output was not JSON or failed the requested schema. These shared code names also appear for agentic action-schema failures. |

The diagnostic records the requested output cap and token counts when supplied by the
provider. Reasoning text is never saved. Compare actual usage and latency: one synthetic
success does not prove every competition request will fit. The OpenRouter references
describe [reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
[API errors](https://openrouter.ai/docs/api/reference/errors-and-debugging) and
[structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).

## Legacy deterministic advice protocol

This section applies to deterministic-mode planning and `llm-check`. It does not
describe the agentic scheduler. See [offline baselines](offline-baselines.md) for
deterministic-mode setup.

With a live provider configured, the deterministic planner can request optional initial
advice. Its fixed Python policy continues if advice fails. Successful advice influences
only the ordering of allowed model families: Python forces an enabled linear baseline
first, appends omitted families and enforces validation, budgets, tuning and submission
permissions. There is no post-training LLM review or automatic second request.

The advice object contains `rationale` (at most 2,000 characters), `priorities` (up to
six allowed model names) and optional `cautions` (up to six strings). Unexpected
properties, Markdown fences and surrounding prose fail validation. The
`initial-planning-v3-compact` prompt in `src/kaggle_slayer/llm/prompts.py` requests a
rationale of at most 60 words and at most two actionable cautions; those prompt targets
are stricter than the parser's compatibility limits.

Its compact context includes task/metric, schema and quality summaries, aggregate target
statistics, validation, compute limits and allowed models. At most eight feature records,
three truncated warnings and three largest class counts are sent. Feature names are
capped at 64 characters and warnings at 120. Feature names and target aggregates can
still be sensitive. Raw rows, category values, class label names, credentials and local
paths are excluded. The same builder produces `llm-check`'s synthetic context.

The deterministic planner persists diagnostics in `metadata.llm_diagnostic`, agent
activity and the state snapshot. Agentic diagnostics instead appear in `swarm.usage`.
Older records containing only a generic warning cannot recover the original HTTP status
or model output retrospectively.

## Diagnostic privacy and offline tests

Diagnostics exclude raw provider messages/metadata, headers, full URLs, prompt bodies,
refusal text and exception strings. Schema failures report controlled field paths and
error types rather than offending values. Known key values and common key patterns are
redacted before successful advice/actions can be logged. Detailed provider errors remain
on the provider's own activity page; no private reasoning is requested or saved.

Offline tests use mock transports, synthetic contexts and fake keys. Workflow tests
default to deterministic planning so a developer's `.env` cannot cause paid API calls.
The [scripted agentic examples](agentic-usage.md#offline-reproduction) exercise the action
protocol with a fake provider and real training workers; live checks remain explicit.
