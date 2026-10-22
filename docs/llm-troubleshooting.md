# LLM diagnostics and OpenRouter troubleshooting

Configure `KS_LLM_PROVIDER`, `KS_LLM_MODEL` and the endpoint's credentials in `.env`, then run `uv run kaggle-slayer llm-check` before a full competition run. This explicitly makes one chat-completion request with synthetic schema/quality information, the configured model, network timeout and output-token budget. `KS_LLM_MAX_TOKENS` has a code default of 700; `.env.example` sets 32768, the maximum allowed value. Choose an allowance supported by your endpoint. The check never trains, submits to Kaggle, changes model/provider, or retries. It may consume credits/free-tier quota. The `deterministic` provider makes no request; deterministic execution mode can still request optional legacy advice when a live provider is configured.

Success reports `status=success`, `code=ok`, `request_attempts=1` and HTTP status 200. Failure reports `status=failed`, exits nonzero, and includes a controlled explanation/action. A deterministic-mode run retains its fixed policy when initial advice fails; agentic mode checkpoints an incomplete outcome. The diagnostic is persisted in `metadata.llm_diagnostic`, SQLite/JSONL agent activity and the state snapshot; the existing warning also becomes specific.

| Code | Meaning / action |
|---|---|
| `configuration`, `missing_api_key`, `invalid_endpoint` | Check `KS_LLM_PROVIDER`, `KS_LLM_MODEL`, `KS_LLM_API_KEY` and the API base URL. These checks do not make a network request. |
| `authentication` | HTTP 401: check whether the configured API key is valid/enabled. |
| `credits` | HTTP/API 402: check balance, key limits and spending capacity. |
| `permission_or_budget` | HTTP/API 403: check model permissions, account/key budget restrictions and provider data policies. |
| `rate_limit` | HTTP/API 429: wait for quota to reset; the client does not retry. |
| `unsupported_format` | The provider rejected the JSON-format request. Choose an endpoint that supports `response_format=json_object` and check routing settings. |
| `invalid_model`, `not_found`, `no_available_provider` | Check the exact model ID, API base, availability and routing/data policies. A model suffix such as `:free` must actually exist at the provider. |
| `timeout`, `provider_timeout` | Client network timeout or provider-reported timeout. Check latency/connectivity and `KS_LLM_TIMEOUT`. This is a network-operation timeout, not a monetary or total-run limit. |
| `connection` | Network, proxy, TLS or connection failure. Check endpoint and connectivity; raw exception URLs are not logged. |
| `provider_unavailable`, `provider_error` | The API or upstream provider failed, including errors embedded in an HTTP 200 body. Inspect the provider's own activity page if necessary. |
| `invalid_response_json`, `invalid_response` | The endpoint/proxy returned non-JSON or an incompatible completion envelope. |
| `empty_response`, `refusal` | No usable text completion or model refusal. Refusal text is excluded from diagnostics. |
| `output_truncated` | The model exhausted its output budget. Reasoning may consume that budget; increase `KS_LLM_MAX_TOKENS` within 128–32768 or use a model that can provide the small JSON within the limit. |
| `invalid_advice_json`, `schema_mismatch` | Model output was not JSON or failed the required schema. Markdown fences/prose and unexpected properties are rejected. |

The expected advice is an object with `rationale` (string, at most 2,000 characters), `priorities` (up to six strings naming allowed models) and optional `cautions` (up to six strings). Only approved model names affect the experiment roster. For schema failures the diagnostic reports field names and Pydantic error types, never offending values, arbitrary extra-property names or the raw completion.

## Initial planner prompt

The system message is versioned in `src/kaggle_slayer/llm/prompts.py` as
`initial-planning-v3-compact`. It identifies the stage as pre-training, explains that
validation scores do not exist yet, and requests concise observations and
testable hypotheses grounded in supplied summaries. It asks for no more than
two actionable cautions and a rationale of at most 60 words. The existing
parser's limits (2,000 characters and six cautions) remain compatible with
earlier valid responses; the prompt is guidance, not a guarantee of compliance.

The user message contains task, metric/direction, target transform, feature
counts and selected dtype/missingness/cardinality records, duplicate count,
aggregate target distribution, effective validation protocol, configured
compute limits, allowed models and compact preprocessing/loss contracts. Class
balance uses numeric counts without sending class label names. Regression
target statistics are aggregates, not raw rows. Feature names and aggregate
target statistics can still be sensitive. Raw records, category values,
fingerprints, credentials, local paths and submission authorization are not
part of the context.

At most eight feature records, three truncated data warnings and three largest
class counts are sent. Feature names are capped at 64 characters, warnings at
120 characters. Feature examples cover missingness and cardinality extremes;
aggregate maxima summarize the complete schema. Repeated per-model metric,
transform and explanatory notes, fold sizes, seed and tuning-budget internals
are omitted because the LLM cannot change them. The context uses compact JSON.
Truncation
is explicit, and absent legacy metadata remains unknown rather than becoming a
fabricated zero. The same builder creates `llm-check`'s synthetic context.

The LLM can only influence initial ordering among allowed model families.
Python still forces the enabled linear baseline first, appends allowed models
the LLM omitted, and enforces validation, budgets, tuning and submission
permissions. Context text is untrusted data rather than executable instructions.
No second request, automatic retry, model switch or post-training LLM review was
added. `KS_LLM_MAX_TOKENS` and fallback behavior are unchanged.

The compact prompt reduces input overhead; it does not set a separate reasoning
budget or disable model reasoning. One successful synthetic `llm-check` does not
prove that every competition request will fit. Compare `prompt_tokens`,
`completion_tokens`, `reasoning_tokens` and latency for the actual request.
No automatic retries or silent increases to token limits are performed.

Diagnostics also record the requested output cap and any nonnegative integer prompt/completion/reasoning-token counts returned by the API. No reasoning text is stored. These counts explain truncation; they are not a complete cost ledger. For example, set `KS_LLM_MAX_TOKENS=4096` in `.env` when a reasoning model cannot finish within 700 tokens. Increasing the cap may increase usage/cost and is never done automatically by the provider client. See OpenRouter's [reasoning-token budget explanation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

Example **illustrative failure**, not an actual measured run:

```json
{
  "provider": "openrouter",
  "model": "configured/model-id",
  "status": "failed",
  "code": "authentication",
  "message": "The API rejected the credentials.",
  "action": "Check KS_LLM_API_KEY and whether the key is enabled.",
  "http_status": 401,
  "api_error_code": 401,
  "validation_issues": [],
  "request_attempts": 1,
  "duration_seconds": 0.5
}
```

Diagnostics deliberately do not persist raw provider messages, provider metadata, headers, full URLs, prompt bodies, refusal text, or exception strings. Provider text is used only to select controlled categories (for example unsupported `response_format`). Known API-key values and common key patterns are also redacted from successful advice before it can become a logged modeling plan. This balances useful diagnosis with credential and dataset privacy. More detailed provider errors remain on the provider's own activity page.

OpenRouter documents both [HTTP errors and errors inside successful HTTP responses](https://openrouter.ai/docs/api/reference/errors-and-debugging). JSON/structured-output support depends on the endpoint/provider; see its [structured-output documentation](https://openrouter.ai/docs/guides/features/structured-outputs). KaggleSlayer keeps basic JSON mode and local Pydantic validation; it does not silently retry with weaker formats.

Earlier runs that only stored the generic warning cannot reveal the original HTTP status or model output retrospectively. Keep those historical records intact. The check diagnoses the current configuration, while future runs record safe diagnostic details automatically.

Offline tests use HTTPX mock transports, synthetic contexts and fake keys. The test suite forces deterministic planning by default so a developer's real `.env` cannot turn workflow tests into paid API calls. Live provider checks are manual and explicit.
# Agentic sessions

The CLI selects `configs/agentic.yaml` and the action loop when `--config` is omitted.
With a custom config, set `execution_mode: agentic` or pass `--execution-mode agentic`.
The existing `llm-check` probes the legacy advice protocol; it is an endpoint diagnostic,
not a full swarm integration test. In agentic mode, inspect `swarm.usage`, `swarm.actions`
and `swarm.termination_reason`. Provider failure ends with an incomplete checkpoint;
there is no deterministic fallback. Increase per-response `KS_LLM_MAX_TOKENS` separately
from aggregate run request/token/turn budgets. `KS_LLM_NATIVE_TOOLS=true` requires native
function-call support; otherwise use the default validated JSON protocol. No automatic
transport retries or provider/model switching occurs. See [agentic usage](agentic-usage.md).
