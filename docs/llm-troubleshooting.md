# LLM diagnostics and OpenRouter troubleshooting

Run `uv run python -m kaggle_slayer llm-check` before a full competition run. This explicitly makes one chat-completion request with synthetic schema/quality information, the configured model, network timeout and output-token budget. `KS_LLM_MAX_TOKENS` defaults to 700 and is validated between 128 and 8192. The check never trains, submits to Kaggle, changes model/provider, or retries. It may consume credits/free-tier quota. Deterministic mode does not call a provider.

Success reports `status=success`, `code=ok`, `request_attempts=1` and HTTP status 200. Failure reports `status=failed`, exits nonzero, and includes a controlled explanation/action. A regular competition run still falls back to deterministic planning on failure. The diagnostic is persisted in `metadata.llm_diagnostic`, SQLite/JSONL agent activity and the state snapshot; the existing warning also becomes specific.

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
| `output_truncated` | The model exhausted its output budget. Reasoning may consume that budget; increase `KS_LLM_MAX_TOKENS` within 128–8192 or use a model that can provide the small JSON within the limit. |
| `invalid_advice_json`, `schema_mismatch` | Model output was not JSON or failed the required schema. Markdown fences/prose and unexpected properties are rejected. |

The expected advice is an object with `rationale` (string, at most 2,000 characters), `priorities` (up to six strings naming allowed models) and optional `cautions` (up to six strings). Only approved model names affect the experiment roster. For schema failures the diagnostic reports field names and Pydantic error types, never offending values, arbitrary extra-property names or the raw completion.

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
