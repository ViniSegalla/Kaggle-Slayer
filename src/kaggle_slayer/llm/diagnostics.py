"""Safe LLM diagnostics: never retain provider bodies, headers or exception text."""

import re
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class LLMDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    model: str | None = None
    status: Literal["success", "failed", "disabled"]
    code: str
    message: str
    action: str | None = None
    http_status: int | None = None
    api_error_code: int | None = None
    validation_issues: list[str] = Field(default_factory=list)
    request_attempts: Literal[0, 1] = 0
    max_output_tokens: int = 700
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    duration_seconds: float = 0

    def warning(self) -> str:
        status = f", HTTP {self.http_status}" if self.http_status is not None else ""
        return (
            f"LLM planning failed [{self.code}{status}; provider={self.provider}; "
            f"model={self.model or '<unset>'}]. {self.message} {self.action or ''} "
            "Deterministic planning used."
        )


def redact(value: str, secret: str) -> str:
    if secret:
        for form in {secret, quote(secret, safe="")}:
            value = value.replace(form, "[REDACTED]")
    return re.sub(r"\b(?:sk-(?:or-v1-)?|gsk_)[A-Za-z0-9_-]{12,}", "[REDACTED]", value)


def safe_label(value: str, secret: str) -> str:
    return re.sub(r"[^A-Za-z0-9_./:@+\[\]-]", "_", redact(value, secret))[:160]


def validation_issues(error: ValidationError) -> list[str]:
    """Pydantic messages/inputs/unknown field names can contain model output or secrets."""
    issues = []
    for issue in error.errors(include_input=False, include_url=False, include_context=False)[:4]:
        location = issue["loc"]
        field = (
            location[0]
            if location and location[0] in {"rationale", "priorities", "cautions"}
            else "response"
        )
        issues.append(f"{field}: {issue['type']}")
    return issues


def api_failure(status: int, error) -> tuple[str, str, str, int | None]:
    """Classify untrusted provider text locally; only controlled explanations leave here."""
    error = error if isinstance(error, dict) else {}
    code = error.get("code")
    if isinstance(code, str) and re.fullmatch(r"[1-5][0-9]{2}", code):
        code = int(code)
    if not isinstance(code, int) or isinstance(code, bool) or not 100 <= code <= 599:
        code = None
    effective = code if status < 400 and code else status
    message = error.get("message")
    message = message.lower() if isinstance(message, str) else ""
    if effective == 401:
        result = (
            "authentication",
            "The API rejected the credentials.",
            "Check KS_LLM_API_KEY and whether the key is enabled.",
        )
    elif effective == 402:
        result = (
            "credits",
            "The API reported insufficient credits or spending capacity.",
            "Check the provider balance and API-key spending limits before trying again.",
        )
    elif effective == 403:
        result = (
            "permission_or_budget",
            "The API denied this request.",
            "Check model access, provider data policies, and account/key budget restrictions.",
        )
    elif effective == 429:
        result = (
            "rate_limit",
            "The API rate limit was reached.",
            "Wait for the provider limit to reset; this client does not retry automatically.",
        )
    elif effective in {408, 504}:
        result = (
            "provider_timeout",
            "The provider reported a request timeout.",
            "Try later or choose a lower-latency model; no automatic retry was made.",
        )
    elif any(term in message for term in ("response_format", "structured output", "json mode")):
        result = (
            "unsupported_format",
            "The provider rejected the JSON response format or its routing requirements.",
            "Select an endpoint/model that supports response_format=json_object; check OpenRouter routing policies.",
        )
    elif any(
        term in message
        for term in ("not a valid model", "model not found", "unknown model", "invalid model")
    ):
        result = (
            "invalid_model",
            "The provider did not recognize the requested model.",
            "Check KS_LLM_MODEL against the exact provider model ID.",
        )
    elif "no endpoints" in message or effective == 503:
        result = (
            "no_available_provider",
            "No provider endpoint is available for this request.",
            "Check model availability, JSON-mode support and provider routing/data policies.",
        )
    elif effective == 404:
        result = (
            "not_found",
            "The endpoint or model was not found.",
            "Check KS_LLM_MODEL and KS_LLM_BASE_URL (the API base, without /chat/completions).",
        )
    elif effective == 400:
        result = (
            "invalid_request",
            "The API rejected the request parameters.",
            "Check the model's JSON-mode, temperature and token-limit support.",
        )
    elif effective >= 500:
        result = (
            "provider_unavailable",
            "The provider returned a server error.",
            "Try later or choose another available model; no automatic retry was made.",
        )
    else:
        result = (
            "provider_error",
            "The API returned an error instead of planning advice.",
            "Check the provider activity page for request details; raw error content is not logged locally.",
        )
    return *result, code
