import json
import time
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kaggle_slayer.config import MAX_LLM_OUTPUT_TOKENS, Settings
from kaggle_slayer.llm.diagnostics import (
    LLMDiagnostic,
    action_validation_issues,
    api_failure,
    redact,
    safe_label,
    validation_issues,
)
from kaggle_slayer.llm.prompts import INITIAL_PLANNING_PROMPT


class Advice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str = Field(max_length=2000)
    priorities: list[str] = Field(max_length=6)
    cautions: list[str] = Field(default_factory=list, max_length=6)


class PlannerLLM:
    """Untrusted suggestions can reorder approved models; never execute generated code."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.last_diagnostic: LLMDiagnostic | None = None

    def advise(self, context: dict) -> tuple[Advice | None, str | None]:
        return self._structured(context, INITIAL_PLANNING_PROMPT, Advice)

    def _structured(self, context, prompt, schema, *, timeout=None, max_tokens=None, native=False):
        settings = self.settings
        max_tokens = max_tokens or settings.llm_max_tokens
        started = time.monotonic()
        attempts = 0
        usage_details = {}
        provider = safe_label(settings.llm_provider, settings.llm_api_key)
        model = safe_label(settings.llm_model, settings.llm_api_key) or None

        def finish(status, code, message, action=None, **details):
            self.last_diagnostic = LLMDiagnostic(
                provider=provider,
                model=model,
                status=status,
                code=code,
                message=message,
                action=action,
                request_attempts=attempts,
                max_output_tokens=max_tokens,
                **usage_details,
                duration_seconds=round(time.monotonic() - started, 3),
                **details,
            )
            return None, self.last_diagnostic.warning() if status == "failed" else None

        if settings.llm_provider == "deterministic":
            return finish(
                "disabled",
                "deterministic",
                "Deterministic planning is configured; no API request was made.",
            )
        urls = {
            "openrouter": "https://openrouter.ai/api/v1",
            "groq": "https://api.groq.com/openai/v1",
            "openai-compatible": "https://api.openai.com/v1",
            "local": "http://localhost:11434/v1",
        }
        base = settings.llm_base_url or urls.get(settings.llm_provider)
        if not base or not settings.llm_model:
            return finish(
                "failed",
                "configuration",
                "LLM configuration is incomplete.",
                "Set KS_LLM_PROVIDER, KS_LLM_MODEL and, for a custom provider, KS_LLM_BASE_URL.",
            )
        if (
            settings.llm_provider in {"openrouter", "groq", "openai-compatible"}
            and not settings.llm_api_key
        ):
            return finish(
                "failed",
                "missing_api_key",
                "No API key is configured for the remote provider.",
                "Set KS_LLM_API_KEY in .env; credentials are never included in diagnostics.",
            )
        try:
            url = httpx.URL(base)
            if url.scheme not in {"http", "https"} or not url.host:
                raise ValueError("invalid endpoint")
        except (httpx.InvalidURL, ValueError):
            return finish(
                "failed",
                "invalid_endpoint",
                "The configured API base URL is invalid.",
                "Set KS_LLM_BASE_URL to an http(s) API base URL, such as https://openrouter.ai/api/v1.",
            )
        try:
            headers = (
                {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
            )
            payload = {
                "model": settings.llm_model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(context, separators=(",", ":"))},
                ],
                "max_tokens": max_tokens,
            }
            if native:
                payload.update(
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "agent_response",
                                "description": "Return typed actions",
                                "parameters": schema.model_json_schema(),
                            },
                        }
                    ],
                    tool_choice={"type": "function", "function": {"name": "agent_response"}},
                )
            else:
                payload["response_format"] = {"type": "json_object"}
            with httpx.Client(timeout=timeout or settings.llm_timeout) as client:
                attempts = 1
                response = client.post(
                    base.rstrip("/") + "/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException:
            return finish(
                "failed",
                "timeout",
                "The API request exceeded its configured network timeout.",
                "Check connectivity or increase KS_LLM_TIMEOUT for a slower model; no automatic retry was made.",
            )
        except httpx.RequestError:
            return finish(
                "failed",
                "connection",
                "The API request could not complete because of a network, TLS, or connection error.",
                "Check network access, proxy/certificate settings and KS_LLM_BASE_URL.",
            )
        except (httpx.InvalidURL, ValueError):
            return finish(
                "failed",
                "request_configuration",
                "The HTTP request could not be constructed.",
                "Check the API URL, credentials/header formatting and timeout configuration.",
            )

        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = None
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            details = usage.get("completion_tokens_details")
            fields = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": details.get("reasoning_tokens")
                if isinstance(details, dict)
                else None,
            }
            usage_details.update(
                {
                    key: value
                    for key, value in fields.items()
                    if isinstance(value, int)
                    and not isinstance(value, bool)
                    and 0 <= value <= 1_000_000_000
                }
            )
        error = body.get("error") if isinstance(body, dict) else None
        if status >= 400 or error is not None:
            code, message, action, api_code = api_failure(status, error)
            return finish(
                "failed", code, message, action, http_status=status, api_error_code=api_code
            )
        if body is None:
            return finish(
                "failed",
                "invalid_response_json",
                "The API response was not valid JSON.",
                "Check the API base URL or an intermediary proxy; response bodies are not logged.",
                http_status=status,
            )
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return finish(
                "failed",
                "invalid_response",
                "The API response did not contain a completion choice.",
                "Check that the endpoint implements OpenAI-compatible chat completions.",
                http_status=status,
            )
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            return finish(
                "failed",
                "output_truncated",
                f"The model exhausted the requested {max_tokens}-token output limit.",
                f"Increase KS_LLM_MAX_TOKENS (128–{MAX_LLM_OUTPUT_TOKENS}) or choose a model that can return the planning JSON within the limit; reasoning tokens may consume the output budget.",
                http_status=status,
            )
        message = choice.get("message")
        if isinstance(message, dict) and message.get("refusal"):
            return finish(
                "failed",
                "refusal",
                "The model refused to provide planning advice.",
                "Review provider policies or choose an appropriate model; refusal text is not logged.",
                http_status=status,
            )
        content = message.get("content") if isinstance(message, dict) else None
        if native and isinstance(message, dict):
            calls = message.get("tool_calls")
            if isinstance(calls, list) and len(calls) == 1:
                function = calls[0].get("function", {})
                content = (
                    function.get("arguments") if function.get("name") == "agent_response" else None
                )
            else:
                content = None
        if not isinstance(content, str) or not content.strip():
            return finish(
                "failed",
                "empty_response",
                "The completion did not contain text advice.",
                "Choose a text model supporting JSON responses; empty/reasoning-only output cannot be used.",
                http_status=status,
            )
        try:
            value = json.loads(content)
        except ValueError:
            return finish(
                "failed",
                "invalid_advice_json",
                "The model's advice was not valid JSON.",
                "Use a model/endpoint supporting JSON mode. Markdown fences and explanatory prose are not accepted.",
                http_status=status,
            )
        try:
            advice = schema.model_validate(value)
        except ValidationError as exc:
            return finish(
                "failed",
                "schema_mismatch",
                "The model returned JSON that does not match the action schema."
                if schema is not Advice
                else "The model returned JSON that does not match the planning schema.",
                "Return the supplied AgentResponse envelope with version and typed actions."
                if schema is not Advice
                else "Expected rationale (string), priorities (list of model names), and optional cautions (list of strings).",
                http_status=status,
                validation_issues=action_validation_issues(exc)
                if schema is not Advice
                else validation_issues(exc),
            )
        advice = schema.model_validate_json(redact(advice.model_dump_json(), settings.llm_api_key))
        finish(
            "success",
            "ok",
            "The API returned valid structured actions."
            if schema is not Advice
            else "The API returned valid structured planning advice.",
            http_status=status,
        )
        return advice, None


class ActionProvider(Protocol):
    """Implementations receive bounded JSON and return validated actions plus safe usage."""

    def decide(self, role: str, context: dict, *, timeout: float, max_tokens: int): ...


class ActionLLM:
    def __init__(self, settings=None):
        self.settings = settings or Settings()

    def decide_isolated(self, role, context, *, timeout, max_tokens, cancelled=lambda: False):
        """Hard total deadline around HTTP, including slow/trickling response bodies.

        Settings travel over an ephemeral local stdin pipe, never a command line or
        persisted request file. Only validated actions/safe diagnostics leave the child.
        """
        import subprocess
        import sys

        payload = json.dumps(
            {
                "settings": self.settings.model_dump(),
                "role": role,
                "context": context,
                "timeout": timeout,
                "max_tokens": max_tokens,
            }
        )
        deadline = time.monotonic() + timeout
        with subprocess.Popen(
            [sys.executable, "-m", "kaggle_slayer.llm.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        ) as process:
            first = True
            try:
                while not cancelled() and time.monotonic() < deadline:
                    try:
                        output, _ = process.communicate(
                            payload if first else None,
                            timeout=min(0.2, max(0.01, deadline - time.monotonic())),
                        )
                        value = json.loads(output)
                        from kaggle_slayer.schemas.actions import AgentResponse

                        return (
                            AgentResponse.model_validate(value["response"])
                            if value["response"]
                            else None
                        ), value["diagnostic"]
                    except subprocess.TimeoutExpired:
                        first = False
                return None, {
                    "provider": safe_label(self.settings.llm_provider, self.settings.llm_api_key),
                    "status": "failed",
                    "code": "deadline_or_cancelled",
                    "request_attempts": 1,
                }
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def decide(self, role, context, *, timeout, max_tokens):
        from kaggle_slayer.agents.swarm import role_prompt
        from kaggle_slayer.schemas.actions import AgentResponse

        # Per-call adapter: diagnostics cannot race between specialist requests.
        adapter = PlannerLLM(self.settings)
        response, _ = adapter._structured(
            context,
            role_prompt(role),
            AgentResponse,
            timeout=timeout,
            max_tokens=max_tokens,
            native=self.settings.llm_native_tools,
        )
        diagnostic = adapter.last_diagnostic.model_dump(mode="json")
        # Legacy advice diagnostics mention fallback; agentic execution never falls back.
        return response, diagnostic
