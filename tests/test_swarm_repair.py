"""Schema repairs use safe feedback and the existing run resource limits."""

import json
from unittest.mock import Mock

import httpx
import pytest
from test_swarm import FixedProvider, setup
from typer.testing import CliRunner

from examples.scripted_swarm import ScriptedTeam
from kaggle_slayer.cli import app
from kaggle_slayer.config import Settings
from kaggle_slayer.llm.provider import ActionLLM
from kaggle_slayer.orchestration.swarm import SwarmRuntime


class RepairTeam(ScriptedTeam):
    def __init__(self):
        super().__init__(refine=False)
        self.rejected = False
        self.repairs = []

    def decide(self, role, context, **limits):
        if role == "data_validation" and context["observations"] and not self.rejected:
            self.rejected = True
            return None, {
                "provider": "fake",
                "status": "failed",
                "code": "schema_mismatch",
                "validation_issues": ["actions.0.finish.summary: missing"],
                "prompt_tokens": 100,
                "completion_tokens": 100,
            }
        if context.get("response_repair"):
            self.repairs.append(context)
        return super().decide(role, context, **limits)


def test_specialist_schema_repair_reaches_real_training(tmp_path):
    provider = RepairTeam()
    engine, state = setup(tmp_path, provider=provider)
    result = engine.execute(state)
    assert result.status == "completed"
    assert len(result.experiments) == 1
    assert provider.repairs[0]["task_id"] == "data"
    assert provider.repairs[0]["response_repair"]["issues"] == ["actions.0.finish.summary: missing"]
    assert (
        "data-inspect_dataset" in provider.repairs[0]["evidence"]
        or provider.repairs[0]["observations"]
    )
    assert result.swarm["invalid_actions"] == 1
    assert not result.swarm["tasks"]["data"]["repair_pending"]


@pytest.mark.parametrize("native", [False, True])
def test_safe_action_field_diagnostics_survive_provider_adapter(monkeypatch, native):
    secret = "private-untrusted-extra-field"
    response = {
        "version": "swarm-v1",
        "actions": [
            {"kind": "finish", "action_id": "stop", "summary": 123, secret: "secret-value"}
        ],
    }
    real = httpx.Client

    def handle(_):
        message = {"content": json.dumps(response)}
        if native:
            message = {
                "tool_calls": [
                    {"function": {"name": "agent_response", "arguments": json.dumps(response)}}
                ]
            }
        return httpx.Response(200, json={"choices": [{"message": message}]})

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real(transport=httpx.MockTransport(handle), **kwargs)
    )
    provider = ActionLLM(
        Settings(_env_file=None, llm_provider="local", llm_model="fake", llm_native_tools=native)
    )
    value, diagnostic = provider.decide("data_validation", {}, timeout=1, max_tokens=1000)
    assert value is None and diagnostic["code"] == "schema_mismatch"
    assert "actions.0.finish.summary: string_type" in diagnostic["validation_issues"]
    assert secret not in json.dumps(diagnostic) and "secret-value" not in json.dumps(diagnostic)
    assert "rationale (string)" not in diagnostic["action"]


def test_schema_repair_is_bounded_and_charged(tmp_path):
    provider = FixedProvider({"version": "swarm-v1", "actions": [{"kind": "finish"}]})
    engine, state = setup(tmp_path, provider=provider)
    result = engine.execute(state)
    assert result.swarm["termination_reason"] == "llm_schema_repair_exhausted"
    assert result.swarm["requests"] == 3
    assert result.swarm["turns"] == 3 and result.swarm["tokens"] > 0
    assert not result.swarm["actions"] and not result.experiments


def test_invalid_json_gets_a_fresh_validated_response(tmp_path):
    provider = FixedProvider()
    provider.decide = Mock(
        side_effect=[
            (None, {"provider": "fake", "status": "failed", "code": "invalid_advice_json"}),
            (
                {
                    "version": "swarm-v1",
                    "actions": [
                        {
                            "kind": "finish",
                            "action_id": "stop",
                            "summary": "No training requested",
                            "outcome": "incomplete",
                        }
                    ],
                },
                {"provider": "fake", "status": "success", "code": "ok"},
            ),
        ]
    )
    engine, state = setup(tmp_path, provider=provider)
    result = engine.execute(state)
    assert result.swarm["termination_reason"] == "coordinator_incomplete"
    assert result.swarm["requests"] == 2
    context = provider.decide.call_args_list[1].args[1]
    assert context["response_repair"]["issues"] == ["response: invalid_json"]


def test_repair_cannot_exceed_request_budget(tmp_path):
    engine, state = setup(tmp_path, provider=FixedProvider({"bad": "json"}), max_llm_requests=1)
    result = engine.execute(state)
    assert result.swarm["termination_reason"] == "llm_request_budget_exhausted"
    assert result.swarm["requests"] == 1
    assert result.swarm["tasks"]["coordinator"]["repair_pending"]


def test_pending_repair_survives_resume(tmp_path, monkeypatch):
    provider = RepairTeam()
    engine, state = setup(tmp_path, provider=provider)
    original = SwarmRuntime.checkpoint
    interrupted = False

    def checkpoint(self, *args, **kwargs):
        nonlocal interrupted
        original(self, *args, **kwargs)
        if not interrupted and any(task.repair_pending for task in self.session.tasks.values()):
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(SwarmRuntime, "checkpoint", checkpoint)
    first = engine.execute(state)
    assert first.status == "interrupted"
    assert first.swarm["tasks"]["data"]["repair_pending"]
    result = engine.execute(engine.store.load(first.run_id))
    assert result.status == "completed" and len(result.experiments) == 1
    assert provider.repairs[0]["task_id"] == "data"


def test_provider_error_is_explicit_without_retry_and_cli_exit_is_clean(tmp_path, monkeypatch):
    provider = FixedProvider()
    provider.decide = Mock(
        return_value=(
            None,
            {
                "provider": "fake",
                "status": "failed",
                "code": "rate_limit",
                "http_status": 429,
                "message": "The API rate limit was reached.",
            },
        )
    )
    engine, state = setup(tmp_path, provider=provider)
    result = engine.execute(state)
    assert result.swarm["termination_reason"] == "llm_rate_limit"
    assert provider.decide.call_count == 1
    monkeypatch.setattr(
        "kaggle_slayer.cli.Engine", Mock(return_value=Mock(execute=Mock(return_value=result)))
    )
    response = CliRunner().invoke(
        app, ["resume", result.run_id, "--artifact-root", result.config.artifact_root]
    )
    assert response.exit_code == 2
    assert "Error: 2" not in response.output
    output = json.loads(response.stdout)
    assert output["llm_diagnostic"]["http_status"] == 429
    assert output["termination_reason"] == "llm_rate_limit"
