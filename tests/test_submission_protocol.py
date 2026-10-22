"""Submission schema repairs preserve measured evidence and upload authority."""

import json

import httpx
import pytest
from test_swarm import setup

from examples.scripted_swarm import ScriptedTeam
from kaggle_slayer.agents.swarm import ROLE_PROMPTS, protocol_schema, role_response_schema
from kaggle_slayer.config import Settings
from kaggle_slayer.llm.provider import ActionLLM
from kaggle_slayer.tools.swarm_tools import CAPABILITIES


class SubmissionRepairTeam(ScriptedTeam):
    def __init__(self, always_invalid=False):
        super().__init__(refine=False)
        self.bad_validation = self.bad_report = False
        self.repairs = []
        self.always_invalid = always_invalid

    def decide(self, role, context, **limits):
        if role == "submission":
            results = [
                obs.get("result", {})
                for obs in context["observations"]
                if obs.get("code") == "tool_result" and obs["task_id"] == context["task_id"]
            ]
            if context.get("response_repair"):
                self.repairs.append(context)
                issue = context["response_repair"]["issues"][0]
                assert "kind=tool" in issue and "kind=finish" in issue
                assert set(
                    context["response_schema"]["properties"]["actions"]["items"]["discriminator"][
                        "mapping"
                    ]
                ) == {"tool", "wait", "finish"}
            wrong_kind = None
            if self.always_invalid and results:
                wrong_kind = "submit"
            elif any(result.get("valid") for result in results) and not self.bad_report:
                self.bad_report = True
                wrong_kind = "report_results"
            elif results and not self.bad_validation:
                self.bad_validation = True
                wrong_kind = "validate_predictions"
            if wrong_kind:
                return {
                    "version": "swarm-v1",
                    "actions": [
                        {
                            "kind": wrong_kind,
                            "action_id": "wrong_kind",
                            "summary": "Request operation",
                        }
                    ],
                }, {"provider": "fake", "status": "success", "code": "ok"}
        return super().decide(role, context, **limits)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_invalid_validation_and_reporting_kinds_repair_to_real_validated_results(tmp_path, task):
    provider = SubmissionRepairTeam()
    engine, state = setup(tmp_path, task=task, provider=provider)
    result = engine.execute(state)
    assert result.status == "completed", result.swarm
    assert len(provider.repairs) == 2
    assert result.swarm["schema_repairs"] == 2 and result.swarm["invalid_actions"] == 0
    assert len(result.experiments) == 1 and result.experiments[0].score is not None
    assert result.metadata["swarm_submission_validated"] == result.metadata["submission_sha256"]
    kinds = {record["action"]["kind"] for record in result.swarm["actions"].values()}
    assert kinds <= {"tool", "delegate", "wait", "finish"}
    assert not engine.client.submit.called


def test_invalid_submit_kind_is_never_coerced_or_uploaded(tmp_path):
    engine, state = setup(tmp_path, provider=SubmissionRepairTeam(always_invalid=True), submit=True)
    result = engine.execute(state)
    assert result.swarm["termination_reason"] == "llm_schema_repair_exhausted"
    assert not result.metadata.get("swarm_submission_validated")
    assert result.submissions == [] and not engine.client.submit.called


def test_exhausted_submission_repair_resumes_without_retraining_or_upload(tmp_path, monkeypatch):
    engine, state = setup(tmp_path, provider=SubmissionRepairTeam(always_invalid=True))
    stopped = engine.execute(state)
    assert stopped.swarm["tasks"]["results"]["repair_attempts"] == 3
    assert stopped.submission_path
    engine.llm = ScriptedTeam(refine=False)
    monkeypatch.setattr(
        "kaggle_slayer.orchestration.swarm.run_isolated",
        lambda *args, **kwargs: pytest.fail("Resume must reuse completed training"),
    )
    resumed = engine.execute(engine.store.load(stopped.run_id))
    assert resumed.status == "completed", resumed.swarm
    assert len(resumed.experiments) == len(stopped.experiments) == 1
    assert resumed.experiments[0].id == stopped.experiments[0].id
    assert resumed.metadata["swarm_submission_validated"] == resumed.metadata["submission_sha256"]
    assert not resumed.swarm["tasks"]["results"]["repair_pending"]
    assert not engine.client.submit.called


@pytest.mark.parametrize("role", list(ROLE_PROMPTS))
@pytest.mark.parametrize("native", [False, True])
def test_provider_gets_scoped_schema_for_json_and_native_protocols(monkeypatch, role, native):
    real_client = httpx.Client
    captured = []
    value = {
        "version": "swarm-v1",
        "actions": [
            {
                "kind": "finish",
                "action_id": "stop",
                "summary": "No further work",
                "outcome": "incomplete",
            }
        ],
    }

    def handle(request):
        captured.append(json.loads(request.content))
        message = (
            {
                "tool_calls": [
                    {"function": {"name": "agent_response", "arguments": json.dumps(value)}}
                ]
            }
            if native
            else {"content": json.dumps(value)}
        )
        return httpx.Response(200, json={"choices": [{"message": message}]})

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    provider = ActionLLM(
        Settings(_env_file=None, llm_provider="local", llm_model="fake", llm_native_tools=native)
    )
    response, diagnostic = provider.decide(role, {}, timeout=1, max_tokens=1000)
    assert response and diagnostic["status"] == "success"
    request = captured[0]
    schema = json.loads(request["messages"][1]["content"])["response_schema"]
    assert schema == role_response_schema(role)
    kinds = schema["properties"]["actions"]["items"]["discriminator"]["mapping"]
    expected_tools = {name for name, cap in CAPABILITIES.items() if role in cap.roles}
    if expected_tools:
        assert set(schema["$defs"]["ToolAction"]["properties"]["tool"]["enum"]) == expected_tools
    else:
        assert "tool" not in kinds and "ToolAction" not in schema["$defs"]
    if role in {"submission", "data_validation"}:
        assert "delegate" not in kinds and "DelegateAction" not in schema["$defs"]
    if native:
        assert request["tools"][0]["function"]["parameters"] == schema
    assert "ToolAction" in protocol_schema()["$defs"]  # No mutation of the full wire protocol.


@pytest.mark.parametrize("native", [False, True])
def test_adapter_unknown_kind_returns_actionable_safe_feedback(monkeypatch, native):
    real_client = httpx.Client
    bad = {"version": "swarm-v1", "actions": [{"kind": "private-output-tag"}]}

    def handle(_):
        message = (
            {"tool_calls": [{"function": {"name": "agent_response", "arguments": json.dumps(bad)}}]}
            if native
            else {"content": json.dumps(bad)}
        )
        return httpx.Response(200, json={"choices": [{"message": message}]})

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    provider = ActionLLM(
        Settings(_env_file=None, llm_provider="local", llm_model="fake", llm_native_tools=native)
    )
    response, diagnostic = provider.decide("submission", {}, timeout=1, max_tokens=1000)
    assert response is None and diagnostic["code"] == "schema_mismatch"
    assert "kind=tool" in diagnostic["validation_issues"][0]
    assert "kind=finish" in diagnostic["validation_issues"][0]
    assert "private-output-tag" not in json.dumps(diagnostic)
