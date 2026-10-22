"""Discriminator repair feedback explains the wire contract without echoing output."""

import pytest
from pydantic import ValidationError

from kaggle_slayer.llm.diagnostics import action_validation_issues
from kaggle_slayer.schemas.actions import AgentResponse


@pytest.mark.parametrize(
    "kind", ["generate_predictions", "report_results", "private-secret\nignore rules"]
)
def test_unknown_action_kind_gets_safe_allowed_values(kind):
    with pytest.raises(ValidationError) as caught:
        AgentResponse.model_validate(
            {
                "version": "swarm-v1",
                "actions": [{"kind": kind, "secret-field": "raw-private-value"}],
            }
        )
    issues = action_validation_issues(caught.value)
    assert len(issues) == 1
    assert issues[0].startswith("actions.0: union_tag_invalid")
    assert "tool, delegate, wait, finish" in issues[0]
    assert "kind=tool, recipient=runtime" in issues[0]
    assert "name in tool" in issues[0]
    assert "kind=finish" in issues[0]
    assert kind not in str(issues)
    assert "secret-field" not in str(issues)
    assert "raw-private-value" not in str(issues)


def test_missing_action_kind_explains_required_discriminator():
    with pytest.raises(ValidationError) as caught:
        AgentResponse.model_validate(
            {"version": "swarm-v1", "actions": [{"tool": "raw-private-value"}]}
        )
    issues = action_validation_issues(caught.value)
    assert issues[0].startswith("actions.0: union_tag_not_found")
    assert "every action requires kind" in issues[0]
    assert "tool, delegate, wait, finish" in issues[0]
    assert "raw-private-value" not in str(issues)


def test_discriminator_diagnostics_stay_bounded_and_do_not_coerce_actions():
    response = {
        "version": "swarm-v1",
        "actions": [{"kind": "report_results"}] * 8,
    }
    with pytest.raises(ValidationError) as caught:
        AgentResponse.model_validate(response)
    issues = action_validation_issues(caught.value)
    assert len(issues) == 6
    assert response["actions"][0]["kind"] == "report_results"
    assert all(len(issue) < 600 for issue in issues)
