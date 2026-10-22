import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from kaggle_slayer.cli import app, llm_failure


def result(reason, usage, status="incomplete"):
    return SimpleNamespace(
        run_id="offline-diagnostic",
        status=status,
        config=SimpleNamespace(execution_mode="agentic", artifact_root="unused"),
        swarm={"termination_reason": reason, "usage": usage},
        best_id=None,
        feedback={},
        submission_path=None,
        warnings=[],
    )


CREDITS = {"status": "failed", "code": "credits", "http_status": 402}
SUCCESS = {"status": "success", "code": "ok"}


@pytest.mark.parametrize(
    "reason,usage,status,expected",
    [
        ("wall_time_exhausted", [CREDITS, SUCCESS], "incomplete", None),
        ("wall_time_exhausted", [CREDITS], "incomplete", None),
        ("llm_request_budget_exhausted", [CREDITS], "incomplete", None),
        ("llm_credits", [CREDITS], "incomplete", CREDITS),
        ("llm_credits", [CREDITS, SUCCESS], "incomplete", None),
        ("llm_rate_limit", [CREDITS], "incomplete", None),
        (None, [CREDITS], "incomplete", None),
        ("llm_credits", [CREDITS], "completed", None),
        ("provider_failure", [CREDITS], "incomplete", None),
    ],
)
def test_only_current_causal_provider_failure_is_reported(reason, usage, status, expected):
    assert llm_failure(result(reason, usage, status)) == expected


def test_exhausted_schema_repair_exposes_final_schema_diagnostic():
    diagnostic = {"status": "failed", "code": "schema_mismatch"}
    assert llm_failure(result("llm_schema_repair_exhausted", [diagnostic])) == diagnostic


@pytest.mark.parametrize("command", ["run", "resume"])
def test_cli_deadline_report_does_not_blame_historical_credits(command, monkeypatch):
    state = result("wall_time_exhausted", [CREDITS, SUCCESS])
    store = Mock(load=Mock(return_value=state))
    monkeypatch.setattr("kaggle_slayer.cli.Store", Mock(return_value=store))
    engine = Mock(create=Mock(return_value=state), execute=Mock(return_value=state))
    monkeypatch.setattr("kaggle_slayer.cli.Engine", Mock(return_value=engine))
    monkeypatch.setattr("kaggle_slayer.cli.load_config", Mock(return_value=(None, state.config)))
    args = ["run", "--competition", "offline"] if command == "run" else ["resume", state.run_id]
    response = CliRunner().invoke(app, args)
    assert response.exit_code == 2
    output = json.loads(response.stdout[response.stdout.index("{") :])
    assert output["termination_reason"] == "wall_time_exhausted"
    assert output["llm_diagnostic"] is None
    assert state.swarm["usage"][0] == CREDITS
