from unittest.mock import Mock

from kaggle_slayer.agents.planner import CompetitionPlanner
from kaggle_slayer.config import Settings
from kaggle_slayer.llm.provider import Advice, PlannerLLM
from kaggle_slayer.schemas import RunState
from kaggle_slayer.storage.store import Store
from kaggle_slayer.tools.data_tools import inspect_dataset


def test_deterministic_mode_makes_no_network_call(monkeypatch):
    client = Mock(side_effect=AssertionError("no network"))
    monkeypatch.setattr("httpx.Client", client)
    assert PlannerLLM(Settings(llm_provider="deterministic")).advise({}) == (None, None)


def test_invalid_llm_output_falls_back(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": '{"execute_code":"malicious"}'}}]
    }
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.post.return_value = response
    monkeypatch.setattr("httpx.Client", Mock(return_value=client))
    advice, warning = PlannerLLM(Settings(llm_provider="local", llm_model="test")).advise({})
    assert advice is None and "deterministic" in warning


def test_llm_can_only_reorder_allowlisted_models(fixture_data, monkeypatch):
    directory, c, config = fixture_data
    state = RunState(run_id="llm-plan", competition=c, config=config)
    state.report = inspect_dataset(directory, c, config).report
    monkeypatch.setattr(
        PlannerLLM,
        "advise",
        lambda self, context: (
            Advice(
                rationale="Prefer tree models",
                priorities=["run_shell", "lightgbm", "random_forest"],
            ),
            None,
        ),
    )
    CompetitionPlanner(Store(config.artifact_root)).plan(state)
    assert [candidate.model for candidate in state.plan.queue] == [
        "linear",
        "lightgbm",
        "random_forest",
    ]
    assert state.plan.source == "llm"
