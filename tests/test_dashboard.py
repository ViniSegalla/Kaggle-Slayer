from pathlib import Path

from streamlit.testing.v1 import AppTest

from kaggle_slayer.agents.data_agent import DataAgent
from kaggle_slayer.agents.planner import CompetitionPlanner
from kaggle_slayer.orchestration.engine import Engine
from kaggle_slayer.storage.store import Store


def test_dashboard_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("KS_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard/app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert "No runs yet" in app.info[0].value


def test_dashboard_with_persisted_run(fixture_data, monkeypatch):
    _, competition, config = fixture_data
    store = Store(config.artifact_root)
    engine = Engine(store)
    state = engine.create(competition, config)
    DataAgent(store).prepare(state, engine.client)
    CompetitionPlanner(store).plan(state)
    store.save(state)
    monkeypatch.setenv("KS_ARTIFACT_ROOT", config.artifact_root)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard/app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 5
    assert app.metric[0].value == "1"
