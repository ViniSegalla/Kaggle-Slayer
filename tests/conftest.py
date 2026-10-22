import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.fixture import make_fixture
from kaggle_slayer.schemas import Competition, RunConfig


@pytest.fixture(autouse=True)
def offline_llm_by_default(monkeypatch):
    # A developer's real .env must never turn offline workflow tests into paid API calls.
    monkeypatch.setenv("KS_LLM_PROVIDER", "deterministic")


@pytest.fixture
def fixture_data(tmp_path):
    directory = make_fixture(tmp_path / "data", rows=90)
    competition = Competition(
        competition="offline-test",
        target="Survived",
        id_column="PassengerId",
        task="classification",
        metric="accuracy",
        sample_file="sample_submission.csv",
        feature_profile="titanic",
    )
    config = RunConfig(
        data_dir=str(directory),
        artifact_root=str(tmp_path / "artifacts"),
        folds=3,
        max_experiments=4,
        max_seconds=180,
        n_jobs=1,
    )
    return directory, competition, config
