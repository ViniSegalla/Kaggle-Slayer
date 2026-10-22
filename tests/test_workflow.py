import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from examples.fixture import make_fixture
from kaggle_slayer.orchestration.engine import Engine
from kaggle_slayer.schemas import Competition, RunConfig
from kaggle_slayer.schemas.models import Candidate
from kaggle_slayer.storage.store import Store
from kaggle_slayer.tools.data_tools import inspect_dataset
from kaggle_slayer.tools.training_tools import run_isolated, train_candidate


def test_default_roster_trains_both_boosters_and_tunes(fixture_data):
    _, competition, config = fixture_data
    # Retain the fixture's smaller folds/CPU budget, but use the production
    # default model roster and experiment/iteration limits, without a run profile.
    defaults = RunConfig()
    config.max_experiments = defaults.max_experiments
    config.max_iterations = defaults.max_iterations
    engine = Engine(Store(config.artifact_root))
    result = engine.execute(engine.create(competition, config))
    assert result.status == "completed"
    baselines = {e.model: e for e in result.experiments if e.id.startswith("baseline-")}
    assert set(baselines) == set(defaults.models)
    assert all(e.status == "completed" for e in baselines.values())
    trials = [e for e in result.experiments if e.id.startswith("tune-")]
    assert len(trials) == defaults.max_experiments - len(defaults.models)
    assert all(e.status == "completed" for e in trials)
    assert Path(result.submission_path).exists()
    assert not result.submissions


def test_end_to_end_and_resume_completed_experiments(fixture_data, monkeypatch):
    _, c, config = fixture_data
    store = Store(config.artifact_root)
    engine = Engine(store)
    state = engine.create(c, config)
    original = engine.evaluator.create_submission
    monkeypatch.setattr(
        engine.evaluator,
        "create_submission",
        Mock(side_effect=RuntimeError("simulated interruption")),
    )
    with pytest.raises(RuntimeError, match="simulated"):
        engine.execute(state)
    saved = store.load(state.run_id)
    assert saved.status == "failed" and len(saved.experiments) == 4
    assert all(e.status == "completed" for e in saved.experiments)
    monkeypatch.setattr(engine.evaluator, "create_submission", original)
    monkeypatch.setattr(
        engine.modeler, "train", Mock(side_effect=AssertionError("must reuse experiments"))
    )
    result = engine.execute(saved)
    assert result.status == "completed" and result.best_id
    assert len(result.experiments) == 4
    assert Path(result.submission_path).exists()
    assert not result.submissions
    assert result.feedback["leaderboard"]["public_score"] is None
    agents = {row["agent"] for row in store.rows("decisions", result.run_id)}
    assert {
        "competition_planner",
        "data_engineer",
        "modeler",
        "evaluator",
        "leaderboard_reviewer",
    } <= agents
    for e in result.experiments:
        oof = np.load(e.oof_path)
        assert oof["valid_mask"].all()
        assert np.isfinite(oof["prediction"]).all()
        assert Path(e.checkpoint).exists()
    log = store.run_dir(result.run_id) / "decisions.jsonl"
    assert all("rationale" in json.loads(line) for line in log.read_text().splitlines())


def test_regression_workflow(tmp_path):
    data = make_fixture(tmp_path / "regression", "regression", rows=60)
    c = Competition(
        competition="regression-fixture",
        target="SalePrice",
        id_column="Id",
        task="regression",
        metric="rmsle",
        target_transform="log1p",
        sample_file="sample_submission.csv",
    )
    config = RunConfig(
        data_dir=str(data),
        artifact_root=str(tmp_path / "artifacts"),
        folds=3,
        max_experiments=4,
        n_jobs=1,
        max_seconds=180,
    )
    engine = Engine(Store(config.artifact_root))
    result = engine.execute(engine.create(c, config))
    eligible = [e.score for e in result.experiments if e.status == "completed"]
    assert result.feedback["local_score"] == min(eligible)
    assert (pd.read_csv(result.submission_path).SalePrice >= 0).all()


def test_multiclass_log_loss_and_probability_output(fixture_data, tmp_path):
    directory, c, config = fixture_data
    train = pd.read_csv(directory / "train.csv")
    train["Survived"] = np.resize(["cat", "dog", "bird"], len(train))
    train.to_csv(directory / "train.csv", index=False)
    c.metric, c.prediction_kind = "log_loss", "probability"
    test = pd.read_csv(directory / "test.csv")
    pd.DataFrame({"PassengerId": test.PassengerId, "bird": 0.0, "cat": 0.0, "dog": 0.0}).to_csv(
        directory / "sample_submission.csv", index=False
    )
    result = train_candidate(
        {
            "competition": c.model_dump(),
            "config": config.model_dump(),
            "candidate": {"id": "multiclass", "model": "linear", "params": {}},
            "output": str(tmp_path / "model"),
        }
    )
    assert result["status"] == "completed" and result["score"] > 0


def test_timeout_records_failure(fixture_data, tmp_path, monkeypatch):
    _, c, config = fixture_data
    monkeypatch.setattr(
        subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("worker", 0.1))
    )
    result = run_isolated(
        Candidate(id="slow", model="linear", rationale="test"), c, config, tmp_path / "slow", 0.1
    )
    assert result.status == "timeout" and "terminated" in result.error


def test_changed_dataset_rejected_on_resume(fixture_data):
    directory, c, config = fixture_data
    engine = Engine(Store(config.artifact_root))
    state = engine.create(c, config)
    state.report = inspect_dataset(directory, c, config).report
    train = pd.read_csv(directory / "train.csv")
    train.loc[0, "Fare"] += 1
    train.to_csv(directory / "train.csv", index=False)
    with pytest.raises(ValueError, match="fingerprints changed"):
        engine.execute(state)


def test_single_run_lock(fixture_data):
    _, _, config = fixture_data
    store = Store(config.artifact_root)
    with store.lock("same-run"):
        with pytest.raises(ValueError, match="locked"):
            with store.lock("same-run"):
                pass


def test_ensemble_respects_total_experiment_budget(fixture_data):
    _, c, config = fixture_data
    config.ensemble, config.max_experiments = True, 5
    engine = Engine(Store(config.artifact_root))
    state = engine.execute(engine.create(c, config))
    assert len(state.experiments) == 5
    assert state.experiments[-1].id == "ensemble-soft"
    assert state.experiments[-1].status == "completed"
    eligible = [e.score for e in state.experiments if e.status == "completed"]
    assert state.feedback["local_score"] == max(eligible)
