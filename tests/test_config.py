import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from kaggle_slayer.cli import app
from kaggle_slayer.config import Settings, load_config
from kaggle_slayer.schemas import Competition, RunConfig


def test_default_roster_and_tuning_capacity():
    _, loaded = load_config("home-data-for-ml-course")
    defaults = RunConfig()
    assert (
        loaded.models
        == defaults.models
        == ["linear", "random_forest", "lightgbm", "catboost", "xgboost"]
    )
    assert loaded.max_experiments == defaults.max_experiments == 12
    assert loaded.max_iterations == defaults.max_iterations == 3
    tuning_slots = loaded.max_experiments - len(loaded.models) - 1 - loaded.confirmation_top_k
    assert tuning_slots > 0
    assert loaded.max_iterations * loaded.optuna_trials >= tuning_slots
    # Explicit smaller budgets still take precedence over the new defaults.
    _, limited = load_config("titanic", max_experiments=5)
    assert limited.max_experiments == 5


def test_profile_and_cli_precedence():
    c, config = load_config("titanic", max_experiments=7)
    assert c.target == "Survived" and c.metric == "accuracy"
    assert config.max_experiments == 7
    c, _ = load_config("house-prices-advanced-regression-techniques")
    assert c.target_transform == "log1p" and c.metric == "log_rmse"


@pytest.mark.parametrize(
    "values",
    [
        {"folds": 1},
        {"max_experiments": 2},
        {"max_submissions": 2},
        {"models": ["linear", "linear", "random_forest"]},
        {"n_jobs": -1},
    ],
)
def test_invalid_budgets(values):
    with pytest.raises(ValidationError):
        RunConfig(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"competition": "../escape"},
        {"competition": "x", "train_file": "../secret.csv"},
        {"competition": "x", "validation": "group"},
    ],
)
def test_invalid_competition(values):
    with pytest.raises(ValidationError):
        Competition(**values)


def test_missing_or_mismatched_profile(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_config("x", tmp_path / "missing.yaml")
    path = tmp_path / "profile.yaml"
    path.write_text("competition: another\n")
    with pytest.raises(ValueError, match="does not match"):
        load_config("x", path)


def test_run_yaml_and_ensemble_budget(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("ensemble: true\nmax_experiments: 6\nfolds: 3\n")
    _, config = load_config("titanic", run_profile=path, max_experiments=7)
    assert config.ensemble and config.folds == 3 and config.max_experiments == 7
    assert RunConfig(ensemble=True, max_experiments=3).max_experiments == 3


def test_run_cli_max_rows_override(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    competition = Competition(competition="store-sales-time-series-forecasting")
    config = RunConfig()
    load = Mock(return_value=(competition, config))
    engine = Mock()
    engine.create.return_value = SimpleNamespace(run_id="test-run")
    engine.execute.return_value = SimpleNamespace(
        run_id="test-run",
        status="completed",
        best_id="model",
        feedback={},
        submission_path=None,
        warnings=[],
    )
    monkeypatch.setattr("kaggle_slayer.cli.load_config", load)
    monkeypatch.setattr("kaggle_slayer.cli.Store", Mock())
    monkeypatch.setattr("kaggle_slayer.cli.Engine", Mock(return_value=engine))

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--competition",
            "store-sales-time-series-forecasting",
            "--max-rows",
            "4000000",
            "--no-submit",
        ],
    )

    assert result.exit_code == 0, result.output
    assert load.call_args.kwargs["max_rows"] == 4_000_000


@pytest.mark.parametrize("value", [0, 127, 8193])
def test_llm_output_budget_is_bounded(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_max_tokens=value)
