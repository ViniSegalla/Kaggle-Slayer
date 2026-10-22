import pytest
from pydantic import ValidationError

from kaggle_slayer.config import Settings, load_config
from kaggle_slayer.schemas import Competition, RunConfig


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
    with pytest.raises(ValidationError, match="at least four"):
        RunConfig(ensemble=True, max_experiments=3)


@pytest.mark.parametrize("value", [0, 127, 8193])
def test_llm_output_budget_is_bounded(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_max_tokens=value)
