from pathlib import Path

import pytest

from examples.fixture import make_fixture
from kaggle_slayer.schemas import Competition, RunConfig
from kaggle_slayer.tools.training_tools import train_candidate


@pytest.mark.parametrize("family", ["hist_gradient", "lightgbm", "xgboost", "catboost"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_additional_model_families(tmp_path: Path, family, task):
    directory = make_fixture(tmp_path / "data", task=task, rows=150)
    classification = task == "classification"
    c = Competition(
        competition="model-fixture",
        target="Survived" if classification else "SalePrice",
        id_column="PassengerId" if classification else "Id",
        task=task,
        metric="accuracy" if classification else "log_rmse",
        sample_file="sample_submission.csv",
        target_transform="none" if classification else "log1p",
    )
    config = RunConfig(data_dir=str(directory), folds=3, n_jobs=1)
    key = {
        "hist_gradient": "max_iter",
        "lightgbm": "n_estimators",
        "xgboost": "n_estimators",
        "catboost": "iterations",
    }[family]
    result = train_candidate(
        {
            "competition": c.model_dump(),
            "config": config.model_dump(),
            "candidate": {"id": family, "model": family, "params": {key: 20}},
            "output": str(tmp_path / "model"),
        }
    )
    assert result["status"] == "completed"
    assert len(result["fold_scores"]) == 3
    assert Path(result["checkpoint"]).exists()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_voting_ensemble_is_measured_on_folds(tmp_path, task):
    data = make_fixture(tmp_path / "data", task=task, rows=60)
    c = Competition(
        competition="ensemble-fixture",
        task=task,
        target="Survived" if task == "classification" else "SalePrice",
        id_column="PassengerId" if task == "classification" else "Id",
        metric="accuracy" if task == "classification" else "rmse",
        sample_file="sample_submission.csv",
    )
    config = RunConfig(data_dir=str(data), folds=3, n_jobs=1)
    result = train_candidate(
        {
            "competition": c.model_dump(),
            "config": config.model_dump(),
            "candidate": {
                "id": "ensemble",
                "model": "ensemble",
                "params": {
                    "members": [
                        {"model": "linear", "params": {}},
                        {"model": "catboost", "params": {"iterations": 15}},
                    ]
                },
            },
            "output": str(tmp_path / "ensemble"),
        }
    )
    assert result["status"] == "completed" and len(result["fold_scores"]) == 3
