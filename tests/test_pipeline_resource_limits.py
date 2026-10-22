from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples.fixture import make_fixture
from kaggle_slayer.ml.preprocessing import preprocessor
from kaggle_slayer.ml.resource_limits import check_pipeline_resources, estimate_pipeline_size
from kaggle_slayer.schemas import Competition, RunConfig
from kaggle_slayer.tools.data_tools import DatasetError, inspect_dataset
from kaggle_slayer.tools.training_tools import load_predictions, train_candidate

FAMILIES = ["linear", "random_forest", "hist_gradient", "lightgbm", "xgboost", "catboost"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("profile", ["generic", "titanic"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "missing_indicators": False,
            "max_categories": 2,
            "min_frequency": 1,
            "features": ["missing_count", "numeric_square", "numeric_signed_log"],
        },
        {"missing_indicators": True, "max_categories": 3, "min_frequency": 100},
    ],
)
def test_size_estimate_bounds_real_fold_preprocessing(family, profile, options):
    X = pd.DataFrame(
        {
            "number": [1.0, np.nan, np.inf, 1e200, 4.0, -np.inf] * 4,
            "empty": [np.nan] * 24,
            "category": ["a", "b", np.nan, "c", "a", "d"] * 4,
        }
    )
    if profile == "titanic":
        X["SibSp"] = np.arange(len(X)) % 3
        X["Parch"] = np.arange(len(X)) % 2
        X["Fare"] = np.arange(len(X), dtype=float)
    test_X = X.iloc[np.arange(41) % len(X)].copy().reset_index(drop=True)
    test_X["category"] = [f"unseen-{i}" for i in range(len(test_X))]
    with np.errstate(over="ignore", invalid="ignore"):
        estimate = estimate_pipeline_size(X, test_X, profile=profile, model=family, options=options)
        # Both the final refit and smaller inner/outer folds must fit the bound.
        for indices in (np.arange(len(X)), np.arange(0, len(X), 2), np.arange(1, len(X), 2)):
            prep = preprocessor(profile, family, options)
            fitted = prep.fit_transform(X.iloc[indices])
            predicted = prep.transform(test_X)
            assert estimate["encoded_width"] >= fitted.shape[1]
            assert estimate["estimated_dense_cells"] >= predicted.shape[0] * predicted.shape[1]
    assert estimate["train_rows"] == len(X)
    assert estimate["test_rows"] == len(test_X)
    assert estimate["rows"] == len(test_X)
    assert estimate["estimated_dense_cells"] == estimate["encoded_width"] * len(test_X)


def test_unseen_test_categories_do_not_inflate_fitted_encoding():
    X = pd.DataFrame({"number": np.arange(30), "category": ["a", "b"] * 15})
    repeated = pd.DataFrame({"number": np.arange(60), "category": ["a"] * 60})
    unseen = repeated.assign(category=[f"new-{i}" for i in range(60)])
    first = estimate_pipeline_size(X, repeated)
    second = estimate_pipeline_size(X, unseen)
    assert first["encoded_width"] == second["encoded_width"]
    assert first["estimated_dense_cells"] == second["estimated_dense_cells"]


def test_stringified_mixed_categories_stay_within_estimate():
    X = pd.DataFrame({"category": pd.Series([True, 1, False, 0, "1", None] * 5, dtype=object)})
    estimate = estimate_pipeline_size(X, model="linear")
    transformed = preprocessor("generic", "linear").fit_transform(X)
    assert estimate["encoded_width"] >= transformed.shape[1]


def test_dataset_inspection_allows_native_or_capped_encoding(fixture_data):
    directory, competition, config = fixture_data
    for name in ("train.csv", "test.csv"):
        path = directory / name
        frame = pd.read_csv(path)
        frame["category"] = [f"group-{i}" for i in range(len(frame))]
        frame.to_csv(path, index=False)
    config.max_dense_cells = 1400
    data = inspect_dataset(directory, competition, config)
    for family, options in [("catboost", {}), ("linear", {"max_categories": 2})]:
        check_pipeline_resources(
            data, competition, config, {"model": family, "params": {}, "pipeline": options}
        )
    with pytest.raises(DatasetError, match="max_dense_cells"):
        check_pipeline_resources(data, competition, config, {"model": "linear", "params": {}})


@pytest.mark.parametrize("columns", [[], ["SibSp", "Parch"], ["SibSp", "Parch", "Fare"]])
def test_titanic_profile_counts_only_features_it_creates(columns):
    X = pd.DataFrame({"number": np.arange(20, dtype=float)})
    for column in columns:
        X[column] = np.arange(len(X), dtype=float) % 3
    estimate = estimate_pipeline_size(X, profile="titanic", model="catboost")
    actual = preprocessor("titanic", "catboost").fit_transform(X)
    assert estimate["feature_count"] == actual.shape[1]
    assert estimate["encoded_width"] == actual.shape[1]


def test_titanic_profile_does_not_double_count_overwritten_columns():
    X = pd.DataFrame(
        {name: np.ones(20) for name in ["SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]}
    )
    estimate = estimate_pipeline_size(X, profile="titanic", model="catboost")
    assert estimate["feature_count"] == 6
    assert estimate["encoded_width"] == 6


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_tight_resource_limit_allows_real_model_training(tmp_path: Path, family, task):
    directory = make_fixture(tmp_path / "data", task=task, rows=180)
    classification = task == "classification"
    competition = Competition(
        competition="resource-fixture",
        target="Survived" if classification else "SalePrice",
        id_column="PassengerId" if classification else "Id",
        task=task,
        metric="accuracy" if classification else "rmse",
        target_transform="none" if classification else "log1p",
        sample_file="sample_submission.csv",
        feature_profile="titanic" if classification else "generic",
    )
    config = RunConfig(data_dir=str(directory), folds=2, n_jobs=1, max_dense_cells=9000)
    params = {
        "linear": {},
        "random_forest": {"n_estimators": 3, "max_depth": 2},
        "hist_gradient": {"max_iter": 3},
        "lightgbm": {"n_estimators": 3},
        "xgboost": {"n_estimators": 3, "max_depth": 2},
        "catboost": {"iterations": 3, "depth": 2},
    }[family]
    data = inspect_dataset(directory, competition, config)
    old_width = sum(2 if pd.api.types.is_numeric_dtype(data.X[column]) else 64 for column in data.X)
    assert old_width * len(data.X) > config.max_dense_cells
    result = train_candidate(
        {
            "competition": competition.model_dump(),
            "config": config.model_dump(),
            "candidate": {"id": family, "model": family, "params": params},
            "output": str(tmp_path / family),
        }
    )
    assert result["status"] == "completed"
    assert len(result["fold_scores"]) == 2
    assert Path(result["checkpoint"]).is_file()
    prediction = load_predictions(Path(result["checkpoint"]), data, competition)
    assert len(prediction) == len(data.test_X)
    assert np.isfinite(prediction).all()


def test_native_and_capped_pipelines_fit_where_full_one_hot_does_not():
    X = pd.DataFrame({"number": np.arange(200), "category": [f"c-{i % 100}" for i in range(200)]})
    data = SimpleNamespace(X=X, test_X=X.iloc[:50].copy())
    competition = Competition(competition="resource-fixture")
    config = RunConfig(max_dense_cells=1000)
    for family, options in [("catboost", {}), ("linear", {"max_categories": 2})]:
        estimates = check_pipeline_resources(
            data, competition, config, {"model": family, "params": {}, "pipeline": options}
        )
        assert estimates[0]["estimated_dense_cells"] <= config.max_dense_cells
    with pytest.raises(DatasetError) as raised:
        check_pipeline_resources(data, competition, config, {"model": "linear", "params": {}})
    assert raised.value.code == "pipeline_dense_limit"


def test_ensemble_checks_second_members_own_encoding_and_options():
    X = pd.DataFrame({"number": np.arange(200), "category": [f"c-{i % 100}" for i in range(200)]})
    data = SimpleNamespace(X=X, test_X=X.iloc[:50].copy())
    competition = Competition(competition="resource-fixture")
    config = RunConfig(max_dense_cells=1000)
    members = [
        {"model": "catboost", "params": {}},
        {"model": "linear", "params": {}, "pipeline": {"max_categories": 2}},
    ]
    candidate = {"model": "ensemble", "params": {"members": members}}
    estimates = check_pipeline_resources(data, competition, config, candidate)
    assert len(estimates) == 2
    assert estimates[0]["encoded_width"] == len(X.columns)
    members[1]["pipeline"] = {"max_categories": 64}
    with pytest.raises(DatasetError) as raised:
        check_pipeline_resources(data, competition, config, candidate)
    assert raised.value.code == "pipeline_dense_limit"


def test_derived_features_obey_feature_limit_even_for_native_categories():
    X = pd.DataFrame({f"number-{i}": np.arange(30, dtype=float) for i in range(4)})
    data = SimpleNamespace(X=X, test_X=X.iloc[:10].copy())
    candidate = {
        "model": "catboost",
        "params": {},
        "pipeline": {"features": ["missing_count", "numeric_square", "numeric_signed_log"]},
    }
    with pytest.raises(DatasetError) as raised:
        check_pipeline_resources(
            data, Competition(competition="resource-fixture"), RunConfig(max_features=10), candidate
        )
    assert raised.value.code == "pipeline_feature_limit"
