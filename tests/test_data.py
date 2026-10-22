import numpy as np
import pandas as pd
import pytest

from kaggle_slayer.ml.preprocessing import preprocessor
from kaggle_slayer.tools.data_tools import inspect_dataset


def test_transforms_learn_only_from_fit_rows():
    fit = pd.DataFrame({"number": [1.0, 3.0, np.nan], "category": ["a", "a", None]})
    validation = pd.DataFrame({"number": [1000.0, np.nan], "category": ["never_seen", "a"]})
    prep = preprocessor("generic").fit(fit)
    imputer = prep.named_steps["columns"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == 2
    result = prep.transform(validation)
    assert np.isfinite(result).all()
    assert imputer.statistics_[0] == 2


def test_empty_numeric_and_mixed_test_types_are_safe():
    fit = pd.DataFrame(
        {"empty": [np.nan, np.nan, np.nan], "number": [1.0, 2.0, 3.0], "category": ["a", "b", "a"]}
    )
    test = pd.DataFrame({"empty": ["unseen text"], "number": ["bad numeric"], "category": [123]})
    values = preprocessor("generic").fit(fit).transform(test)
    assert np.isfinite(values).all()


def test_inspection_and_stratification(fixture_data):
    directory, c, config = fixture_data
    data = inspect_dataset(directory, c, config)
    assert c.id_column not in data.X and c.target not in data.X
    for fit, valid in data.splits:
        assert not set(fit) & set(valid)
        assert data.y.iloc[valid].nunique() == 2
    assert len(set(np.concatenate([b for _, b in data.splits]))) == len(data.y)


def test_ambiguous_target_and_unknown_metric(fixture_data):
    directory, c, config = fixture_data
    c.metric = None
    with pytest.raises(ValueError, match="metric cannot"):
        inspect_dataset(directory, c, config)
    c.metric, c.target = "accuracy", None
    train = pd.read_csv(directory / "train.csv")
    train["extra_target"] = 1
    train.to_csv(directory / "train.csv", index=False)
    with pytest.raises(ValueError, match="Target is ambiguous"):
        inspect_dataset(directory, c, config)


def test_template_misalignment_rejected(fixture_data):
    directory, c, config = fixture_data
    sample = pd.read_csv(directory / c.sample_file).iloc[::-1]
    sample.to_csv(directory / c.sample_file, index=False)
    with pytest.raises(ValueError, match="identifiers"):
        inspect_dataset(directory, c, config)


def test_exact_leakage_rejected(fixture_data):
    directory, c, config = fixture_data
    train, test = pd.read_csv(directory / "train.csv"), pd.read_csv(directory / "test.csv")
    train["leak"], test["leak"] = train[c.target], 0
    train.to_csv(directory / "train.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)
    with pytest.raises(ValueError, match="leakage"):
        inspect_dataset(directory, c, config)


def test_group_validation_has_no_overlap(fixture_data):
    directory, c, config = fixture_data
    for name in ("train", "test"):
        frame = pd.read_csv(directory / f"{name}.csv")
        frame["group"] = np.arange(len(frame)) // 5
        frame.to_csv(directory / f"{name}.csv", index=False)
    c.validation, c.group_column = "group", "group"
    data = inspect_dataset(directory, c, config)
    for fit, valid in data.splits:
        assert not set(data.train.group.iloc[fit]) & set(data.train.group.iloc[valid])
    assert "group" not in data.X


def test_time_validation_respects_chronology(fixture_data):
    directory, c, config = fixture_data
    for name in ("train", "test"):
        frame = pd.read_csv(directory / f"{name}.csv")
        frame["date"] = pd.date_range("2020-01-01", periods=len(frame)).astype(str)
        frame.to_csv(directory / f"{name}.csv", index=False)
    c.validation, c.time_column = "time", "date"
    data = inspect_dataset(directory, c, config)
    assert all(max(a) < min(b) for a, b in data.splits)


def test_identifiers_keep_leading_zeros(fixture_data):
    directory, c, config = fixture_data
    for name in ("train", "test", "sample_submission"):
        path = directory / f"{name}.csv"
        frame = pd.read_csv(path)
        frame["PassengerId"] = frame.PassengerId.map(lambda value: f"{value:06d}")
        frame.to_csv(path, index=False)
    data = inspect_dataset(directory, c, config)
    assert data.test.PassengerId.iloc[0] == "000091"
    assert data.sample.PassengerId.equals(data.test.PassengerId)


def test_dense_memory_guard(fixture_data):
    directory, c, config = fixture_data
    config.max_dense_cells = 10
    with pytest.raises(ValueError, match="memory guard"):
        inspect_dataset(directory, c, config)
