import numpy as np
import pandas as pd
import pytest

from kaggle_slayer.schemas import Competition, Experiment
from kaggle_slayer.tools.evaluation_tools import rank_experiments, score, validate_submission


@pytest.mark.parametrize(
    "metric,y,prediction,expected",
    [
        ("accuracy", [0, 1, 1, 0], [0, 1, 0, 0], 0.75),
        ("rmse", [1, 3], [2, 2], 1),
        ("mae", [1, 3], [2, 2], 1),
        ("rmsle", [0, 3], [0, 3], 0),
        ("r2", [1, 2, 3], [1, 2, 3], 1),
        ("roc_auc", [0, 1], [[0.9, 0.1], [0.1, 0.9]], 1),
        ("f1", [0, 1], [0, 1], 1),
    ],
)
def test_metrics(metric, y, prediction, expected):
    assert score(metric, y, prediction, [0, 1]) == pytest.approx(expected)


def test_log_loss():
    assert score("log_loss", [0, 1], [[0.9, 0.1], [0.1, 0.9]], [0, 1]) == pytest.approx(
        -np.log(0.9)
    )


def test_natural_log_metric_and_string_labels():
    assert score("log_rmse", [1, np.e**2], [np.e, np.e]) == pytest.approx(1)
    assert score("accuracy", ["cat", "dog"], ["cat", "dog"]) == 1
    with pytest.raises(ValueError, match="positive"):
        score("log_rmse", [0, 1], [1, 1])


def test_ranking_respects_direction_and_failure():
    candidates = [
        Experiment(id=str(i), model="linear", params={}, status="completed", score=value)
        for i, value in enumerate([0.2, 0.8])
    ]
    candidates.append(Experiment(id="failed", model="linear", params={}, status="failed"))
    assert rank_experiments(candidates, "rmse")[0].id == "0"
    assert rank_experiments(candidates, "accuracy")[0].id == "1"


def test_submission_contract():
    c = Competition(
        competition="test",
        target="y",
        id_column="id",
        task="classification",
        metric="accuracy",
        prediction_kind="label",
    )
    test = pd.DataFrame({"id": [1, 2]})
    sample = pd.DataFrame({"id": [1, 2], "y": [0, 0]})
    valid = sample.copy()
    valid["y"] = [0, 1]
    assert validate_submission(valid, test, sample, c, [0, 1])["valid"]
    for invalid in (
        valid.iloc[::-1].reset_index(drop=True),
        valid.iloc[:1],
        valid.assign(y=[0, 9]),
        valid.assign(y=[0, np.nan]),
        valid[["y", "id"]],
    ):
        with pytest.raises(ValueError):
            validate_submission(invalid, test, sample, c, [0, 1])


def test_probability_submission_contract():
    c = Competition(
        competition="test",
        target="y",
        id_column="id",
        task="classification",
        metric="log_loss",
        prediction_kind="probability",
    )
    test = pd.DataFrame({"id": [1, 2]})
    sample = pd.DataFrame({"id": [1, 2], "cat": [0.0, 0.0], "dog": [0.0, 0.0]})
    assert validate_submission(
        sample.assign(cat=[0.3, 0.8], dog=[0.7, 0.2]), test, sample, c, ["cat", "dog"]
    )["valid"]
    with pytest.raises(ValueError, match="sum to one"):
        validate_submission(sample, test, sample, c, ["cat", "dog"])
