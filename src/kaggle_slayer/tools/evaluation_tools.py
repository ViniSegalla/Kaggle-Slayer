import numpy as np
import pandas as pd
from sklearn import metrics

LOWER_IS_BETTER = {"rmse", "rmsle", "log_rmse", "mae", "log_loss"}


def direction(metric: str) -> str:
    return "minimize" if metric in LOWER_IS_BETTER else "maximize"


def score(metric: str, y, prediction, classes=None) -> float:
    prediction = np.asarray(prediction)
    if prediction.dtype.kind not in "OUS" and not np.isfinite(prediction).all():
        raise ValueError("Non-finite model predictions")
    if metric == "accuracy":
        value = metrics.accuracy_score(y, prediction)
    elif metric == "f1":
        value = metrics.f1_score(y, prediction, pos_label=classes[-1], zero_division=0)
    elif metric == "roc_auc":
        value = metrics.roc_auc_score(np.asarray(y) == classes[-1], prediction[:, -1])
    elif metric == "log_loss":
        value = metrics.log_loss(y, prediction, labels=classes)
    elif metric == "rmse":
        value = np.sqrt(metrics.mean_squared_error(y, prediction))
    elif metric == "rmsle":
        value = np.sqrt(metrics.mean_squared_log_error(y, np.maximum(prediction, 0)))
    elif metric == "log_rmse":
        if (np.asarray(y) <= 0).any() or (prediction <= 0).any():
            raise ValueError("Log RMSE requires strictly positive targets and predictions")
        value = np.sqrt(metrics.mean_squared_error(np.log(y), np.log(prediction)))
    elif metric == "mae":
        value = metrics.mean_absolute_error(y, prediction)
    elif metric == "r2":
        value = metrics.r2_score(y, prediction)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    if not np.isfinite(value):
        raise ValueError("Validation metric is non-finite")
    return float(value)


def rank_experiments(experiments, metric):
    eligible = [
        e
        for e in experiments
        if e.status == "completed"
        and e.score is not None
        and np.isfinite(e.score)
        and e.validation_role == "primary"
    ]
    return sorted(eligible, key=lambda e: e.score, reverse=direction(metric) == "maximize")


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame | None,
    competition,
    classes: list,
) -> dict:
    if len(submission) != len(test) or submission.empty:
        raise ValueError("Submission row count differs from test data")
    expected = (
        list(sample.columns)
        if sample is not None
        else [x for x in (competition.id_column, competition.target) if x]
    )
    if list(submission.columns) != expected:
        raise ValueError(f"Submission columns/order must be {expected}")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    if competition.id_column:
        key = competition.id_column
        if not submission[key].reset_index(drop=True).equals(test[key].reset_index(drop=True)):
            raise ValueError("Submission identifier alignment differs from test rows")
        if sample is not None and not submission[key].equals(sample[key]):
            raise ValueError("Submission identifiers differ from template")
    columns = [col for col in submission if col != competition.id_column]
    values = submission[columns]
    if competition.prediction_kind == "label":
        if len(columns) != 1 or not values.iloc[:, 0].isin(classes).all():
            raise ValueError("Submission labels are not in the training classes")
    else:
        try:
            numeric = values.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Submission predictions must be numeric") from exc
        if not np.isfinite(numeric).all():
            raise ValueError("Submission contains non-finite predictions")
        if competition.prediction_kind == "probability":
            if (numeric < 0).any() or (numeric > 1).any():
                raise ValueError("Probabilities must be between zero and one")
            if len(columns) > 1 and not np.allclose(numeric.sum(axis=1), 1, atol=1e-6):
                raise ValueError("Multiclass probabilities must sum to one")
        if competition.metric == "rmsle" and (numeric < 0).any():
            raise ValueError("RMSLE submission predictions must be nonnegative")
        if competition.metric == "log_rmse" and (numeric <= 0).any():
            raise ValueError("Log RMSE submission predictions must be positive")
    return {
        "valid": True,
        "rows": len(submission),
        "columns": list(submission.columns),
        "identifier_alignment": "verified" if competition.id_column else "no identifier configured",
    }
