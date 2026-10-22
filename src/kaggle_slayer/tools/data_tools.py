"""Strict CSV discovery, dataset contracts and reproducible validation indices."""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from kaggle_slayer.schemas import Competition, RunConfig
from kaggle_slayer.storage.store import write_json


class DatasetError(ValueError):
    """Dataset failure with a controlled category for isolated workers."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame | None
    X: pd.DataFrame
    y: pd.Series
    test_X: pd.DataFrame
    splits: list[tuple[np.ndarray, np.ndarray]]
    report: dict


def read_csv(path: Path, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(
            f"Required CSV missing: {path}. Set train_file/test_file/sample_file in YAML."
        )
    if path.stat().st_size > 500_000_000:
        raise ValueError(f"CSV exceeds 500 MB in-memory limit: {path.name}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("CSV must contain unique header columns")
    frame = pd.read_csv(path, nrows=max_rows + 1)
    if len(frame) > max_rows:
        raise DatasetError(
            "dataset_row_limit", f"{path.name} exceeds configured max_rows={max_rows}"
        )
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("CSV must contain rows and unique columns")
    return frame


def fingerprint(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[path.name] = digest.hexdigest()
    return result


def inspect_dataset(directory: Path, competition: Competition, config: RunConfig) -> Dataset:
    c = competition
    train_path, test_path = directory / c.train_file, directory / c.test_file
    train, test = read_csv(train_path, config.max_rows), read_csv(test_path, config.max_rows)
    sample_path = directory / c.sample_file if c.sample_file else None
    if sample_path is None:
        matches = [
            directory / name
            for name in ("sample_submission.csv", "gender_submission.csv")
            if (directory / name).exists()
        ]
        if len(matches) > 1:
            raise ValueError("Multiple submission templates found; set sample_file in YAML")
        if matches:
            sample_path = matches[0]
            c.sample_file = sample_path.name
    sample = read_csv(sample_path, config.max_rows) if sample_path else None
    candidates = [column for column in train if column not in test]
    if c.target is None:
        if len(candidates) != 1:
            raise ValueError(f"Target is ambiguous ({candidates}); set target in competition YAML")
        c.target = candidates[0]
    if c.target not in train or c.target in test:
        raise ValueError("Target must exist only in the training CSV")
    if set(train.columns) - {c.target} != set(test.columns):
        raise ValueError("Train and test feature schemas differ; specify related files correctly")
    if c.id_column is None and sample is not None:
        ids = [
            column
            for column in sample
            if column in test
            and column in train
            and test[column].is_unique
            and train[column].is_unique
        ]
        if len(ids) != 1:
            raise ValueError(f"Identifier ambiguous ({ids}); set id_column in YAML")
        c.id_column = ids[0]
    if c.id_column:
        if c.id_column not in test:
            raise ValueError("Configured identifier missing from test data")
        # Read identifiers verbatim so leading zeros and large integer IDs survive CSV round trips.
        for frame, path in ((train, train_path), (test, test_path), (sample, sample_path)):
            if frame is not None and c.id_column in frame:
                frame[c.id_column] = pd.read_csv(path, usecols=[c.id_column], dtype=str)[
                    c.id_column
                ]
        for frame in (train, test):
            if not frame[c.id_column].is_unique or frame[c.id_column].isna().any():
                raise ValueError("Identifiers must be unique and non-missing")
    if sample is not None:
        if len(sample) != len(test):
            raise ValueError("Submission template and test row counts differ")
        if c.id_column and (
            c.id_column not in sample
            or not sample[c.id_column]
            .reset_index(drop=True)
            .equals(test[c.id_column].reset_index(drop=True))
        ):
            raise ValueError("Submission template identifiers must exactly match test row order")
    y = train[c.target]
    if y.isna().any() or y.nunique() < 2:
        raise ValueError("Target must have at least two distinct non-missing values")
    if pd.api.types.is_numeric_dtype(y) and not np.isfinite(y).all():
        raise ValueError("Target values must be finite")
    if c.task is None:
        if config.execution_mode == "agentic" and not c.metric:
            raise DatasetError(
                "metric_requirement_unresolved",
                "Unresolved task/metric: supply an authoritative metric in the profile",
            )
        c.task = (
            (
                "classification"
                if c.metric in {"accuracy", "f1", "roc_auc", "log_loss"}
                else "regression"
            )
            if config.execution_mode == "agentic"
            else (
                "classification"
                if not pd.api.types.is_numeric_dtype(y)
                or (y.nunique() <= 20 and np.allclose(y, np.round(y)))
                else "regression"
            )
        )
    if c.task == "regression" and (
        not pd.api.types.is_numeric_dtype(y) or not np.isfinite(y).all()
    ):
        raise ValueError("Regression target must be finite numeric values")
    if c.task == "classification" and y.nunique() > 100:
        raise ValueError("More than 100 classes; verify task in competition YAML")
    if c.target_transform == "log1p" and (c.task != "regression" or (y < 0).any()):
        raise ValueError("log1p target transform requires nonnegative regression targets")
    if c.metric is None:
        raise DatasetError(
            "metric_requirement_unresolved",
            "Competition metric cannot be reliably inferred. Set metric in YAML; supported: accuracy, f1, roc_auc, log_loss, rmse, rmsle, log_rmse, mae, r2",
        )
    classification_metrics = {"accuracy", "f1", "roc_auc", "log_loss"}
    if (c.task == "classification") != (c.metric in classification_metrics):
        raise ValueError("Configured metric is incompatible with the task")
    if c.metric == "rmsle" and (y < 0).any():
        raise ValueError("RMSLE requires nonnegative targets")
    if c.metric == "log_rmse" and (y <= 0).any():
        raise ValueError("Log RMSE requires strictly positive targets")
    if c.metric in {"roc_auc", "f1"} and y.nunique() != 2:
        raise ValueError(
            "roc_auc and f1 currently require binary classification; use log_loss or accuracy for multiclass"
        )
    if not c.prediction_kind:
        c.prediction_kind = (
            "probability"
            if c.metric in {"roc_auc", "log_loss"}
            else ("label" if c.task == "classification" else "value")
        )
    if c.task == "classification" and c.prediction_kind == "value":
        raise ValueError("Classification requires label or probability predictions")
    if c.task == "regression" and c.prediction_kind != "value":
        raise ValueError("Regression requires value predictions")
    excluded = set(
        c.drop_columns + [x for x in (c.target, c.id_column, c.group_column, c.time_column) if x]
    )
    features = [column for column in test if column not in excluded]
    if not features or len(features) > config.max_features:
        raise ValueError("Feature count outside configured limits")
    X, test_X = train[features].copy(), test[features].copy()
    estimated_width = sum(
        2
        if pd.api.types.is_numeric_dtype(X[column])
        else min(max(X[column].nunique(), test_X[column].nunique()) + 1, 64)
        for column in features
    ) + (6 if c.feature_profile == "titanic" else 0)
    if estimated_width * max(len(X), len(test_X)) > config.max_dense_cells:
        raise DatasetError(
            "dataset_dense_limit",
            f"Encoded dataset requires an estimated {estimated_width * max(len(X), len(test_X))} "
            f"cells, exceeding max_dense_cells={config.max_dense_cells} memory guard; "
            "reduce data/features or set --max-dense-cells explicitly for a new run",
        )
    if c.validation == "time":
        if c.time_column not in train:
            raise ValueError("Configured time column is missing")
        dates = pd.to_datetime(train[c.time_column], errors="raise", utc=True)
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError(
                "Time validation requires unique, non-missing timestamps; use group validation for tied periods"
            )
        order = np.argsort(dates.to_numpy(), kind="stable")
        X, y = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
        split_source = TimeSeriesSplit(n_splits=config.folds).split(X)
    elif c.validation == "group":
        if c.group_column not in train or train[c.group_column].isna().any():
            raise ValueError("Group column missing or contains nulls")
        if train[c.group_column].nunique() < config.folds:
            raise ValueError("Not enough distinct groups for requested folds")
        split_source = GroupKFold(n_splits=config.folds).split(X, y, train[c.group_column])
    elif c.task == "classification" and c.validation in {"auto", "stratified"}:
        if y.value_counts().min() < config.folds:
            raise ValueError("Minority class has fewer rows than CV folds; reduce folds")
        split_source = StratifiedKFold(config.folds, shuffle=True, random_state=config.seed).split(
            X, y
        )
    else:
        if c.validation == "stratified":
            raise ValueError("Stratified validation requires classification")
        split_source = KFold(config.folds, shuffle=True, random_state=config.seed).split(X)
    splits = list(split_source)
    if len(train) < config.folds * 4:
        raise ValueError("Too few training rows for requested validation folds")
    if c.task == "classification":
        labels = set(y.unique())
        if any(
            set(y.iloc[a].unique()) != labels or set(y.iloc[b].unique()) != labels
            for a, b in splits
        ):
            raise ValueError(
                "Every classification training and validation fold must contain all classes; revise validation"
            )
    warnings = []
    duplicate_count = int(X.duplicated().sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate feature rows; consider grouped validation")
    for column in X:
        if X[column].equals(y):
            raise ValueError(
                f"Potential exact target leakage in feature {column}; review and exclude it explicitly"
            )
        if not pd.api.types.is_numeric_dtype(X[column]) and X[column].nunique() > len(X) * 0.8:
            warnings.append(
                f"High-cardinality feature {column}; audit model-specific categorical handling and generalization"
            )
    paths = [train_path, test_path] + ([sample_path] if sample_path else [])
    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "features": features,
        "numeric_features": list(X.select_dtypes(include="number").columns),
        "categorical_features": list(X.select_dtypes(exclude="number").columns),
        "categorical_cardinality": {
            c: int(X[c].nunique(dropna=True)) for c in X.select_dtypes(exclude="number")
        },
        "missing_fraction": X.isna().mean().to_dict(),
        "dtypes": X.dtypes.astype(str).to_dict(),
        "target_distribution": y.value_counts().to_dict()
        if c.task == "classification"
        else y.describe().to_dict(),
        "warnings": warnings,
        "duplicate_rows": duplicate_count,
        "fingerprints": fingerprint(paths),
        "classes": sorted(y.unique().tolist()) if c.task == "classification" else [],
        "fold_sizes": [{"train": len(a), "validation": len(b)} for a, b in splits],
    }
    return Dataset(train, test, sample, X, y, test_X, splits, report)


def save_eda(data: Dataset, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "dataset_report.json", data.report)
    np.savez_compressed(
        output / "splits.npz",
        **{
            f"{i}_{name}": values
            for i, split in enumerate(data.splits)
            for name, values in zip(("train", "validation"), split)
        },
    )
    missing = pd.DataFrame(
        {"feature": data.X.columns, "missing_fraction": data.X.isna().mean().values}
    )
    px.bar(
        missing,
        x="feature",
        y="missing_fraction",
        title="Training feature missingness",
        template="plotly_dark",
    ).write_html(output / "missingness.html", include_plotlyjs=True)
    px.histogram(
        data.y.to_frame("target"),
        x="target",
        title="Training target distribution",
        template="plotly_dark",
    ).write_html(output / "target.html", include_plotlyjs=True)
    return [str(path) for path in output.iterdir() if path.is_file()]
