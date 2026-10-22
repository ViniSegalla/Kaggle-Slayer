"""Fit each fold in isolation and retain OOF evidence, then refit on all training rows."""

import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from kaggle_slayer.ml.models import estimator, pipeline
from kaggle_slayer.ml.preprocessing import preprocessor
from kaggle_slayer.schemas import Competition, Experiment, RunConfig
from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.data_tools import inspect_dataset
from kaggle_slayer.tools.evaluation_tools import score


def early_stopping_params(name, params, X, y, competition, config):
    """Use only an inner portion of the outer training fold for stopping decisions."""
    if (
        name not in {"lightgbm", "xgboost", "catboost"}
        or competition.validation in {"group", "time"}
        or len(X) < 80
    ):
        return params
    if competition.task == "classification" and (
        y.value_counts().min() < 2 or int(np.ceil(len(y) * 0.15)) < y.nunique()
    ):
        return params
    a, b = train_test_split(
        np.arange(len(X)),
        test_size=0.15,
        random_state=config.seed,
        stratify=y if competition.task == "classification" else None,
    )
    prep = preprocessor(competition.feature_profile)
    fit_X, stop_X = prep.fit_transform(X.iloc[a]), prep.transform(X.iloc[b])
    work_y = np.log1p(np.asarray(y)) if competition.target_transform == "log1p" else np.asarray(y)
    model = estimator(name, competition.task, config.seed, config.n_jobs, params)
    if name == "lightgbm":
        from lightgbm import early_stopping, log_evaluation

        model.fit(
            fit_X,
            work_y[a],
            eval_set=[(stop_X, work_y[b])],
            callbacks=[early_stopping(15, verbose=False), log_evaluation(0)],
        )
        count = model.best_iteration_
    elif name == "xgboost":
        model.set_params(early_stopping_rounds=15)
        model.fit(fit_X, work_y[a], eval_set=[(stop_X, work_y[b])], verbose=False)
        count = model.best_iteration + 1
    else:
        model.fit(fit_X, work_y[a], eval_set=(stop_X, work_y[b]), early_stopping_rounds=15)
        count = model.get_best_iteration() + 1
    return params | {"iterations" if name == "catboost" else "n_estimators": max(int(count), 1)}


def predictions(model, X, competition, *, for_metric=False):
    if competition.task == "classification" and (
        (for_metric and competition.metric in {"roc_auc", "log_loss"})
        or (not for_metric and competition.prediction_kind == "probability")
    ):
        return model.predict_proba(X)
    result = model.predict(X)
    if competition.metric == "log_rmse":
        return np.maximum(result, 1e-9)
    return np.maximum(result, 0) if competition.metric == "rmsle" else result


def train_candidate(request: dict) -> dict:
    started = time.monotonic()
    c = Competition.model_validate(request["competition"])
    config = RunConfig.model_validate(request["config"])
    candidate = request["candidate"]
    output = Path(request["output"])
    data = inspect_dataset(Path(config.data_dir), c, config)
    classes = data.report["classes"]
    # Use contiguous integer labels inside every classifier, including XGBoost.
    encoder = LabelEncoder().fit(data.y) if c.task == "classification" else None
    y = (
        pd.Series(encoder.transform(data.y))
        if encoder is not None
        else data.y.reset_index(drop=True)
    )
    metric_classes = list(range(len(classes))) if encoder is not None else None
    scores, train_scores, counts = [], [], []
    oof = None
    valid_mask = np.zeros(len(y), dtype=bool)
    with threadpool_limits(limits=config.n_jobs):
        for fit_indices, valid_indices in data.splits:
            params = early_stopping_params(
                candidate["model"],
                candidate["params"],
                data.X.iloc[fit_indices],
                y.iloc[fit_indices],
                c,
                config,
            )
            counts.append(
                params.get("iterations" if candidate["model"] == "catboost" else "n_estimators")
            )
            model = pipeline(candidate["model"], c, config, params)
            model.fit(data.X.iloc[fit_indices], y.iloc[fit_indices])
            prediction = predictions(model, data.X.iloc[valid_indices], c, for_metric=True)
            training = predictions(model, data.X.iloc[fit_indices], c, for_metric=True)
            scores.append(score(c.metric, y.iloc[valid_indices], prediction, metric_classes))
            train_scores.append(score(c.metric, y.iloc[fit_indices], training, metric_classes))
            if oof is None:
                shape = (len(y),) + np.asarray(prediction).shape[1:]
                oof = np.full(shape, np.nan)
            oof[valid_indices] = prediction
            valid_mask[valid_indices] = True
        final_params = dict(candidate["params"])
        if any(value is not None for value in counts):
            final_params["iterations" if candidate["model"] == "catboost" else "n_estimators"] = (
                int(np.median([v for v in counts if v is not None]))
            )
        final = pipeline(candidate["model"], c, config, final_params)
        final.fit(data.X, y)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.joblib"
    joblib.dump(
        {
            "pipeline": final,
            "encoder": encoder,
            "competition": c.model_dump(),
            "features": list(data.X.columns),
            "fingerprints": data.report["fingerprints"],
        },
        model_path,
    )
    oof_path = output / "oof.npz"
    np.savez_compressed(oof_path, prediction=oof, y=y.to_numpy(), valid_mask=valid_mask)
    importance_path = None
    fitted_pipeline = final.regressor_ if c.target_transform == "log1p" else final
    fitted = fitted_pipeline.named_steps["model"]
    importance = getattr(fitted, "feature_importances_", None)
    if importance is None and hasattr(fitted, "coef_"):
        importance = (
            np.abs(fitted.coef_).reshape(-1)
            if fitted.coef_.ndim == 1
            else np.abs(fitted.coef_).mean(axis=0)
        )
    if importance is not None:
        names = (
            fitted_pipeline.named_steps["preprocess"].named_steps["columns"].get_feature_names_out()
        )
        importance_path = str(output / "feature_importance.csv")
        pd.DataFrame({"feature": names, "importance": importance}).sort_values(
            "importance", ascending=False
        ).to_csv(importance_path, index=False)
    return Experiment(
        id=candidate["id"],
        model=candidate["model"],
        params=candidate["params"],
        effective_params=final_params,
        status="completed",
        score=float(np.mean(scores)),
        train_score=float(np.mean(train_scores)),
        std=float(np.std(scores)),
        fold_scores=scores,
        duration=time.monotonic() - started,
        checkpoint=str(model_path),
        oof_path=str(oof_path),
        importance_path=importance_path,
    ).model_dump()


def run_isolated(candidate, competition, config, output: Path, timeout: float) -> Experiment:
    output.mkdir(parents=True, exist_ok=True)
    request_path, result_path = output / "request.json", output / "result.json"
    write_json(
        request_path,
        {
            "candidate": candidate.model_dump(),
            "competition": competition.model_dump(),
            "config": config.model_dump(),
            "output": str(output),
        },
    )
    started = time.monotonic()
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(config.n_jobs)
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kaggle_slayer.ml.worker",
                    str(request_path),
                    str(result_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=max(timeout, 0.1),
                check=False,
            )
            if completed.returncode or not result_path.exists():
                raise RuntimeError(
                    f"Model worker exited with code {completed.returncode}; inspect worker.log"
                )
            return Experiment.model_validate_json(result_path.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            status, message = "timeout", f"Experiment exceeded {timeout:.1f}s; worker terminated"
        except (OSError, ValueError, RuntimeError) as exc:
            status, message = "failed", str(exc)
    return Experiment(
        id=candidate.id,
        model=candidate.model,
        params=candidate.params,
        status=status,
        error=message,
        duration=time.monotonic() - started,
    )


def load_predictions(checkpoint: Path, data, competition):
    # Load only locally generated trusted checkpoints. joblib can execute code.
    bundle = joblib.load(checkpoint)
    if bundle["fingerprints"] != data.report["fingerprints"]:
        raise ValueError("Dataset changed since model fitting; refuse stale model predictions")
    prediction = predictions(bundle["pipeline"], data.test_X, competition)
    if competition.prediction_kind == "label":
        prediction = bundle["encoder"].inverse_transform(np.asarray(prediction, dtype=int))
    return prediction
