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

from kaggle_slayer.ml.diagnostics import error_report
from kaggle_slayer.ml.models import estimator, pipeline
from kaggle_slayer.ml.objectives import objective_params, training_contract
from kaggle_slayer.ml.preprocessing import preprocessor
from kaggle_slayer.ml.resource_limits import check_pipeline_resources
from kaggle_slayer.orchestration.progress import WorkerEvents, worker_event
from kaggle_slayer.schemas import Competition, Experiment, RunConfig
from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.data_tools import inspect_dataset
from kaggle_slayer.tools.evaluation_tools import score


def early_stopping_params(name, params, X, y, competition, config, event=None, options=None):
    """Use only an inner portion of the outer training fold for stopping decisions."""
    if name == "ensemble":
        return params | {
            "members": [
                {
                    "model": m["model"],
                    "params": early_stopping_params(
                        m["model"],
                        m["params"],
                        X,
                        y,
                        competition,
                        config,
                        event=event,
                        options=m.get("pipeline"),
                    ),
                    "pipeline": m.get("pipeline", {}),
                }
                for m in params["members"]
            ]
        }
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
    if event is not None:
        event()
    a, b = train_test_split(
        np.arange(len(X)),
        test_size=0.15,
        random_state=config.seed,
        stratify=y if competition.task == "classification" else None,
    )
    prep = preprocessor(competition.feature_profile, name, options)
    fit_X, stop_X = prep.fit_transform(X.iloc[a]), prep.transform(X.iloc[b])
    work_y = np.log1p(np.asarray(y)) if competition.target_transform == "log1p" else np.asarray(y)
    model = estimator(
        name,
        competition.task,
        config.seed,
        config.n_jobs,
        objective_params(name, competition) | params,
    )
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
    worker_event(output, "inspect")
    data = inspect_dataset(Path(config.data_dir), c, config)
    options = candidate.get("pipeline", {})
    resource_estimates = check_pipeline_resources(data, c, config, candidate)
    classes = data.report["classes"]
    # Use contiguous integer labels inside every classifier, including XGBoost.
    encoder = LabelEncoder().fit(data.y) if c.task == "classification" else None
    y = (
        pd.Series(encoder.transform(data.y))
        if encoder is not None
        else data.y.reset_index(drop=True)
    )
    metric_classes = list(range(len(classes))) if encoder is not None else None
    scores, train_scores, fold_params = [], [], []
    oof = None
    valid_mask = np.zeros(len(y), dtype=bool)
    with threadpool_limits(limits=config.n_jobs):
        for fold, (fit_indices, valid_indices) in enumerate(data.splits, 1):
            worker_event(output, "prepare_fold", fold=fold)
            params = early_stopping_params(
                candidate["model"],
                candidate["params"],
                data.X.iloc[fit_indices],
                y.iloc[fit_indices],
                c,
                config,
                event=lambda: worker_event(output, "early_stopping", fold=fold),
                options=options,
            )
            fold_params.append(params)
            model = pipeline(candidate["model"], c, config, params, options)
            worker_event(output, "fit_fold", fold=fold)
            model.fit(data.X.iloc[fit_indices], y.iloc[fit_indices])
            prediction = predictions(model, data.X.iloc[valid_indices], c, for_metric=True)
            training = predictions(model, data.X.iloc[fit_indices], c, for_metric=True)
            scores.append(score(c.metric, y.iloc[valid_indices], prediction, metric_classes))
            worker_event(output, "fold_complete", fold=fold, score=scores[-1])
            train_scores.append(score(c.metric, y.iloc[fit_indices], training, metric_classes))
            if oof is None:
                shape = (len(y),) + np.asarray(prediction).shape[1:]
                oof = np.full(shape, np.nan)
            oof[valid_indices] = prediction
            valid_mask[valid_indices] = True
        final_params = aggregate_params(candidate["model"], candidate["params"], fold_params)
        final = pipeline(candidate["model"], c, config, final_params, options)
        worker_event(output, "refit")
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
    np.savez_compressed(
        output / "splits.npz",
        **{
            f"{i}_{name}": rows
            for i, split in enumerate(data.splits)
            for name, rows in zip(("train", "validation"), split)
        },
    )
    diagnostics_path = output / "error_analysis.json"
    diagnostic = error_report(data.X, y, oof, valid_mask, c.task)
    diagnostic["source"] = f"{candidate.get('validation_role', 'primary')}_out_of_fold_predictions"
    write_json(diagnostics_path, diagnostic)
    importance_path = None
    fitted_pipeline = final.regressor_ if hasattr(final, "regressor_") else final
    fitted = (
        fitted_pipeline.named_steps["model"] if hasattr(fitted_pipeline, "named_steps") else final
    )
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
    worker_event(output, "saved")
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
        diagnostics_path=str(diagnostics_path),
        training_contract=training_contract(candidate["model"], c, final_params, options)
        | {
            "pipeline_config": options,
            "dataset_fingerprints": data.report["fingerprints"],
            "validation": {"method": c.validation, "folds": config.folds, "seed": config.seed},
            "effective_params": final_params,
            "resource_estimates": resource_estimates,
        },
        lineage=candidate.get("lineage"),
        validation_role=candidate.get("validation_role", "primary"),
        validation_seed=config.seed,
        parent_id=candidate.get("parent_id"),
    ).model_dump()


def aggregate_params(name, base, folds):
    if name == "ensemble":
        return base | {
            "members": [
                {
                    "model": member["model"],
                    "params": aggregate_params(
                        member["model"],
                        member["params"],
                        [p["members"][i]["params"] for p in folds],
                    ),
                    "pipeline": member.get("pipeline", {}),
                }
                for i, member in enumerate(base["members"])
            ]
        }
    key = "iterations" if name == "catboost" else "n_estimators"
    counts = [p[key] for p in folds if key in p]
    return base | ({key: int(np.median(counts))} if counts else {})


def run_isolated(
    candidate, competition, config, output: Path, timeout: float, progress=None, cancelled=None
) -> Experiment:
    output.mkdir(parents=True, exist_ok=True)
    request_path, result_path = output / "request.json", output / "result.json"
    # Unrecorded work may be retried after interruption; never replay stale output.
    result_path.unlink(missing_ok=True)
    (output / "progress.jsonl").unlink(missing_ok=True)
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
            command = [
                sys.executable,
                "-m",
                "kaggle_slayer.ml.worker",
                str(request_path),
                str(result_path),
            ]
            if progress is None and cancelled is None:
                returncode = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    timeout=max(timeout, 0.1),
                    check=False,
                ).returncode
            else:
                returncode = monitor_worker(
                    command,
                    log,
                    environment,
                    timeout,
                    output,
                    candidate,
                    competition,
                    config,
                    progress or (lambda message: None),
                    cancelled,
                )
            if returncode or not result_path.exists():
                raise RuntimeError(
                    f"Model worker exited with code {returncode}; inspect worker.log"
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
        validation_role=candidate.validation_role,
        validation_seed=config.seed,
        parent_id=candidate.parent_id,
        lineage=candidate.lineage,
    )


def monitor_worker(
    command,
    log,
    environment,
    timeout,
    output,
    candidate,
    competition,
    config,
    progress,
    cancelled=None,
):
    events = WorkerEvents(output / "progress.jsonl", config.folds, competition.metric, candidate.id)
    started = last_heartbeat = time.monotonic()
    deadline = started + max(timeout, 0.1)
    with subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=environment
    ) as process:
        try:
            while True:
                if cancelled and cancelled():
                    raise subprocess.TimeoutExpired(command, timeout)
                for message in events.read():
                    progress(message)
                elapsed = time.monotonic() - started
                if time.monotonic() - last_heartbeat >= 10:
                    progress(
                        f"{candidate.id} | still running: {events.stage} | elapsed={elapsed:.0f}s"
                    )
                    last_heartbeat = time.monotonic()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    code = process.wait(timeout=min(0.5, remaining))
                    for message in events.read():
                        progress(message)
                    return code
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
        except BaseException:
            process.kill()
            process.wait()
            raise


def load_predictions(checkpoint: Path, data, competition):
    # Load only locally generated trusted checkpoints. joblib can execute code.
    bundle = joblib.load(checkpoint)
    if bundle["fingerprints"] != data.report["fingerprints"]:
        raise ValueError("Dataset changed since model fitting; refuse stale model predictions")
    prediction = predictions(bundle["pipeline"], data.test_X, competition)
    if competition.prediction_kind == "label":
        prediction = bundle["encoder"].inverse_transform(np.asarray(prediction, dtype=int))
    return prediction
