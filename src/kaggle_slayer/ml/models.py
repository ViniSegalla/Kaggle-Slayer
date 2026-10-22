"""Explicit model registry: provider suggestions cannot instantiate arbitrary code."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from kaggle_slayer.ml.preprocessing import preprocessor


def estimator(name, task, seed, jobs, params):
    classification = task == "classification"
    if name == "ensemble":
        members = params.get("members", [])
        if len(members) != 2 or any(
            member["model"]
            not in {"linear", "random_forest", "hist_gradient", "lightgbm", "xgboost", "catboost"}
            for member in members
        ):
            raise ValueError("Ensemble requires exactly two approved model families")
        estimators = [
            (f"member_{i}", estimator(member["model"], task, seed, jobs, member["params"]))
            for i, member in enumerate(members)
        ]
        return (
            VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
            if classification
            else VotingRegressor(estimators=estimators, n_jobs=1)
        )
    if name == "linear":
        return (
            LogisticRegression(max_iter=1500, random_state=seed, **params)
            if classification
            else Ridge(**params)
        )
    if name == "random_forest":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(
            n_jobs=jobs,
            random_state=seed,
            **({"n_estimators": 120, "min_samples_leaf": 2} | params),
        )
    if name == "hist_gradient":
        cls = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
        return cls(
            random_state=seed,
            **(
                {
                    "max_iter": 160,
                    "early_stopping": True,
                    "validation_fraction": 0.15,
                    "n_iter_no_change": 15,
                }
                | params
            ),
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMClassifier if classification else LGBMRegressor
        return cls(
            n_jobs=jobs,
            random_state=seed,
            verbosity=-1,
            **(
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "num_leaves": 15,
                    "deterministic": True,
                    "force_col_wise": True,
                }
                | params
            ),
        )
    if name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if classification else XGBRegressor
        return cls(
            n_jobs=jobs,
            random_state=seed,
            **(
                {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "tree_method": "hist"}
                | params
            ),
        )
    if name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostClassifier if classification else CatBoostRegressor
        return cls(
            thread_count=jobs,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            **({"iterations": 200, "depth": 5, "learning_rate": 0.05} | params),
        )
    raise ValueError(f"Model not in registry: {name}")


def pipeline(name, competition, config, params):
    model = Pipeline(
        [
            ("preprocess", preprocessor(competition.feature_profile)),
            ("model", estimator(name, competition.task, config.seed, config.n_jobs, params)),
        ]
    )
    if competition.target_transform == "log1p":
        return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model
