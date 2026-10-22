"""Explicit model registry: provider suggestions cannot instantiate arbitrary code."""

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
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

from kaggle_slayer.ml.objectives import objective_params
from kaggle_slayer.ml.preprocessing import preprocessor


class NativeCategoricalFit:
    def fit(self, X, y=None, **kwargs):
        self.set_params(cat_features=list(X.select_dtypes(exclude="number").columns))
        return super().fit(X, y, **kwargs)


class NativeCatBoostClassifier(NativeCategoricalFit, CatBoostClassifier):
    pass


class NativeCatBoostRegressor(NativeCategoricalFit, CatBoostRegressor):
    pass


def estimator(name, task, seed, jobs, params):
    classification = task == "classification"
    if name == "ensemble":
        raise ValueError("Build ensembles with pipeline() so each member retains its preprocessing")
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
        cls = NativeCatBoostClassifier if classification else NativeCatBoostRegressor
        return cls(
            thread_count=jobs,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            **({"iterations": 200, "depth": 5, "learning_rate": 0.05} | params),
        )
    raise ValueError(f"Model not in registry: {name}")


def pipeline(name, competition, config, params, options=None):
    if name == "hist_gradient" and competition.validation in {"group", "time"}:
        params = params | {"early_stopping": False}
    if name == "ensemble":
        members = params.get("members", [])
        allowed = {"linear", "random_forest", "hist_gradient", "lightgbm", "xgboost", "catboost"}
        if len(members) != 2 or any(m["model"] not in allowed for m in members):
            raise ValueError("Ensemble requires exactly two approved model families")
        estimators = [
            (
                f"member_{i}",
                pipeline(m["model"], competition, config, m["params"], m.get("pipeline")),
            )
            for i, m in enumerate(members)
        ]
        return (
            VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
            if competition.task == "classification"
            else VotingRegressor(estimators=estimators, n_jobs=1)
        )
    model = Pipeline(
        [
            ("preprocess", preprocessor(competition.feature_profile, name, options)),
            (
                "model",
                estimator(
                    name,
                    competition.task,
                    config.seed,
                    config.n_jobs,
                    objective_params(name, competition) | params,
                ),
            ),
        ]
    )
    if competition.target_transform == "log1p":
        return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model
