"""Explicit training-loss contracts; evaluation always uses the competition metric."""


def objective_params(name, competition):
    if competition.task == "classification":
        return {}
    absolute = competition.metric == "mae"
    if name == "lightgbm":
        return {"objective": "regression_l1" if absolute else "regression_l2"}
    if name == "xgboost":
        return {
            "objective": "reg:absoluteerror" if absolute else "reg:squarederror",
            "eval_metric": "mae" if absolute else "rmse",
        }
    if name == "catboost":
        return {"loss_function": "MAE" if absolute else "RMSE"}
    if name == "hist_gradient":
        return {"loss": "absolute_error" if absolute else "squared_error"}
    if name == "random_forest":
        return {"criterion": "absolute_error" if absolute else "squared_error"}
    return {}


def training_contract(name, competition, params, options=None):
    options = options or {}
    if name == "ensemble":
        return {
            "model": name,
            "preprocessing": "independent_member_pipelines",
            "aggregation": "equal_weight_probabilities_or_original_target_values",
            "members": [
                training_contract(m["model"], competition, m["params"], m.get("pipeline"))
                | {"pipeline_config": m.get("pipeline", {})}
                for m in params["members"]
            ],
        }
    objective = objective_params(name, competition) | params
    scaled = options.get("scaling", "auto") == "standard" or (
        options.get("scaling", "auto") == "auto" and name == "linear"
    )
    indicators = options.get("missing_indicators", name != "linear")
    exact = (
        competition.task == "regression"
        and competition.target_transform == "none"
        and competition.metric in {"mae", "rmse", "r2"}
        and (name != "linear" or competition.metric != "mae")
    )
    return {
        "model": name,
        "preprocessing": "native_categories"
        if name == "catboost"
        else (
            ("scaled_one_hot" if scaled else "unscaled_one_hot")
            + ("_with_missing_indicators" if indicators else "")
        ),
        "feature_profile": competition.feature_profile,
        "metric": competition.metric,
        "target_transform": competition.target_transform,
        "objective_params": {
            k: v
            for k, v in objective.items()
            if k in {"objective", "loss_function", "criterion", "loss", "eval_metric"}
        },
        "metric_aligned_loss": exact,
        "note": "Aligned loss (RMSE/R2 use squared error)."
        if exact
        else (
            "Surrogate or transformed-target loss; selection uses the original competition metric. "
            "Classification uses a probabilistic surrogate rather than directly optimizing accuracy/F1/AUC."
        ),
    }
