"""Bound preprocessing allocations without fitting encoders or allocating dense matrices."""

import numpy as np
import pandas as pd


def estimate_input_size(X, test_X=None):
    train_rows, test_rows = len(X), len(test_X) if test_X is not None else 0
    width = len(X.columns)
    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "rows": max(train_rows, test_rows),
        "feature_count": width,
        "derived_features": 0,
        "encoded_width": width,
        "estimated_dense_cells": width * max(train_rows, test_rows),
    }


def _numeric_missing(values):
    values = pd.to_numeric(values, errors="coerce").astype(float)
    return bool((~np.isfinite(values)).any())


def _category_count(values):
    # Features stringifies nonnumeric values before encoding (True and 1 may
    # compare equal in pandas but become distinct categorical strings).
    strings = values.map(lambda value: str(value) if pd.notna(value) else np.nan)
    return int(strings.nunique(dropna=True)) + int(strings.isna().any())


def estimate_pipeline_size(X, test_X=None, *, profile="generic", model="linear", options=None):
    """Upper bound any training subset's width, including inner CV and full refit.

    Only training categories can become encoded columns. Frequency grouping may shrink
    a fitted encoder, but cannot safely shrink this bound across different CV subsets.
    Numeric derived columns are inspected one at a time, before any dense allocation.
    """
    options = options or {}
    source_numeric = list(X.select_dtypes(include="number").columns)
    operations = set(options.get("features", []))
    generated = {"__ks_missing_count"} if "missing_count" in operations else set()
    generated.update(
        f"__ks_{op}_{column}"
        for op in operations & {"numeric_square", "numeric_signed_log"}
        for column in source_numeric
    )
    if generated & set(X.columns):
        raise ValueError("Derived feature name collision")
    profile_numeric = {"FamilySize", "IsAlone", "FarePerPerson"} if profile == "titanic" else set()
    profile_generated = set()
    if profile == "titanic" and {"SibSp", "Parch"}.issubset(X.columns):
        profile_generated.update({"FamilySize", "IsAlone"})
        if "Fare" in X:
            profile_generated.add("FarePerPerson")
    names = set(X.columns) | generated | profile_generated
    estimate = estimate_input_size(X, test_X)
    estimate.update(feature_count=len(names), derived_features=len(names - set(X.columns)))
    if model == "catboost":
        width = len(names)
    else:
        numeric = set(source_numeric) | generated | (profile_numeric & names)
        width = len(numeric)
        category_cap = options.get("max_categories", 64)
        width += sum(
            min(_category_count(X[column]), category_cap)
            for column in X.columns
            if column not in numeric
        )
        if options.get("missing_indicators", model != "linear"):
            missing = sum(
                _numeric_missing(X[column])
                for column in X.columns
                if column in numeric and column not in profile_generated
            )
            for column in source_numeric:
                values = pd.to_numeric(X[column], errors="coerce").astype(float)
                if "numeric_square" in operations:
                    with np.errstate(over="ignore", invalid="ignore"):
                        missing += _numeric_missing(values**2)
                if "numeric_signed_log" in operations:
                    missing += _numeric_missing(values)
            if profile_generated:
                family = X["SibSp"] + X["Parch"] + 1
                missing += _numeric_missing(family)
                if "FarePerPerson" in profile_generated:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        missing += _numeric_missing(X["Fare"] / family.clip(lower=1))
            width += missing
    estimate["encoded_width"] = int(width)
    estimate["estimated_dense_cells"] = int(width * estimate["rows"])
    return estimate


def check_pipeline_resources(data, competition, config, candidate):
    """Check each independent ensemble member using the same contract as its preprocessor."""
    from kaggle_slayer.tools.data_tools import DatasetError

    members = candidate["params"]["members"] if candidate["model"] == "ensemble" else [candidate]
    estimates = []
    for member in members:
        estimate = estimate_pipeline_size(
            data.X,
            data.test_X,
            profile=competition.feature_profile,
            model=member["model"],
            options=member.get("pipeline", {}),
        )
        details = estimate | {
            "max_features": config.max_features,
            "max_dense_cells": config.max_dense_cells,
        }
        if estimate["feature_count"] > config.max_features:
            raise DatasetError(
                "pipeline_feature_limit", "Effective pipeline exceeds max_features.", details
            )
        if estimate["estimated_dense_cells"] > config.max_dense_cells:
            raise DatasetError(
                "pipeline_dense_limit", "Effective pipeline exceeds max_dense_cells.", details
            )
        estimates.append(estimate)
    return estimates
