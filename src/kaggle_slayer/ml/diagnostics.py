"""Aggregate held-out errors into bounded, descriptive feature slices."""

import numpy as np
import pandas as pd


def error_report(X, y, prediction, valid_mask, task):
    frame = X.loc[valid_mask].reset_index(drop=True)
    actual = np.asarray(y)[valid_mask]
    predicted = np.asarray(prediction)[valid_mask]
    if task == "classification":
        labels = predicted.argmax(axis=1) if predicted.ndim == 2 else predicted
        errors = (labels != actual).astype(float)
        measure = "misclassification_rate"
    else:
        errors = np.abs(actual - predicted)
        measure = "mean_absolute_error_original_units"
    rows = []
    minimum = max(10, int(len(frame) * 0.02))
    # Prefer fields with missingness; cap reporting to avoid huge metadata.
    columns = sorted(frame, key=lambda c: -frame[c].isna().mean())[:12]
    for column in columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            finite = values.replace([np.inf, -np.inf], np.nan)
            groups = pd.qcut(finite, 4, duplicates="drop").astype(str)
            groups = groups.where(finite.notna(), "<missing>")
        else:
            values = values.map(lambda v: "value:" + str(v) if pd.notna(v) else "<missing>")
            top = values.value_counts().head(5).index
            groups = values.where(values.isin(top), "<other>")
        for label in groups.unique():
            mask = (groups == label).to_numpy()
            if mask.sum() >= minimum:
                rows.append(
                    {
                        "feature": column,
                        "slice": str(label),
                        "rows": int(mask.sum()),
                        "error": float(errors[mask].mean()),
                    }
                )
    return {
        "source": "primary_out_of_fold_predictions",
        "measure": measure,
        "rows": len(actual),
        "overall_error": float(errors.mean()),
        "slices": sorted(rows, key=lambda r: -r["error"])[:30],
        "note": "Exploratory overlapping slices, not significance tests or independent holdouts.",
    }
