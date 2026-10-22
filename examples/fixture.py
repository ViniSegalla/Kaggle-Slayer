"""Small, explicitly synthetic Titanic-shaped and house-price-shaped demo data."""

from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_slayer.storage.store import write_json


def make_fixture(directory: Path, task="classification", rows=240, seed=42):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total = rows + rows // 3
    if task == "classification":
        frame = pd.DataFrame(
            {
                "PassengerId": np.arange(1, total + 1),
                "Pclass": rng.integers(1, 4, total),
                "Sex": rng.choice(["male", "female"], total),
                "Age": np.clip(rng.normal(30, 14, total), 1, 80),
                "SibSp": rng.integers(0, 3, total),
                "Parch": rng.integers(0, 3, total),
                "Fare": rng.lognormal(3, 0.8, total),
                "Embarked": rng.choice(["S", "C", "Q"], total),
            }
        )
        signal = (frame.Sex == "female") * 2.6 - frame.Pclass * 0.6 - frame.Age * 0.01
        labels = (rng.random(total) < 1 / (1 + np.exp(-signal))).astype(int)
        target, identifier = "Survived", "PassengerId"
        frame.loc[rng.choice(total, total // 10, replace=False), "Age"] = np.nan
    else:
        frame = pd.DataFrame(
            {
                "Id": np.arange(1, total + 1),
                "OverallQual": rng.integers(1, 10, total),
                "GrLivArea": rng.integers(500, 3500, total),
                "Neighborhood": rng.choice(["North", "South", "East"], total),
                "GarageCars": rng.integers(0, 4, total),
                "YearBuilt": rng.integers(1920, 2020, total),
            }
        )
        labels = np.exp(
            10 + frame.OverallQual * 0.15 + frame.GrLivArea * 0.0003 + rng.normal(0, 0.15, total)
        )
        target, identifier = "SalePrice", "Id"
    train, test = frame.iloc[:rows].copy(), frame.iloc[rows:].copy()
    train[target] = labels[:rows]
    train.to_csv(directory / "train.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)
    pd.DataFrame({identifier: test[identifier], target: np.zeros(len(test), dtype=int)}).to_csv(
        directory / "sample_submission.csv", index=False
    )
    write_json(
        directory / "provenance.json",
        {
            "kind": "synthetic_fixture",
            "seed": seed,
            "description": "Generated data, not downloaded Kaggle data; no leaderboard claims",
        },
    )
    return directory
