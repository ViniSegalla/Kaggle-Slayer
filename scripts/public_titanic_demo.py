"""Real public Titanic data demo; never an official Kaggle competition execution.

uv run python scripts/public_titanic_demo.py
Downloads a public seaborn dataset, splits an untouched holdout, and runs the same agents.
"""

import hashlib
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd
from sklearn.model_selection import train_test_split

from kaggle_slayer.orchestration.engine import Engine
from kaggle_slayer.schemas import Competition, RunConfig
from kaggle_slayer.storage.store import Store, write_json
from kaggle_slayer.tools.evaluation_tools import score


def main():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    columns = {
        "survived": "Survived",
        "pclass": "Pclass",
        "sex": "Sex",
        "age": "Age",
        "sibsp": "SibSp",
        "parch": "Parch",
        "fare": "Fare",
        "embarked": "Embarked",
    }
    frame = raw[list(columns)].rename(columns=columns)
    frame.insert(0, "PassengerId", range(1, len(frame) + 1))
    fit, holdout = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame.Survived)
    fit, holdout = fit.sort_index(), holdout.sort_index()
    directory = Path("data/public-titanic-demo")
    directory.mkdir(parents=True, exist_ok=True)
    fit.to_csv(directory / "train.csv", index=False)
    holdout.drop(columns="Survived").to_csv(directory / "test.csv", index=False)
    holdout[["PassengerId", "Survived"]].assign(Survived=0).to_csv(
        directory / "sample_submission.csv", index=False
    )
    provenance = {
        "kind": "public_dataset_demo",
        "source_url": url,
        "source_sha256": hashlib.sha256(response.content).hexdigest(),
        "description": "Real public Titanic observations with a local stratified 80/20 holdout. Not an official Kaggle download or submission; identifiers are local row IDs.",
    }
    write_json(directory / "provenance.json", provenance)
    c = Competition(
        competition="public-titanic-demo",
        target="Survived",
        id_column="PassengerId",
        task="classification",
        metric="accuracy",
        prediction_kind="label",
        feature_profile="titanic",
        sample_file="sample_submission.csv",
    )
    config = RunConfig(data_dir=str(directory), folds=5, max_experiments=5, max_seconds=600)
    store = Store(config.artifact_root)
    engine = Engine(store, progress=print)
    state = engine.execute(engine.create(c, config))
    predictions = pd.read_csv(state.submission_path)
    evidence = {
        "run_id": state.run_id,
        "provenance": provenance,
        "train_rows": len(fit),
        "holdout_rows": len(holdout),
        "local_cv_accuracy": state.feedback["local_score"],
        "untouched_holdout_accuracy": score("accuracy", holdout.Survived, predictions.Survived),
        "note": "Holdout was evaluated once after model selection; never fed into agent planning.",
        "kaggle_submission": "not_attempted",
        "kaggle_score": None,
    }
    write_json(store.run_dir(state.run_id) / "public_holdout_evidence.json", evidence)
    print(evidence)


if __name__ == "__main__":
    main()
