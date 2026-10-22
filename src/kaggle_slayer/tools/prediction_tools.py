"""Pure trusted prediction generation; also callable in a deadline-bound worker."""

import hashlib

import pandas as pd

from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.evaluation_tools import validate_submission
from kaggle_slayer.tools.training_tools import load_predictions


def generate_predictions(checkpoint, data, competition, output):
    values = load_predictions(checkpoint, data, competition)
    c = competition
    submission = (
        data.sample.copy()
        if data.sample is not None
        else pd.DataFrame(
            {c.id_column: data.test[c.id_column]} if c.id_column else {}, index=data.test.index
        )
    )
    columns = [column for column in submission if column != c.id_column]
    columns = columns or [c.target]
    if c.prediction_kind == "probability":
        if len(columns) == 1 and len(data.report["classes"]) == 2:
            submission[columns[0]] = values[:, -1]
        elif len(columns) == len(data.report["classes"]) and set(columns) == set(
            map(str, data.report["classes"])
        ):
            for index, label in enumerate(data.report["classes"]):
                submission[str(label)] = values[:, index]
        else:
            raise ValueError(
                "Probability submission columns must identify every class, or have one column for binary positive-class probability"
            )
    else:
        if len(columns) != 1:
            raise ValueError("Label/value submission requires one prediction column")
        submission[columns[0]] = values
    output.mkdir(exist_ok=True)
    path = output / "submission.csv"
    # Validate both in memory and after CSV serialization.
    report = validate_submission(submission, data.test, data.sample, c, data.report["classes"])
    submission.to_csv(path, index=False)
    reloaded = pd.read_csv(
        path, dtype={c.id_column: data.test[c.id_column].dtype} if c.id_column else None
    )
    if c.prediction_kind == "label" and isinstance(data.report["classes"][0], str):
        reloaded[columns[0]] = reloaded[columns[0]].astype(str)
    validate_submission(reloaded, data.test, data.sample, c, data.report["classes"])
    write_json(output / "validation.json", report)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "report": report,
    }
