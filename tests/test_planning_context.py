import json
from unittest.mock import Mock

import pytest

from kaggle_slayer.agents.planner import CompetitionPlanner
from kaggle_slayer.llm.prompts import MAX_FEATURES, PROMPT_VERSION, build_initial_context
from kaggle_slayer.llm.provider import Advice
from kaggle_slayer.schemas import Competition, RunConfig, RunState
from kaggle_slayer.storage.store import Store
from kaggle_slayer.tools.data_tools import inspect_dataset


def test_planner_sends_actual_aggregates_not_labels_rows_or_paths(fixture_data, monkeypatch):
    directory, competition, config = fixture_data
    report = inspect_dataset(directory, competition, config).report
    report["target_distribution"] = {"private-class-a": 60, "private-class-b": 30}
    report["raw_rows"] = [{"secret": "private-row-value"}]
    report["classes"] = ["private-class-a", "private-class-b"]
    report["fingerprints"] = {"train.csv": "private-fingerprint"}
    advice = Mock(
        return_value=(
            Advice(
                rationale="Observed: imbalance. Hypothesis: compare trees.", priorities=["lightgbm"]
            ),
            None,
        )
    )
    llm = Mock(advise=advice, last_diagnostic=None)
    monkeypatch.setattr("kaggle_slayer.agents.planner.PlannerLLM", lambda: llm)
    state = RunState(run_id="context-probe", competition=competition, config=config, report=report)
    CompetitionPlanner(Store(config.artifact_root)).plan(state)
    advice.assert_called_once()
    context = advice.call_args.args[0]
    encoded = json.dumps(context, allow_nan=False)
    for forbidden in (
        "private-class-a",
        "private-class-b",
        "private-row-value",
        "private-fingerprint",
        str(directory),
        config.artifact_root,
    ):
        assert forbidden not in encoded
    assert context["target_summary"]["majority_fraction"] == 2 / 3
    assert context["target_summary"]["minority_count"] == 30
    assert context["quality"]["duplicate_rows"] == report["duplicate_rows"]
    assert context["quality"]["train_rows"] == 90
    assert context["budget"]["max_experiments"] == config.max_experiments
    assert context["validation"]["effective"] == "stratified"
    assert "fold_sizes" not in context["validation"]
    assert context["validation"]["folds"] == config.folds
    assert context["validation_evidence"] == {"status": "not_available_before_training"}
    assert state.metadata["planning_prompt_version"] == PROMPT_VERSION
    # Advice still only orders allowed families; Python forces the simple baseline first.
    assert [c.model for c in state.plan.queue] == ["linear", "lightgbm", "random_forest"]


def test_context_caps_wide_schema_and_long_strings_without_changing_report():
    dtypes = {f"column-{i}-" + "x" * 200: "object" for i in range(200)}
    report = {
        "dtypes": dtypes,
        "missing_fraction": {c: 0 for c in dtypes},
        "categorical_cardinality": {c: i + 1 for i, c in enumerate(dtypes)},
        "warnings": ["x" * 1000] * 100,
        "target_distribution": {f"private-class-{i}": i + 1 for i in range(100)},
    }
    c = Competition(competition="wide", task="classification", metric="accuracy")
    context = build_initial_context(c, RunConfig(), report)
    features = context["quality"]["features"]
    assert len(features) == MAX_FEATURES
    assert all(len(f["name"]) <= 64 for f in features)
    assert any(f["categorical_cardinality"] == 200 for f in features)
    assert context["quality"]["features_truncated"]
    assert context["quality"]["warnings_truncated"]
    assert len(context["quality"]["warnings"]) == 3
    assert all(len(w) <= 120 for w in context["quality"]["warnings"])
    assert context["target_summary"]["class_count"] == 100
    assert len(context["target_summary"]["largest_class_counts"]) == 3
    assert context["target_summary"]["counts_truncated"]
    assert "private-class-" not in json.dumps(context)
    assert len(json.dumps(context, allow_nan=False)) < 5000
    assert context == build_initial_context(c, RunConfig(), report)
    assert len(report["dtypes"]) == 200 and len(report["warnings"][0]) == 1000


def test_compact_context_preserves_aggregate_risks_but_not_repeated_model_notes():
    c = Competition(competition="compact", task="regression", metric="mae")
    report = {
        "dtypes": {f"column{i}": "object" for i in range(80)},
        "missing_fraction": {"column0": 0.95},
        "categorical_cardinality": {"column79": 1000},
        "duplicate_rows": 7,
    }
    result = build_initial_context(c, RunConfig(), report)
    assert result["quality"]["max_missing_fraction"] == 0.95
    assert result["quality"]["max_categorical_cardinality"] == 1000
    assert {"column0", "column79"} <= {f["name"] for f in result["quality"]["features"]}
    for capability in result["model_capabilities"].values():
        assert set(capability) == {"preprocessing", "objective_params", "metric_aligned_loss"}
    assert "seed" not in result["validation"]


@pytest.mark.parametrize(
    "validation,group,time",
    [("auto", None, None), ("group", "household", None), ("time", None, "event_date")],
)
def test_regression_context_preserves_validation_transform_and_model_contracts(
    validation, group, time
):
    c = Competition(
        competition="regression",
        task="regression",
        metric="mae",
        validation=validation,
        group_column=group,
        time_column=time,
        target_transform="log1p",
    )
    context = build_initial_context(
        c,
        RunConfig(),
        {
            "target_distribution": {
                "count": 100,
                "mean": 2500.0,
                "std": float("nan"),
                "raw_samples": ["private-target-value"],
            }
        },
    )
    assert context["metric_direction"] == "minimize"
    assert context["target_transform"] == "log1p"
    assert context["validation"]["effective"] == ("kfold" if validation == "auto" else validation)
    assert context["validation"]["group_column"] == group
    assert context["validation"]["time_column"] == time
    assert context["target_summary"] == {"count": 100, "mean": 2500.0, "std": None}
    assert context["model_capabilities"]["catboost"]["preprocessing"] == "native_categories"
    assert not context["model_capabilities"]["catboost"]["metric_aligned_loss"]
    assert "private-target-value" not in json.dumps(context, allow_nan=False)


def test_minimal_legacy_report_has_explicit_unknowns():
    context = build_initial_context(
        Competition(competition="old", task="classification", metric="accuracy"),
        RunConfig(),
        {"train_rows": 90},
    )
    assert context["quality"]["train_rows"] == 90
    assert context["quality"]["duplicate_rows"] is None
    assert context["quality"]["numeric_feature_count"] is None
    assert context["target_summary"]["class_count"] is None
    assert context["quality"]["features"] == []
