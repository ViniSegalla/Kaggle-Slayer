import json

import numpy as np
import pandas as pd
import pytest

from kaggle_slayer.agents.evaluator import EvaluationAgent
from kaggle_slayer.agents.planner import CompetitionPlanner
from kaggle_slayer.ml.diagnostics import error_report
from kaggle_slayer.ml.models import pipeline
from kaggle_slayer.schemas import Competition, Experiment, RunConfig, RunState
from kaggle_slayer.storage.store import Store
from kaggle_slayer.tools.evaluation_tools import rank_experiments


def experiment(id, model, value, **kwargs):
    return Experiment(
        id=id,
        model=model,
        params={},
        status="completed",
        score=value,
        train_score=value,
        std=0.01,
        **kwargs,
    )


def test_native_categories_handle_missing_unseen_and_keep_identity():
    X = pd.DataFrame(
        {"number": np.arange(30, dtype=float), "category": [None, "missing:", "x"] * 10}
    )
    c = Competition(competition="native", task="regression", metric="mae")
    fitted = pipeline("catboost", c, RunConfig(), {"iterations": 15}).fit(X, np.arange(30.0))
    prep = fitted.named_steps["preprocess"]
    transformed = prep.transform(X)
    assert transformed.loc[0, "category"] != transformed.loc[1, "category"]
    assert fitted.named_steps["model"].get_cat_feature_indices() == [1]
    assert fitted.named_steps["model"].get_params()["loss_function"] == "MAE"
    assert np.isfinite(
        fitted.predict(pd.DataFrame({"number": [np.nan], "category": ["new"]}))
    ).all()


@pytest.mark.parametrize(
    "family,params",
    [
        ("lightgbm", {"n_estimators": 15}),
        ("xgboost", {"n_estimators": 15}),
        ("random_forest", {"n_estimators": 15}),
        ("hist_gradient", {"max_iter": 15}),
    ],
)
def test_absolute_error_training_handles_categorical_and_missing_inputs(family, params):
    X = pd.DataFrame({"number": np.r_[np.arange(29.0), np.nan], "category": [None, "a", "b"] * 10})
    c = Competition(competition="loss", task="regression", metric="mae")
    fitted = pipeline(family, c, RunConfig(n_jobs=1), params).fit(X, np.arange(30.0))
    assert np.isfinite(fitted.predict(X)).all()


def test_failed_confirmation_is_not_eligible(fixture_data, tmp_path):
    from kaggle_slayer.schemas.models import Candidate
    from kaggle_slayer.tools.training_tools import run_isolated

    _, c, config = fixture_data
    result = run_isolated(
        Candidate(
            id="failed-audit",
            model="unknown",
            params={},
            validation_role="confirmation",
            parent_id="primary",
            rationale="Test error recovery",
        ),
        c,
        config,
        tmp_path / "failed-audit",
        20,
    )
    assert result.status == "failed"
    assert result.validation_role == "confirmation"
    assert result.parent_id == "primary"
    assert rank_experiments([result], "accuracy") == []


def test_error_slices_only_use_held_out_rows():
    X = pd.DataFrame({"x": np.arange(40), "category": ["a"] * 20 + ["b"] * 20})
    y = np.zeros(40)
    predictions = np.r_[np.full(20, 1e9), np.ones(20)]
    report = error_report(X, y, predictions, np.arange(40) >= 20, "regression")
    assert report["overall_error"] == 1
    assert report["rows"] == 20
    assert all(row["error"] == 1 for row in report["slices"])


def test_confirmation_cannot_win_primary_ranking_or_change_selection(fixture_data):
    _, c, config = fixture_data
    primary = experiment("best", "lightgbm", 0.8)
    confirmation = experiment(
        "confirm-other",
        "catboost",
        0.99,
        validation_role="confirmation",
        parent_id="other",
        validation_seed=1051,
    )
    state = RunState(
        run_id="stability",
        competition=c,
        config=config,
        best_id="best",
        experiments=[
            primary,
            experiment("other", "catboost", 0.79),
            confirmation,
            experiment(
                "confirm-best",
                "lightgbm",
                0.7,
                validation_role="confirmation",
                parent_id="best",
                validation_seed=1051,
            ),
        ],
    )
    assert rank_experiments(state.experiments, "accuracy")[0].id == "best"
    EvaluationAgent(Store(config.artifact_root)).review_stability(state)
    assert state.best_id == "best"
    assert state.feedback["stability"]["same_winner"] is False
    assert state.feedback["stability"]["selection_changed"] is False


@pytest.mark.parametrize("metric,blend", [("accuracy", 0.80001), ("mae", 0.79999)])
def test_small_ensemble_gain_is_not_promoted(fixture_data, metric, blend):
    _, c, config = fixture_data
    c.metric = metric
    state = RunState(
        run_id="ensemble-gate",
        competition=c,
        config=config,
        experiments=[
            experiment("single", "lightgbm", 0.8),
            experiment("blend", "ensemble", blend),
        ],
    )
    EvaluationAgent(Store(config.artifact_root)).evaluate(state)
    assert state.best_id == "single"
    assert state.feedback["ensemble_below_improvement_threshold"] == ["blend"]


def test_small_budget_prioritizes_baselines_and_structural_splits(fixture_data):
    _, c, config = fixture_data
    config.models = RunConfig().models
    config.ensemble, config.confirmation_top_k = True, 2
    config.max_experiments = 5
    state = RunState(run_id="budgets", competition=c, config=config)
    planner = CompetitionPlanner(Store(config.artifact_root))
    assert planner.ensemble_slots(state) == 0
    assert planner.confirmation_slots(state) == 0
    config.max_experiments = 12
    assert planner.confirmation_slots(state) == 2
    c.validation, c.group_column = "group", "group"
    assert planner.confirmation_slots(state) == 0
    c.validation, c.group_column, c.time_column = "time", None, "date"
    assert planner.confirmation_slots(state) == 0


def test_planning_advice_is_not_an_execution_warning(fixture_data, monkeypatch):
    from kaggle_slayer.llm.provider import Advice, PlannerLLM
    from kaggle_slayer.tools.data_tools import inspect_dataset

    directory, c, config = fixture_data
    state = RunState(run_id="advice", competition=c, config=config)
    state.report = inspect_dataset(directory, c, config).report
    monkeypatch.setattr(
        PlannerLLM,
        "advise",
        lambda *args: (
            Advice(
                rationale="Compare models",
                priorities=["linear"],
                cautions=["No validation scores supplied yet"],
            ),
            None,
        ),
    )
    CompetitionPlanner(Store(config.artifact_root)).plan(state)
    assert not state.warnings
    assert state.metadata["planning_advice"] == ["No validation scores supplied yet"]
    assert json.dumps(state.model_dump())


def test_multi_trial_batches_keep_unique_trials_and_separate_studies(fixture_data):
    from kaggle_slayer.schemas import Plan

    _, c, config = fixture_data
    config.max_experiments, config.optuna_trials = 10, 6
    state = RunState(
        run_id="trial-batch",
        competition=c,
        config=config,
        plan=Plan(strategy="compare", queue=[], validation="auto", direction="maximize"),
        experiments=[
            experiment("baseline-linear", "linear", 0.6),
            experiment("baseline-random_forest", "random_forest", 0.7),
            experiment("baseline-lightgbm", "lightgbm", 0.8),
        ],
    )
    planner = CompetitionPlanner(Store(config.artifact_root))
    assert planner.feedback_plan(state)
    queue = state.plan.queue
    assert len(queue) == len({candidate.id for candidate in queue}) == 6
    assert {candidate.model for candidate in queue} == {"linear", "random_forest", "lightgbm"}
    for candidate in queue:
        planner.record_trial(state, candidate, experiment(candidate.id, candidate.model, 0.8))
    assert all(t.state.name == "COMPLETE" for t in planner.study(state, "lightgbm").trials)
