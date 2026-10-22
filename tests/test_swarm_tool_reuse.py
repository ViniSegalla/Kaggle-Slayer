"""Verified local operations can share evidence across independently delegated tasks."""

from copy import deepcopy
from pathlib import Path

import pytest
from test_swarm import setup

from kaggle_slayer.orchestration.swarm import SwarmRuntime
from kaggle_slayer.schemas.actions import ActionRecord, ToolAction
from kaggle_slayer.schemas.models import Candidate, RunConfig
from kaggle_slayer.tools.data_tools import inspect_dataset
from kaggle_slayer.tools.swarm_tools import ToolError


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    engine, state = setup(tmp_path_factory.mktemp("reuse"))
    result = engine.execute(state)
    assert result.status == "completed"
    return engine, result.run_id


@pytest.mark.parametrize("name", ["evaluate", "generate_predictions", "validate_predictions"])
def test_equivalent_tools_reuse_recorded_evidence_across_tasks(completed_run, name):
    engine, run_id = completed_run
    runtime = SwarmRuntime(engine, engine.store.load(run_id))
    try:
        action = ToolAction(kind="tool", action_id="another_request", tool=name, summary="Reuse.")
        args = runtime.dispatcher.validate(
            "evaluation" if name == "evaluate" else "submission", action
        )
        key = runtime.dispatcher.reuse_key(action, args)
        assert key
        source, result = runtime.dispatcher.reuse(action, key)
        assert result["reused_from"] == source
        assert runtime.session.actions[source].action.tool == name
        assert result != runtime.session.actions[source].result
        # A changed measured comparison lineage cannot reuse old rankings/predictions.
        runtime.session.validation["lineage"] = "new-comparison"
        if name == "evaluate":
            assert runtime.dispatcher.reuse(action, key) is None
        else:
            with pytest.raises(ToolError, match="invalid_experiment_ref"):
                runtime.dispatcher.reuse(action, key)
    finally:
        runtime.pool.shutdown(wait=True)


def test_reuse_requires_unchanged_dataset_and_artifacts(completed_run):
    engine, run_id = completed_run
    runtime = SwarmRuntime(engine, engine.store.load(run_id))
    action = ToolAction(kind="tool", action_id="eval_again", tool="evaluate", summary="Reuse.")
    args = runtime.dispatcher.validate("evaluation", action)
    try:
        key = runtime.dispatcher.reuse_key(action, args)
        dataset = Path(runtime.state.config.data_dir) / runtime.state.competition.train_file
        original = dataset.read_bytes()
        try:
            dataset.write_bytes(original + b"\n")
            with pytest.raises(ToolError, match="dataset_fingerprint_changed"):
                runtime.dispatcher.reuse(action, key)
        finally:
            dataset.write_bytes(original)
        model = Path(runtime.state.experiments[0].checkpoint)
        original = model.read_bytes()
        try:
            model.write_bytes(original + b"tampered")
            with pytest.raises(ToolError, match="artifact_integrity"):
                runtime.dispatcher.reuse(action, key)
        finally:
            model.write_bytes(original)
        predictions = Path(runtime.state.submission_path)
        original = predictions.read_bytes()
        generate = action.model_copy(update={"tool": "generate_predictions"})
        generate_args = runtime.dispatcher.validate("submission", generate)
        generate_key = runtime.dispatcher.reuse_key(generate, generate_args)
        try:
            predictions.write_bytes(original + b"tampered")
            with pytest.raises(ToolError, match="artifact_integrity|prediction_integrity"):
                runtime.dispatcher.reuse(generate, generate_key)
        finally:
            predictions.write_bytes(original)
        runtime.session.actions["future_training"] = ActionRecord(
            task_id="coordinator",
            action=ToolAction(
                kind="tool",
                action_id="future_training",
                tool="run_experiment",
                arguments={"pipeline_ref": "some_pipeline"},
                summary="Pending training.",
            ),
        )
        assert runtime.dispatcher.reuse(action, key) is None
    finally:
        runtime.pool.shutdown(wait=True)


def test_legacy_records_and_feedback_are_not_reused(completed_run):
    engine, run_id = completed_run
    runtime = SwarmRuntime(engine, engine.store.load(run_id))
    try:
        for record in runtime.session.actions.values():
            record.tool_snapshot = None
        action = ToolAction(
            kind="tool", action_id="eval_again", tool="evaluate", summary="Evaluate."
        )
        args = runtime.dispatcher.validate("evaluation", action)
        assert runtime.dispatcher.reuse(action, runtime.dispatcher.reuse_key(action, args)) is None
        action.tool = "retrieve_feedback"
        assert runtime.dispatcher.reuse_key(action, args) is None
    finally:
        runtime.pool.shutdown(wait=True)


@pytest.fixture
def training_runtime(completed_run):
    engine, run_id = completed_run
    runtime = SwarmRuntime(engine, engine.store.load(run_id))
    runtime.session.finalized = False
    runtime.data = inspect_dataset(
        Path(runtime.state.config.data_dir), runtime.state.competition, runtime.state.config
    )
    try:
        yield runtime
    finally:
        runtime.pool.shutdown(wait=True)


def recorded_training(runtime):
    ref, record = next(
        (ref, record)
        for ref, record in runtime.session.actions.items()
        if record.status == "completed" and record.reservation.get("candidate")
    )
    candidate = Candidate.model_validate(record.reservation["candidate"])
    config = RunConfig.model_validate(record.reservation["config"])
    return ref, record, candidate, config


def test_training_reuses_verified_measurement_across_fresh_ids_and_parent_aliases(training_runtime):
    runtime = training_runtime
    ref, record, candidate, config = recorded_training(runtime)
    another = candidate.model_copy(
        update={"id": "another_training", "rationale": "Independent request", "parent_id": ref}
    )
    key = runtime.dispatcher.training_key(another, config)
    assert key == record.tool_snapshot
    source, result = runtime.dispatcher.reuse_training(key)
    assert source == ref and result["experiment_ref"] == result["artifact_ref"] == ref
    assert result["reused_from"] == ref and "reused_from" not in record.result
    runtime.session.actions[another.id] = ActionRecord(
        task_id="coordinator",
        action=record.action.model_copy(update={"action_id": another.id}),
        status="completed",
        result=result,
        reused_from=ref,
    )
    # A downstream request may name the reused action rather than the canonical model ID.
    assert runtime.dispatcher.experiment(another.id).id == ref
    tune = ToolAction(
        kind="tool",
        action_id="tune_alias",
        tool="tune_experiment",
        arguments={"parent_ref": another.id, "params": {"C": 0.5}},
        summary="Tune the shared model.",
    )
    resolved = runtime.dispatcher.candidate(
        tune, runtime.dispatcher.validate("pipeline_modeling", tune), for_reuse=True
    )
    assert resolved.parent_id == ref and resolved.params["C"] == 0.5
    blend = ToolAction(
        kind="tool",
        action_id="self_blend",
        tool="ensemble",
        arguments={"members": [ref, another.id]},
        summary="Repeated canonical member.",
    )
    with pytest.raises(ToolError, match="ensemble_not_available"):
        runtime.dispatcher.candidate(
            blend, runtime.dispatcher.validate("pipeline_modeling", blend), for_reuse=True
        )


def test_training_identity_tracks_parameters_pipeline_seed_and_validation(training_runtime):
    runtime = training_runtime
    _, _, candidate, config = recorded_training(runtime)
    key = runtime.dispatcher.training_key(candidate, config)
    assert (
        runtime.dispatcher.training_key(
            candidate.model_copy(update={"params": candidate.params | {"C": 0.75}}), config
        )
        != key
    )
    assert (
        runtime.dispatcher.training_key(
            candidate.model_copy(update={"pipeline": candidate.pipeline | {"scaling": "none"}}),
            config,
        )
        != key
    )
    assert runtime.dispatcher.training_key(candidate, config.model_copy(update={"seed": 43})) != key
    assert runtime.dispatcher.training_key(candidate, config.model_copy(update={"folds": 4})) != key
    assert (
        runtime.dispatcher.training_key(
            candidate, config.model_copy(update={"max_dense_cells": config.max_dense_cells + 1})
        )
        != key
    )
    assert (
        runtime.dispatcher.training_key(
            candidate,
            config.model_copy(update={"submit": not config.submit, "max_experiments": 99}),
        )
        == key
    )
    assert (
        runtime.dispatcher.training_key(
            candidate.model_copy(update={"validation_role": "confirmation"}), config
        )
        != key
    )
    runtime.session.validation["lineage"] = "different-folds"
    assert runtime.dispatcher.reuse_training(key) is None


def test_training_identity_recognizes_equal_weight_ensemble_member_order(training_runtime):
    runtime = training_runtime
    _, _, candidate, config = recorded_training(runtime)
    members = [
        {
            "model": experiment.model,
            "params": experiment.params,
            "pipeline": experiment.training_contract["pipeline_config"],
        }
        for experiment in runtime.state.experiments
    ]
    ensemble = candidate.model_copy(update={"model": "ensemble", "params": {"members": members}})
    reversed_members = ensemble.model_copy(update={"params": {"members": list(reversed(members))}})
    assert runtime.dispatcher.training_key(ensemble, config) == runtime.dispatcher.training_key(
        reversed_members, config
    )


def test_training_reuse_rejects_changed_dataset_artifacts_and_unverified_receipts(training_runtime):
    runtime = training_runtime
    ref, record, candidate, config = recorded_training(runtime)
    key = runtime.dispatcher.training_key(candidate, config)
    dataset = Path(runtime.state.config.data_dir) / runtime.state.competition.train_file
    original = dataset.read_bytes()
    try:
        dataset.write_bytes(original + b"\n")
        with pytest.raises(ToolError, match="dataset_fingerprint_changed"):
            runtime.dispatcher.reuse_training(key)
    finally:
        dataset.write_bytes(original)
    model = Path(runtime.dispatcher.experiment(ref).checkpoint)
    original = model.read_bytes()
    try:
        model.write_bytes(original + b"tampered")
        with pytest.raises(ToolError, match="artifact_integrity"):
            runtime.dispatcher.reuse_training(key)
    finally:
        model.write_bytes(original)
    record.status = "failed"
    assert runtime.dispatcher.reuse_training(key) is None
    record.status, record.tool_snapshot = "completed", None
    # Historical receipts retain the immutable worker request and verified artifacts.
    source, _ = runtime.dispatcher.reuse_training(key)
    assert source == ref and record.tool_snapshot == key
    record.tool_snapshot = None
    record.reservation["competition"]["metric"] = "log_loss"
    assert runtime.dispatcher.reuse_training(key) is None


def test_legacy_evaluation_receipts_are_recomputed_for_eligibility_explanations(completed_run):
    engine, run_id = completed_run
    runtime = SwarmRuntime(engine, engine.store.load(run_id))
    try:
        action = ToolAction(
            kind="tool", action_id="explain_cached_rank", tool="evaluate", summary="Explain."
        )
        args = runtime.dispatcher.validate("evaluation", action)
        key = runtime.dispatcher.reuse_key(action, args)
        assert runtime.dispatcher.reuse(action, key)
        for record in runtime.session.actions.values():
            if record.action.kind == "tool" and record.action.tool == "evaluate":
                record.result.pop("eligibility", None)
        assert runtime.dispatcher.reuse(action, key) is None
    finally:
        runtime.pool.shutdown(wait=True)


def test_candidate_preview_does_not_charge_or_reject_training_quotas(training_runtime):
    runtime = training_runtime
    ref, _, _, _ = recorded_training(runtime)
    runtime.session.tuning_attempts = (
        runtime.state.config.max_iterations * runtime.state.config.optuna_trials
    )
    tune = ToolAction(
        kind="tool",
        action_id="tune_preview",
        tool="tune_experiment",
        arguments={"parent_ref": ref, "params": {"C": 0.5}},
        summary="Resolve before reuse.",
    )
    args = runtime.dispatcher.validate("pipeline_modeling", tune)
    assert runtime.dispatcher.candidate(tune, args, for_reuse=True).params["C"] == 0.5
    with pytest.raises(ToolError, match="tuning_budget"):
        runtime.dispatcher.candidate(tune, args)
    confirm = ToolAction(
        kind="tool",
        action_id="confirm_preview",
        tool="confirm",
        arguments={"parent_ref": ref},
        summary="Resolve diagnostic before reuse.",
    )
    args = runtime.dispatcher.validate("evaluation", confirm)
    assert not runtime.session.confirmation_started
    resolved = runtime.dispatcher.candidate(confirm, args, for_reuse=True)
    assert resolved.validation_role == "confirmation"
    assert not runtime.session.confirmation_started
    runtime.state.config.confirmation_top_k = 0
    assert runtime.dispatcher.candidate(confirm, args, for_reuse=True)
    with pytest.raises(ToolError, match="confirmation_budget"):
        runtime.dispatcher.candidate(confirm, args)


@pytest.mark.parametrize("metric", ["accuracy", "rmse"])
def test_evaluation_explains_below_threshold_ensembles_with_bounded_exclusions(
    training_runtime, metric
):
    runtime = training_runtime
    # The dispatcher reports the trusted evaluator's existing eligibility policy.
    runtime.state.competition.metric = metric
    baseline = max(runtime.state.experiments, key=lambda experiment: experiment.score)
    if metric == "rmse":
        baseline = min(runtime.state.experiments, key=lambda experiment: experiment.score)
    threshold = abs(baseline.score) * runtime.state.config.min_relative_improvement
    sign = 1 if metric == "accuracy" else -1
    for index in range(18):
        excluded = baseline.model_copy(
            update={
                "id": f"marginal_ensemble_{index}",
                "model": "ensemble",
                "score": baseline.score + sign * threshold / 2,
            }
        )
        runtime.state.experiments.append(excluded)
        runtime.session.artifacts[excluded.id] = deepcopy(runtime.session.artifacts[baseline.id])
    action = ToolAction(kind="tool", action_id="explain_rank", tool="evaluate", summary="Rank.")
    result = runtime.dispatcher.execute(action, runtime.dispatcher.validate("evaluation", action))
    eligibility = result["eligibility"]
    assert result["best_ref"] == baseline.id
    assert eligibility["complete"] and eligibility["ensemble_baseline_ref"] == baseline.id
    assert eligibility["required_ensemble_improvement"] == pytest.approx(threshold)
    assert eligibility["ensemble_comparison"] == "strictly_greater_than"
    assert eligibility["excluded_count"] == 18 and eligibility["excluded_truncated"]
    assert len(eligibility["excluded"]) == 16
    for excluded in eligibility["excluded"]:
        assert excluded["reason"] == "ensemble_below_improvement_threshold"
        assert excluded["signed_improvement"] == pytest.approx(threshold / 2)
