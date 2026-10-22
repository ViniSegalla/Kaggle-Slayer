"""Resource guard failures keep useful counts without exposing raw exception values."""

import json
from unittest.mock import Mock

import pytest
from test_swarm import setup

from kaggle_slayer.orchestration.swarm import StopRun, SwarmRuntime
from kaggle_slayer.schemas.actions import ActionRecord, Task, ToolAction
from kaggle_slayer.schemas.models import Candidate, Experiment
from kaggle_slayer.tools.data_tools import DatasetError, inspect_dataset
from kaggle_slayer.tools.runtime_worker import RuntimeOperationError


def guarded_error(code):
    return DatasetError(
        code,
        "private-credential-and-raw-row-value",
        {
            "encoded_width": 62,
            "estimated_dense_cells": 43377370,
            "max_dense_cells": 40000000,
            "train_rows": 699635,
            "private-column-name": 123,
            "test_rows": "private-credential-and-raw-row-value",
            "max_features": True,
            "feature_count": -1,
            "derived_features": 2**64,
        },
    )


@pytest.mark.parametrize("code", ["dataset_dense_limit", "pipeline_dense_limit"])
def test_training_worker_preserves_controlled_resource_diagnostics(
    fixture_data, tmp_path, monkeypatch, capsys, code
):
    from kaggle_slayer.ml.worker import main

    _, competition, config = fixture_data
    candidate = Candidate(id="guard", model="linear", rationale="Check limits")
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(
        json.dumps(
            {
                "candidate": candidate.model_dump(),
                "competition": competition.model_dump(),
                "config": config.model_dump(),
                "output": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kaggle_slayer.ml.worker.train_candidate", Mock(side_effect=guarded_error(code))
    )
    monkeypatch.setattr("sys.argv", ["worker", str(request), str(result)])
    main()
    experiment = Experiment.model_validate_json(result.read_text())
    assert experiment.status == "failed"
    assert experiment.error_code == code
    assert experiment.error_details == {
        "encoded_width": 62,
        "estimated_dense_cells": 43377370,
        "max_dense_cells": 40000000,
        "train_rows": 699635,
    }
    assert "private" not in result.read_text() + capsys.readouterr().err


def reserved_runtime(tmp_path):
    engine, state = setup(tmp_path)
    engine.live.emit = Mock()
    runtime = SwarmRuntime(engine, state)
    runtime.session.tasks["model"] = Task(
        id="model", role="pipeline_modeling", parent="coordinator", goal="Fit candidate"
    )
    candidate = Candidate(
        id="guard", model="linear", params={}, rationale="Check limits", lineage="frozen"
    )
    runtime.session.actions[candidate.id] = ActionRecord(
        task_id="model",
        action=ToolAction(
            kind="tool", action_id=candidate.id, tool="run_experiment", summary="Fit candidate"
        ),
        status="reserved",
        reservation={"candidate": candidate.model_dump()},
    )
    return runtime, candidate


@pytest.mark.parametrize(
    "code,stops",
    [
        ("dataset_dense_limit", True),
        ("dataset_feature_limit", True),
        ("pipeline_dense_limit", False),
    ],
)
def test_scheduler_exposes_guard_counts_and_stops_only_immutable_dataset_failure(
    tmp_path, code, stops
):
    runtime, candidate = reserved_runtime(tmp_path)
    runtime.session.experiment_attempts = 1
    experiment = Experiment(
        id=candidate.id,
        model=candidate.model,
        params=candidate.params,
        lineage=candidate.lineage,
        status="failed",
        error="private-worker-exception",
        error_code=code,
        error_details={"estimated_dense_cells": 43377370, "max_dense_cells": 40000000},
    )
    try:
        if stops:
            with pytest.raises(StopRun, match=code):
                runtime.commit_experiment(candidate.id, experiment)
            assert code in runtime.state.error
        else:
            runtime.commit_experiment(candidate.id, experiment)
            assert runtime.state.error is None
        observation = runtime.session.actions[candidate.id].result
        assert observation["error"] == code
        assert observation["details"]["estimated_dense_cells"] == 43377370
        assert "retry" in observation["action"] if not stops else "new run" in observation["action"]
        assert "private" not in json.dumps(observation)
        assert runtime.session.experiment_attempts == 1
        console = " ".join(call.args[2] for call in runtime.engine.live.emit.call_args_list)
        assert code in console and "estimated_dense_cells=43377370" in console
        assert "private" not in console
    finally:
        runtime.pool.shutdown()


@pytest.mark.parametrize("code", ["pipeline_dense_limit", "dataset_dense_limit"])
def test_preflight_resource_limit_does_not_reserve_an_experiment(tmp_path, monkeypatch, code):
    runtime, candidate = reserved_runtime(tmp_path)
    record = runtime.session.actions[candidate.id]
    record.status, record.reservation = "pending", {}
    monkeypatch.setattr(runtime.dispatcher, "validate", Mock(return_value={}))
    monkeypatch.setattr(runtime.dispatcher, "candidate", Mock(side_effect=guarded_error(code)))
    try:
        if code == "dataset_dense_limit":
            with pytest.raises(StopRun, match=code):
                runtime.execute_ready()
            assert code in runtime.state.error
        else:
            runtime.execute_ready()
        assert runtime.session.experiment_attempts == 0
        assert not runtime.state.experiments
        assert record.status == "failed" and record.reservation == {}
        assert record.result["error"] == code
        assert "private" not in json.dumps(record.result)
        assert runtime.session.termination_reason is None
    finally:
        runtime.pool.shutdown()


def test_unknown_worker_error_remains_generic_in_agent_observation(tmp_path):
    runtime, candidate = reserved_runtime(tmp_path)
    result = Experiment(
        id=candidate.id,
        model=candidate.model,
        params=candidate.params,
        lineage=candidate.lineage,
        status="failed",
        error="private-worker-exception",
        error_code="private-unrecognized-code",
        error_details={"private-column-name": 123},
    )
    try:
        runtime.commit_experiment(candidate.id, result)
        observation = runtime.session.actions[candidate.id].result
        assert observation["error"] == "training_failed"
        assert "private" not in json.dumps(observation)
        assert "details" not in observation
    finally:
        runtime.pool.shutdown()


def test_diagnostic_sanitizes_details_even_when_exception_code_is_unrecognized():
    diagnostic = RuntimeOperationError("private-unrecognized-code", guarded_error("unused").details)
    assert diagnostic.code == "bounded_operation_failed"
    assert diagnostic.details == {}
    assert "private" not in json.dumps(diagnostic.diagnostic()) + str(diagnostic)


def test_real_preflight_can_reduce_pipeline_and_train_with_one_experiment_slot(tmp_path):
    engine, state = setup(tmp_path, max_dense_cells=1000, max_experiments=1)
    runtime = SwarmRuntime(engine, state, engine.llm)
    task = Task(id="model", role="pipeline_modeling", goal="Fit within resource limits")
    runtime.session.tasks[task.id] = task
    from pathlib import Path

    runtime.data = inspect_dataset(Path(state.config.data_dir), state.competition, state.config)
    state.report = runtime.data.report
    runtime.session.validation = {"lineage": "frozen", "method": "stratified"}
    try:
        for ref, features in [
            ("large", ["missing_count", "numeric_square", "numeric_signed_log"]),
            ("small", []),
        ]:
            runtime.accept(
                task,
                ToolAction(
                    kind="tool",
                    action_id=ref,
                    tool="configure_pipeline",
                    summary="Configure",
                    arguments={"model": "linear", "pipeline": {"features": features}},
                ),
            )
            runtime.accept(
                task,
                ToolAction(
                    kind="tool",
                    action_id=ref + "_run",
                    tool="run_experiment",
                    summary="Train",
                    arguments={"pipeline_ref": ref},
                    depends_on=[ref],
                ),
            )
            runtime.drain_ready()
            if ref == "large":
                failure = runtime.session.actions["large_run"].result
                assert failure["error"] == "pipeline_dense_limit"
                assert failure["details"]["estimated_dense_cells"] > 1000
                assert runtime.session.experiment_attempts == 0
                assert not state.experiments
        assert runtime.session.actions["small_run"].status == "completed"
        assert runtime.session.experiment_attempts == 1
        assert len(state.experiments) == 1 and state.experiments[0].score is not None
    finally:
        runtime.pool.shutdown()
