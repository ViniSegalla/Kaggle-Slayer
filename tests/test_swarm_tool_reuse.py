"""Verified local operations can share evidence across independently delegated tasks."""

from pathlib import Path

import pytest
from test_swarm import setup

from kaggle_slayer.orchestration.swarm import SwarmRuntime
from kaggle_slayer.schemas.actions import ActionRecord, ToolAction
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
