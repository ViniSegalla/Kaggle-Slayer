"""Equivalent training requests share verified evidence across task and pipeline IDs."""

from contextlib import contextmanager
from pathlib import Path

import pytest
from test_swarm import setup

from kaggle_slayer.orchestration.swarm import SwarmRuntime
from kaggle_slayer.schemas.actions import Task, ToolAction
from kaggle_slayer.tools.training_tools import run_isolated


def request(runtime, task, ref, tool, arguments=None, depends_on=None):
    runtime.accept(
        task,
        ToolAction(
            kind="tool",
            action_id=ref,
            tool=tool,
            arguments=arguments or {},
            depends_on=depends_on or [],
            summary="Request measured evidence.",
        ),
    )


@contextmanager
def prepared_runtime(tmp_path, *, max_experiments):
    engine, state = setup(tmp_path, max_experiments=max_experiments, n_jobs=2)
    runtime = SwarmRuntime(engine, state)
    data = Task(id="data", role="data_validation", goal="Inspect and configure validation.")
    runtime.session.tasks[data.id] = data
    try:
        request(runtime, data, "inspect", "inspect_dataset")
        request(
            runtime,
            data,
            "validation",
            "configure_validation",
            {"method": "stratified"},
            ["inspect"],
        )
        runtime.drain_ready()
        assert runtime.session.validation
        yield runtime
    finally:
        runtime.pool.shutdown(wait=True)


def configure(runtime, owner, *, regularization=2.0):
    task = Task(id=owner, role="pipeline_modeling", goal=f"Measure pipeline for {owner}.")
    runtime.session.tasks[owner] = task
    ref = owner + "_pipeline"
    request(
        runtime,
        task,
        ref,
        "configure_pipeline",
        {"model": "linear", "params": {"C": regularization}},
    )
    runtime.drain_ready()
    assert runtime.session.actions[ref].status == "completed"
    return task, ref


def count_workers(monkeypatch):
    trained = []

    def worker(candidate, *args, **kwargs):
        trained.append(candidate.id)
        return run_isolated(candidate, *args, **kwargs)

    monkeypatch.setattr("kaggle_slayer.orchestration.swarm.run_isolated", worker)
    return trained


def test_cross_task_duplicates_share_inflight_and_completed_training(tmp_path, monkeypatch):
    trained = count_workers(monkeypatch)
    with prepared_runtime(tmp_path, max_experiments=1) as runtime:
        first, first_pipeline = configure(runtime, "first")
        duplicate, duplicate_pipeline = configure(runtime, "duplicate")
        assert first_pipeline != duplicate_pipeline
        request(runtime, first, "first_train", "run_experiment", {"pipeline_ref": first_pipeline})
        request(
            runtime,
            duplicate,
            "duplicate_train",
            "run_experiment",
            {"pipeline_ref": duplicate_pipeline},
        )
        checkpoints = []
        checkpoint = runtime.checkpoint

        def capture_inflight_alias(*args, **kwargs):
            alias = runtime.session.actions["duplicate_train"]
            source = runtime.session.actions["first_train"]
            if alias.reused_from and alias.status == "pending" and source.status == "reserved":
                checkpoints.append(alias.reused_from)
            return checkpoint(*args, **kwargs)

        monkeypatch.setattr(runtime, "checkpoint", capture_inflight_alias)
        runtime.drain_ready()
        assert checkpoints == ["first_train"]
        assert trained == ["first_train"]
        assert runtime.session.experiment_attempts == len(runtime.state.experiments) == 1
        canonical = runtime.state.experiments[0]

        # A rejected new confirmation must not seal selection when no slot remains.
        audit = Task(id="audit", role="evaluation", goal="Audit the canonical model.")
        runtime.session.tasks[audit.id] = audit
        request(runtime, audit, "rank", "evaluate")
        runtime.drain_ready()
        request(runtime, audit, "confirm", "confirm", {"parent_ref": canonical.id})
        runtime.drain_ready()
        assert runtime.session.actions["confirm"].result["error"] == "experiment_budget_exhausted"
        assert not runtime.session.confirmation_started
        alias = runtime.session.actions["duplicate_train"]
        assert alias.status == "completed" and alias.reused_from == "first_train"
        assert alias.result["experiment_ref"] == alias.result["artifact_ref"] == canonical.id
        assert alias.result["score"] == canonical.score
        runtime.verify_artifact(canonical.id)
        assert Path(canonical.checkpoint).is_file()
        assert Path(canonical.oof_path).is_file()

        # A fresh pipeline/task can reuse the completed worker even with no budget left.
        later, later_pipeline = configure(runtime, "later")
        request(runtime, later, "later_train", "run_experiment", {"pipeline_ref": later_pipeline})
        runtime.drain_ready()
        completed = runtime.session.actions["later_train"]
        assert completed.status == "completed"
        assert completed.reused_from == "first_train"
        assert completed.result["experiment_ref"] == canonical.id
        assert not completed.reservation
        assert trained == ["first_train"]
        assert runtime.session.experiment_attempts == len(runtime.state.experiments) == 1


def test_duplicate_training_alias_can_be_tuned_and_reused_after_quota(tmp_path, monkeypatch):
    trained = count_workers(monkeypatch)
    with prepared_runtime(tmp_path, max_experiments=2) as runtime:
        first, first_pipeline = configure(runtime, "first")
        duplicate, duplicate_pipeline = configure(runtime, "duplicate")
        request(runtime, first, "first_train", "run_experiment", {"pipeline_ref": first_pipeline})
        request(
            runtime,
            duplicate,
            "duplicate_train",
            "run_experiment",
            {"pipeline_ref": duplicate_pipeline},
        )
        runtime.drain_ready()
        tune = {"parent_ref": "duplicate_train", "params": {"C": 0.25}}
        request(runtime, duplicate, "tuned_alias", "tune_experiment", tune)
        runtime.drain_ready()
        tuned = runtime.session.actions["tuned_alias"]
        assert tuned.status == "completed", tuned.result
        assert tuned.result["experiment_ref"] == "tuned_alias"
        experiment = runtime.dispatcher.experiment("tuned_alias")
        assert experiment.parent_id == "first_train" and experiment.params["C"] == 0.25
        assert trained == ["first_train", "tuned_alias"]
        assert runtime.session.experiment_attempts == 2
        assert runtime.session.tuning_attempts == 1

        # Equivalent tuning through another parent reference also reuses canonical evidence.
        request(
            runtime,
            first,
            "tune_again",
            "tune_experiment",
            {"parent_ref": "first_train", "params": {"C": 0.25}},
        )
        runtime.drain_ready()
        repeated = runtime.session.actions["tune_again"]
        assert repeated.status == "completed", repeated.result
        assert repeated.result["experiment_ref"] == "tuned_alias"
        assert repeated.reused_from == "tuned_alias"
        assert runtime.dispatcher.experiment("tune_again").id == "tuned_alias"
        assert trained == ["first_train", "tuned_alias"]
        assert runtime.session.experiment_attempts == len(runtime.state.experiments) == 2
        assert runtime.session.tuning_attempts == 1


def test_inflight_alias_survives_interruption_and_adopts_source_worker(tmp_path, monkeypatch):
    trained = count_workers(monkeypatch)
    with prepared_runtime(tmp_path, max_experiments=1) as runtime:
        first, first_pipeline = configure(runtime, "first")
        duplicate, duplicate_pipeline = configure(runtime, "duplicate")
        request(runtime, first, "first_train", "run_experiment", {"pipeline_ref": first_pipeline})
        request(
            runtime,
            duplicate,
            "duplicate_train",
            "run_experiment",
            {"pipeline_ref": duplicate_pipeline},
        )

        def interrupt_before_commit(ref, result):
            raise KeyboardInterrupt

        monkeypatch.setattr(runtime, "commit_experiment", interrupt_before_commit)
        with pytest.raises(KeyboardInterrupt):
            runtime.drain_ready()
        engine, run_id = runtime.engine, runtime.state.run_id
        assert runtime.session.actions["first_train"].status == "reserved"
        assert runtime.session.actions["duplicate_train"].reused_from == "first_train"
        assert runtime.session.actions["duplicate_train"].status == "pending"
        assert not runtime.state.experiments
        assert (runtime.directory / "experiments" / "first_train" / "result.json").is_file()

    resumed = SwarmRuntime(engine, engine.store.load(run_id))
    try:
        resumed.recover()
        resumed.drain_ready()
        canonical = resumed.session.actions["first_train"]
        alias = resumed.session.actions["duplicate_train"]
        assert canonical.status == alias.status == "completed"
        assert alias.result["experiment_ref"] == alias.result["artifact_ref"] == "first_train"
        assert alias.result["reused_from"] == "first_train"
        resumed.verify_artifact("first_train")
        assert resumed.session.experiment_attempts == len(resumed.state.experiments) == 1
        assert trained == ["first_train"]
    finally:
        resumed.pool.shutdown(wait=True)


def test_inflight_alias_propagates_worker_failure_without_retry(tmp_path, monkeypatch):
    trained = []

    def failed_worker(candidate, *args, **kwargs):
        trained.append(candidate.id)
        return None

    monkeypatch.setattr("kaggle_slayer.orchestration.swarm.run_isolated", failed_worker)
    with prepared_runtime(tmp_path, max_experiments=1) as runtime:
        first, first_pipeline = configure(runtime, "first")
        duplicate, duplicate_pipeline = configure(runtime, "duplicate")
        request(runtime, first, "first_train", "run_experiment", {"pipeline_ref": first_pipeline})
        request(
            runtime,
            duplicate,
            "duplicate_train",
            "run_experiment",
            {"pipeline_ref": duplicate_pipeline},
        )
        runtime.drain_ready()
        source = runtime.session.actions["first_train"]
        alias = runtime.session.actions["duplicate_train"]
        assert source.status == alias.status == "failed"
        assert alias.result["error"] == "training_failed"
        assert alias.result["reused_from"] == "first_train"
        assert not alias.reservation
        assert trained == ["first_train"]
        assert runtime.session.experiment_attempts == len(runtime.state.experiments) == 1
        assert not runtime.session.artifacts.get("first_train")
