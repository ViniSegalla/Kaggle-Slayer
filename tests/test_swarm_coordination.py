"""Feedback-driven coordination drains bounded, reusable work without re-delegation."""

import json

import pytest
from test_swarm import setup

from examples.scripted_swarm import ScriptedTeam
from kaggle_slayer.orchestration.swarm import SwarmRuntime
from kaggle_slayer.schemas.actions import (
    ActionRecord,
    AgentResponse,
    DelegateAction,
    FinishAction,
    Task,
    ToolAction,
    WaitAction,
)
from kaggle_slayer.tools.swarm_tools import ToolError


class DuplicateTeam(ScriptedTeam):
    """Repeat intentions under fresh IDs, and request finish before children finish."""

    def __init__(self):
        super().__init__(refine=False)
        self.finishing = False

    def decide(self, role, context, **limits):
        if role == "coordinator" and self.finishing:
            raise AssertionError("A queued finish must drain without new coordinator turns")
        response, diagnostic = super().decide(role, context, **limits)
        action = response.actions[0]
        if (
            role == "coordinator"
            and action.kind == "delegate"
            and action.action_id in {"audit", "results"}
        ):
            values = [
                action.model_dump(),
                action.model_dump() | {"action_id": action.action_id + "_again"},
            ]
            if action.action_id == "results":
                self.finishing = True
                values.append(
                    {
                        "kind": "finish",
                        "action_id": "finish_when_drained",
                        "summary": "Finish after verified artifacts.",
                    }
                )
            response = AgentResponse.model_validate({"version": "swarm-v1", "actions": values})
        return response, diagnostic


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_duplicate_delegations_finish_barrier_and_resume_train_real_artifacts(
    tmp_path, monkeypatch, task
):
    provider = DuplicateTeam()
    engine, state = setup(tmp_path, task=task, provider=provider)
    checkpoint = SwarmRuntime.checkpoint
    interrupted = False

    def interrupt_after_prediction_validation(runtime, *args, **kwargs):
        nonlocal interrupted
        checkpoint(runtime, *args, **kwargs)
        if (
            task == "regression"
            and not interrupted
            and runtime.state.metadata.get("swarm_submission_validated")
            and runtime.session.actions["finish_when_drained"].status == "pending"
        ):
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(SwarmRuntime, "checkpoint", interrupt_after_prediction_validation)
    result = engine.execute(state)
    if task == "regression":
        assert result.status == "interrupted"
        assert result.swarm["actions"]["finish_when_drained"]["status"] == "pending"
        result = engine.execute(engine.store.load(result.run_id))
    assert result.status == "completed", result.swarm
    assert result.swarm["termination_reason"] == "coordinator_finished"
    assert len(result.experiments) == 1 and result.experiments[0].score is not None
    assert result.metadata["swarm_submission_validated"] == result.metadata["submission_sha256"]
    for name in ("audit", "results"):
        assert name + "_again" not in result.swarm["tasks"]
        record = result.swarm["actions"][name + "_again"]
        assert record["status"] == "completed" and record["reused_from"] == name
    assert result.swarm["requests"] < 30
    assert not engine.client.submit.called
    # Durable evidence survives rolling-window eviction and carries no local paths.
    runtime = SwarmRuntime(engine, engine.store.load(result.run_id), provider)
    try:
        runtime.session.observations.clear()
        for index in range(40):
            action = WaitAction(
                kind="wait", action_id=f"old_wait_{index}", depends_on=["audit"], summary="Old wait"
            )
            runtime.session.actions[action.action_id] = ActionRecord(
                task_id="coordinator", action=action, status="completed"
            )
        context = runtime.context(runtime.session.tasks["coordinator"])
        assert "results-validate_predictions" not in context["evidence"]
        assert set(context["progress"]["receipts"]) == {"evaluation", "predictions", "validation"}
        assert context["progress"]["ready_to_finish"]
        assert str(tmp_path) not in json.dumps(context)
        # A completed delegation with identical inputs also resolves to its receipt.
        later = runtime.session.actions["audit"].action.model_copy(
            update={"action_id": "audit_later"}
        )
        runtime.accept(runtime.session.tasks["coordinator"], later)
        runtime.session.termination_reason = None
        runtime.drain_ready()
        assert runtime.session.actions["audit_later"].result["reused_from"] == "audit"
        assert "audit_later" not in runtime.session.tasks
    finally:
        runtime.pool.shutdown()


def test_wait_suppresses_coordinator_churn_and_ready_tasks_are_fair(tmp_path):
    class FairTeam(ScriptedTeam):
        def decide(self, role, context, **limits):
            if role != "coordinator":
                return super().decide(role, context, **limits)
            with self.lock:
                self.calls.append((role, context))
            if len(context["tasks"]) == 1:
                actions = [
                    {
                        "kind": "delegate",
                        "action_id": name,
                        "recipient": "data_validation",
                        "task": f"Inspect scope {name}",
                        "summary": "Inspect",
                    }
                    for name in ("one", "two", "three")
                ]
                actions.append(
                    {
                        "kind": "wait",
                        "action_id": "await_all",
                        "depends_on": ["one", "two", "three"],
                        "summary": "Wait for all scopes",
                    }
                )
            else:
                assert all(
                    t["status"] == "completed" for t in context["tasks"] if t["id"] != "coordinator"
                )
                actions = [
                    {
                        "kind": "finish",
                        "action_id": "done",
                        "summary": "Inspection only",
                        "outcome": "incomplete",
                    }
                ]
            return {"version": "swarm-v1", "actions": actions}, {
                "provider": "fake",
                "status": "success",
                "code": "ok",
            }

    provider = FairTeam(refine=False)
    engine, state = setup(tmp_path, provider=provider)
    result = engine.execute(state)
    assert result.swarm["termination_reason"] == "coordinator_incomplete"
    assert len([role for role, _ in provider.calls if role == "coordinator"]) == 2
    task_order = [context["task_id"] for role, context in provider.calls if role != "coordinator"]
    assert set(task_order[:2]) == {"one", "two"}
    assert task_order[2] == "three"


def test_cyclic_wait_returns_observation_instead_of_silent_spinning(tmp_path):
    class CycleTeam(ScriptedTeam):
        def decide(self, role, context, **limits):
            if not context["observations"]:
                action = {
                    "kind": "wait",
                    "action_id": "cycle",
                    "depends_on": ["coordinator"],
                    "summary": "Wait",
                }
            else:
                assert context["observations"][-1]["result"]["error"] == "coordination_deadlock"
                action = {
                    "kind": "finish",
                    "action_id": "stop",
                    "outcome": "incomplete",
                    "summary": "Cannot resolve dependency",
                }
            return {"version": "swarm-v1", "actions": [action]}, {
                "provider": "fake",
                "status": "success",
                "code": "ok",
            }

    engine, state = setup(tmp_path, provider=CycleTeam())
    result = engine.execute(state)
    assert result.swarm["termination_reason"] == "coordinator_incomplete"
    assert result.swarm["requests"] == 2
    assert result.swarm["actions"]["cycle"]["status"] == "failed"


def test_own_finish_cannot_be_dependency_of_new_work(tmp_path):
    engine, state = setup(tmp_path)
    runtime = SwarmRuntime(engine, state, ScriptedTeam())
    try:
        task = runtime.session.tasks["coordinator"]
        runtime.accept(
            task,
            FinishAction(kind="finish", action_id="stop", summary="Stop", outcome="incomplete"),
        )
        with pytest.raises(ToolError, match="work_after_finish"):
            runtime.accept(
                task,
                WaitAction(
                    kind="wait",
                    action_id="after_stop",
                    depends_on=["stop"],
                    summary="Invalid later work",
                ),
            )
    finally:
        runtime.pool.shutdown()


def test_legacy_specialist_finish_cycle_returns_error_and_can_finish_incomplete(tmp_path):
    class RecoverTeam(ScriptedTeam):
        def decide(self, role, context, **limits):
            assert any(
                obs.get("result", {}).get("error") == "coordination_deadlock"
                for obs in context["observations"]
            )
            return {
                "version": "swarm-v1",
                "actions": [
                    {
                        "kind": "finish",
                        "action_id": context["task_id"] + "_stop",
                        "outcome": "incomplete",
                        "summary": "Dependency cycle prevents completion",
                    }
                ],
            }, {"provider": "fake", "status": "success", "code": "ok"}

    engine, state = setup(tmp_path, provider=RecoverTeam())
    runtime = SwarmRuntime(engine, state, engine.llm)
    try:
        runtime.session.tasks["audit"] = Task(
            id="audit", role="evaluation", parent="coordinator", goal="Evaluate"
        )
        # Legacy checkpoints could contain a later own action depending on finish.
        for owner, action, status in [
            (
                "coordinator",
                DelegateAction(
                    kind="delegate",
                    action_id="audit",
                    recipient="evaluation",
                    task="Evaluate",
                    summary="Evaluate",
                ),
                "reserved",
            ),
            (
                "audit",
                FinishAction(kind="finish", action_id="audit_finish", summary="Done"),
                "pending",
            ),
            (
                "audit",
                WaitAction(
                    kind="wait",
                    action_id="audit_cycle",
                    depends_on=["audit_finish"],
                    summary="Legacy cycle",
                ),
                "pending",
            ),
            (
                "coordinator",
                WaitAction(
                    kind="wait", action_id="await_audit", depends_on=["audit"], summary="Wait"
                ),
                "pending",
            ),
        ]:
            runtime.session.actions[action.action_id] = ActionRecord(
                task_id=owner, action=action, status=status
            )
        runtime.checkpoint()
    finally:
        runtime.pool.shutdown()
    result = engine.execute(engine.store.load(state.run_id))
    assert result.swarm["termination_reason"] == "coordinator_incomplete"
    assert result.swarm["requests"] == 2
    assert result.swarm["tasks"]["audit"]["status"] == "failed"


@pytest.mark.parametrize("fresh_ids,broken_finish", [(False, False), (True, False), (True, True)])
def test_repeated_evaluator_returns_evidence_and_run_finishes(tmp_path, fresh_ids, broken_finish):
    class LoopingEvaluator(ScriptedTeam):
        def decide(self, role, context, **limits):
            if role != "evaluation":
                return super().decide(role, context, **limits)
            with self.lock:
                self.calls.append((role, context))
            ref = "evaluate_" + str(len(self.calls)) if fresh_ids else "evaluate_again"
            actions = [
                {
                    "kind": "tool",
                    "action_id": ref,
                    "tool": "evaluate",
                    "summary": "Rank the unchanged measurements again.",
                }
            ]
            if broken_finish and sum(role == "evaluation" for role, _ in self.calls) >= 3:
                actions.append(
                    {
                        "kind": "finish",
                        "action_id": "audit_done",
                        "depends_on": [ref],
                        "summary": "Finish after rejected evaluation.",
                    }
                )
            return {"version": "swarm-v1", "actions": actions}, {
                "provider": "fake",
                "status": "success",
                "code": "ok",
            }

    provider = LoopingEvaluator(refine=False)
    engine, state = setup(tmp_path, provider=provider)
    result = engine.execute(state)
    assert result.status == "completed", result.swarm
    assert result.swarm["termination_reason"] == "coordinator_finished"
    assert result.swarm["tasks"]["audit"]["status"] == "failed"
    audit = result.swarm["actions"]["audit"]
    assert audit["result"]["error"] == "repeated_action_limit"
    evidence = [result.swarm["actions"][ref] for ref in audit["result"]["evidence"]]
    assert evidence and all(record["status"] == "completed" for record in evidence)
    assert evidence[0]["result"]["best_ref"] == result.best_id
    assert result.swarm["invalid_actions"] == 0
    assert result.metadata["swarm_submission_validated"] == result.metadata["submission_sha256"]
    assert len(result.experiments) == 1
    assert not engine.client.submit.called
    requests = len(provider.calls)
    assert engine.execute(engine.store.load(result.run_id)).status == "completed"
    assert len(provider.calls) == requests


@pytest.mark.parametrize("finish_first", [False, True])
def test_specialist_repeat_does_not_discard_queued_finish(tmp_path, finish_first):
    engine, state = setup(tmp_path)
    runtime = SwarmRuntime(engine, state, ScriptedTeam())
    try:
        coordinator = runtime.session.tasks["coordinator"]
        runtime.accept(
            coordinator,
            DelegateAction(
                kind="delegate",
                action_id="audit",
                recipient="evaluation",
                task="Audit existing results.",
                summary="Audit",
            ),
        )
        runtime.drain_ready()
        task = runtime.session.tasks["audit"]
        action = ToolAction(kind="tool", action_id="old_evaluation", tool="evaluate", summary="Old")
        runtime.session.actions[action.action_id] = ActionRecord(
            task_id=task.id, action=action, status="completed", result={"best_ref": "recorded"}
        )
        for _ in range(state.config.max_action_repeats - 1):
            runtime.accept(task, action)
        finish = FinishAction(kind="finish", action_id="audit_done", summary="Done")
        actions = [finish, action] if finish_first else [action, finish]
        runtime.commit_response(
            (task, 0),
            (AgentResponse(version="swarm-v1", actions=actions), {"status": "success"}),
        )
        assert task.status == "ready"
        runtime.drain_ready()
        assert task.status == "completed"
        assert runtime.session.actions["audit"].status == "completed"
        assert runtime.session.actions[action.action_id].result == {"best_ref": "recorded"}
        assert runtime.session.termination_reason is None
    finally:
        runtime.pool.shutdown()
