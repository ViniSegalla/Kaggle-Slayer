"""Task-local IDs and repair accounting preserve accepted work across recovery."""

from unittest.mock import Mock

import pytest
from test_swarm import FixedProvider, setup

from examples.scripted_swarm import ScriptedTeam
from kaggle_slayer.orchestration.swarm import SwarmRuntime
from kaggle_slayer.schemas.actions import AgentResponse, Task, ToolAction


class LocalIdsTeam(ScriptedTeam):
    def decide(self, role, context, **limits):
        response, diagnostic = super().decide(role, context, **limits)
        for action in response.actions:
            if action.kind == "tool":
                action.action_id = action.tool
            elif action.kind == "finish":
                action.action_id = "finish"
        return response, diagnostic


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_local_ids_execute_real_tools_and_resume_without_replay(tmp_path, task):
    provider = LocalIdsTeam()
    engine, state = setup(tmp_path, task=task, provider=provider)
    result = engine.execute(state)
    assert result.status == "completed", result.swarm
    assert len(result.experiments) == 2
    assert all(e.status == "completed" and e.score is not None for e in result.experiments)
    assert result.metadata["swarm_submission_validated"]
    assert result.swarm["invalid_actions"] == result.swarm["schema_repairs"] == 0
    aliases = result.swarm["tasks"]["reaudit"]["action_aliases"]
    assert aliases["evaluate"] != "evaluate"
    assert result.swarm["actions"][aliases["evaluate"]]["status"] == "completed"
    assert result.swarm["tasks"]["results"]["action_aliases"]["finish"] != "finish"
    count = len(provider.calls)
    resumed = engine.execute(engine.store.load(result.run_id))
    assert len(provider.calls) == count and len(resumed.experiments) == 2
    assert resumed.swarm["tasks"]["reaudit"]["action_aliases"] == aliases
    assert not engine.client.submit.called


def test_batch_references_are_scoped_and_persisted(tmp_path):
    engine, state = setup(tmp_path)
    runtime = SwarmRuntime(engine, state, ScriptedTeam())
    try:
        tasks = [
            Task(id=name, role="pipeline_modeling", goal="Configure") for name in ("one", "two")
        ]
        runtime.session.tasks.update({task.id: task for task in tasks})
        pipeline = ToolAction(
            kind="tool",
            action_id="cfg",
            tool="configure_pipeline",
            arguments={"model": "linear"},
            summary="Configure",
        )
        runtime.accept(tasks[0], pipeline)
        batch = [
            pipeline,
            ToolAction(
                kind="tool",
                action_id="train",
                tool="run_experiment",
                arguments={"pipeline_ref": "cfg"},
                depends_on=["cfg"],
                summary="Train",
            ),
        ]
        scoped = runtime.scope_actions(tasks[1], batch)
        assert scoped[0].action_id != "cfg"
        assert scoped[1].depends_on == [scoped[0].action_id]
        assert scoped[1].arguments["pipeline_ref"] == scoped[0].action_id
        for action in scoped:
            runtime.accept(tasks[1], action)
        runtime.session.actions[scoped[0].action_id].status = "completed"
        references = [
            ToolAction(
                kind="tool",
                action_id="tune",
                tool="tune_experiment",
                arguments={"parent_ref": "train", "params": {"C": 1}},
                evidence=["cfg"],
                summary="Tune",
            ),
            ToolAction(
                kind="tool",
                action_id="blend",
                tool="ensemble",
                arguments={"members": ["train", "foreign"]},
                summary="Blend",
            ),
        ]
        more = runtime.scope_actions(tasks[1], references)
        assert more[0].evidence == [scoped[0].action_id]
        assert more[0].arguments["parent_ref"] == scoped[1].action_id
        assert more[1].arguments["members"] == [scoped[1].action_id, "foreign"]
        runtime.checkpoint()
        saved = engine.store.load(state.run_id).swarm
        assert saved["tasks"]["two"]["action_aliases"] == tasks[1].action_aliases
        assert runtime.context(tasks[1])["action_aliases"] == tasks[1].action_aliases
        assert saved["actions"]["cfg"]["task_id"] == "one"
    finally:
        runtime.pool.shutdown()


def test_changed_contract_requests_fresh_id_without_partial_acceptance(tmp_path):
    engine, state = setup(tmp_path)
    runtime = SwarmRuntime(engine, state, ScriptedTeam())
    task = runtime.session.tasks["coordinator"]
    diagnostic = {"status": "success", "provider": "fake", "code": "ok"}

    def commit(actions):
        runtime.commit_response(
            (task, 0), ({"version": "swarm-v1", "actions": actions}, diagnostic)
        )

    original = {
        "kind": "delegate",
        "action_id": "model",
        "recipient": "pipeline_modeling",
        "task": "Train linear",
        "summary": "Delegate",
    }
    try:
        commit([original])
        commit([original | {"summary": "Same operation, revised explanation"}])
        assert len(runtime.session.actions) == 1
        commit([original | {"action_id": "new"}, original | {"task": "Train forest"}])
        assert task.repair_pending and len(runtime.session.actions) == 1
        assert runtime.session.actions["model"].action.task == "Train linear"
        assert runtime.session.schema_repairs == 1 and runtime.session.invalid_actions == 0
        assert runtime.context(task)["owned_actions"][0]["task"] == "Train linear"
        commit([original | {"action_id": "forest", "task": "Train forest"}])
        assert not task.repair_pending and set(runtime.session.actions) == {"model", "forest"}
    finally:
        runtime.pool.shutdown()


def test_repairs_in_different_tasks_do_not_exhaust_invalid_action_budget(tmp_path):
    engine, state = setup(tmp_path, max_invalid_actions=1)
    runtime = SwarmRuntime(engine, state, ScriptedTeam())
    try:
        for index in range(5):
            task = Task(id=f"data{index}", role="data_validation", goal="Inspect")
            runtime.session.tasks[task.id] = task
            runtime.request_repair(task, ["actions.0.tool.summary: string_too_long"])
            response = AgentResponse(
                version="swarm-v1",
                actions=[
                    ToolAction(
                        kind="tool", action_id="inspect", tool="inspect_dataset", summary="Inspect"
                    )
                ],
            )
            runtime.commit_response((task, 0), (response, {"status": "success", "code": "ok"}))
            assert not task.repair_pending and task.repair_attempts == 0
        assert runtime.session.schema_repairs == 5 and runtime.session.invalid_actions == 0
        assert len(runtime.session.actions) == 5
    finally:
        runtime.pool.shutdown()


def test_old_checkpoint_reclassifies_only_recorded_schema_failures_once(tmp_path):
    engine, state = setup(tmp_path)
    state.swarm.pop("counter_version")
    state.swarm.pop("schema_repairs")
    state.swarm["invalid_actions"] = 4
    state.swarm["observations"] = [
        {"task_id": "coordinator", "code": code}
        for code in [
            "response_repair_required",
            "role_permission",
            "response_repair_required",
            "action_id_collision",
        ]
    ]
    state.swarm["usage"] = [{"code": "schema_mismatch"}, {"code": "schema_mismatch"}]
    runtime = SwarmRuntime(engine, state, ScriptedTeam())
    try:
        assert runtime.session.schema_repairs == runtime.session.invalid_actions == 2
        assert runtime.session.counter_version == 2
        runtime.checkpoint()
    finally:
        runtime.pool.shutdown()
    resumed = SwarmRuntime(engine, engine.store.load(state.run_id), ScriptedTeam())
    try:
        assert resumed.session.schema_repairs == resumed.session.invalid_actions == 2
    finally:
        resumed.pool.shutdown()


def test_duplicate_batch_ids_are_repaired_without_accepting_any_action(tmp_path):
    provider = FixedProvider()
    duplicate = {
        "kind": "delegate",
        "action_id": "data",
        "recipient": "data_validation",
        "task": "Inspect",
        "summary": "Inspect",
    }
    provider.decide = Mock(
        side_effect=[
            ({"version": "swarm-v1", "actions": [duplicate, duplicate]}, {"status": "success"}),
            (
                {
                    "version": "swarm-v1",
                    "actions": [
                        {
                            "kind": "finish",
                            "action_id": "stop",
                            "summary": "Stop",
                            "outcome": "incomplete",
                        }
                    ],
                },
                {"status": "success"},
            ),
        ]
    )
    engine, state = setup(tmp_path, provider=provider)
    result = engine.execute(state)
    assert result.swarm["termination_reason"] == "coordinator_incomplete"
    assert set(result.swarm["actions"]) == {"stop"}
    assert result.swarm["schema_repairs"] == 1 and result.swarm["invalid_actions"] == 0
