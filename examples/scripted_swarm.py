"""Offline protocol demonstration. Scripted LLM responses, real numerical workers.

uv run python -m examples.scripted_swarm --output artifacts/swarm-example
"""

import argparse
import json
import threading
from pathlib import Path

from examples.fixture import make_fixture
from kaggle_slayer.config import Settings
from kaggle_slayer.orchestration.engine import Engine
from kaggle_slayer.schemas import Competition, RunConfig
from kaggle_slayer.schemas.actions import AgentResponse
from kaggle_slayer.storage.store import Store, write_json


class ScriptedTeam:
    """Fake provider for tests/examples, branching on executed observations."""

    def __init__(self, refine=True, parallel=False, target_accuracy=1.01):
        self.settings = Settings(_env_file=None, llm_provider="deterministic", llm_max_tokens=1600)
        self.refine, self.parallel = refine, parallel
        self.target_accuracy = target_accuracy
        self.calls = []
        self.lock = threading.Lock()

    def decide(self, role, context, **limits):
        with self.lock:
            self.calls.append((role, context))
            index = len(self.calls)
        own = [o for o in context["observations"] if o["task_id"] == context["task_id"]]
        results = {o["action_id"]: o["result"] for o in own if o.get("code") == "tool_result"}
        task = context["task_id"]

        def tool(name, arguments=None, suffix="", **extra):
            return {
                "kind": "tool",
                "action_id": f"{task}-{name}{suffix}",
                "tool": name,
                "arguments": arguments or {},
                "summary": "Request bounded numerical evidence.",
                **extra,
            }

        def finish():
            return {
                "kind": "finish",
                "action_id": f"{task}-finish",
                "summary": "Requested work recorded.",
                "evidence": list(results)[-8:],
            }

        if role == "coordinator":
            tasks = {t["id"]: t for t in context["tasks"]}
            order = [
                ("data", "data_validation"),
                ("model", "pipeline_modeling"),
                ("audit", "evaluation"),
            ]
            measured = [v for v in context["evidence"].values() if "ranking" in v]
            # Observe measured results before deciding to request another experiment.
            if measured and self.refine and measured[-1]["ranking"]:
                score = measured[-1]["ranking"][0]["score"]
                if score is not None and (
                    score < self.target_accuracy
                    if context["task_requirements"]["task"] == "classification"
                    else score >= 0
                ):
                    order += [("refine", "pipeline_modeling"), ("reaudit", "evaluation")]
            order += [("results", "submission")]
            pending = next(
                (
                    key
                    for key, t in tasks.items()
                    if key != "coordinator" and t["status"] == "ready"
                ),
                None,
            )
            missing = next(((key, role) for key, role in order if key not in tasks), None)
            if pending:
                action = {
                    "kind": "wait",
                    "action_id": f"wait-{index}",
                    "depends_on": [pending],
                    "summary": "Wait for delegated evidence.",
                }
            elif missing:
                key, recipient = missing
                evidence = [
                    ref
                    for ref, result in context["evidence"].items()
                    if "ranking" in result or "lineage" in result
                ]
                action = {
                    "kind": "delegate",
                    "action_id": key,
                    "recipient": recipient,
                    "task": "Refine measured winner."
                    if key == "refine"
                    else "Execute scoped work.",
                    "evidence": evidence[-8:],
                    "summary": "Delegate based on current observations.",
                }
            else:
                action = {
                    "kind": "finish",
                    "action_id": "coordinator-finish",
                    "summary": "Measured artifacts and validated predictions recorded.",
                    "evidence": [
                        ref for ref, value in context["evidence"].items() if value.get("valid")
                    ],
                }
        elif role == "data_validation":
            if not results:
                action = tool("inspect_dataset")
            elif not any("lineage" in v for v in results.values()):
                action = tool(
                    "configure_validation",
                    {
                        "method": "stratified"
                        if context["task_requirements"]["task"] == "classification"
                        else "kfold"
                    },
                )
            else:
                action = finish()
        elif role == "pipeline_modeling":
            if task == "refine":
                if not results:
                    evaluation = next(v for v in context["evidence"].values() if "best_ref" in v)
                    key = (
                        "C" if context["task_requirements"]["task"] == "classification" else "alpha"
                    )
                    action = tool(
                        "tune_experiment",
                        {"parent_ref": evaluation["best_ref"], "params": {key: 0.25}},
                    )
                else:
                    action = finish()
            elif not results:
                key = "C" if context["task_requirements"]["task"] == "classification" else "alpha"
                action = tool(
                    "configure_pipeline",
                    {
                        "model": "linear",
                        "params": {key: 2.0},
                        "pipeline": {
                            "numeric_imputation": "mean",
                            "scaling": "standard",
                            "features": ["missing_count", "numeric_signed_log"],
                        },
                    },
                )
            elif not any("experiment_ref" in v for v in results.values()):
                ref = next(v["pipeline_ref"] for v in results.values() if "pipeline_ref" in v)
                action = tool("run_experiment", {"pipeline_ref": ref})
                if self.parallel:
                    action = [action, tool("run_experiment", {"pipeline_ref": ref}, "-second")]
            else:
                action = finish()
        elif role == "evaluation":
            action = tool("evaluate") if not results else finish()
        elif role == "submission":
            if not results:
                action = tool("generate_predictions")
            elif not any(v.get("valid") for v in results.values()):
                action = tool("validate_predictions")
            else:
                action = finish()
        else:
            raise AssertionError("unexpected role")
        response = AgentResponse.model_validate(
            {"version": "swarm-v1", "actions": action if isinstance(action, list) else [action]}
        )
        return response, {
            "provider": "scripted-test",
            "model": "fake",
            "status": "success",
            "code": "ok",
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "request_attempts": 1,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/swarm-example"))
    parser.add_argument(
        "--task", choices=["classification", "regression"], default="classification"
    )
    args = parser.parse_args()
    directory = make_fixture(args.output / "data", args.task, rows=90)
    classification = args.task == "classification"
    config = RunConfig(
        execution_mode="agentic",
        data_dir=str(directory),
        artifact_root=str(args.output),
        models=["linear"],
        max_experiments=3,
        folds=3,
        n_jobs=1,
        max_seconds=180,
    )
    competition = Competition(
        competition="offline-swarm",
        task=args.task,
        target="Survived" if classification else "SalePrice",
        id_column="PassengerId" if classification else "Id",
        metric="accuracy" if classification else "rmse",
        sample_file="sample_submission.csv",
    )
    engine = Engine(Store(args.output), llm=ScriptedTeam(), progress=print)
    result = engine.execute(engine.create(competition, config))
    trace = {
        "provider_execution": "scripted fake; no live LLM or Kaggle",
        "run_id": result.run_id,
        "status": result.status,
        "trace": [
            {
                "action_id": ref,
                "task": record["task_id"],
                "action": record["action"],
                "status": record["status"],
                "result": record["result"],
            }
            for ref, record in result.swarm["actions"].items()
        ],
        "artifacts": result.swarm["artifacts"],
    }
    write_json(args.output / "example_trace.json", trace)
    print(
        json.dumps(
            {
                "status": result.status,
                "termination": result.swarm["termination_reason"],
                "trace": str((args.output / "example_trace.json").resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
