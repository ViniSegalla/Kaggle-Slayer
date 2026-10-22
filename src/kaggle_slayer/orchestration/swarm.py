"""LLM action/observation scheduler. One thread owns state, SQLite and reservations."""

import hashlib
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from queue import Empty, SimpleQueue

from pydantic import ValidationError

from kaggle_slayer.config import Settings
from kaggle_slayer.llm.diagnostics import action_validation_issues, redact
from kaggle_slayer.llm.provider import ActionLLM
from kaggle_slayer.schemas.actions import (
    VERSION,
    ActionRecord,
    AgentResponse,
    SwarmState,
    Task,
    ToolAction,
)
from kaggle_slayer.schemas.models import Decision, Experiment, RunState
from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.data_tools import DatasetError, inspect_dataset
from kaggle_slayer.tools.runtime_worker import (
    IMMUTABLE_DATASET_ERRORS,
    OPERATION_ERRORS,
    TRAINING_RESOURCE_ERRORS,
    RuntimeOperationError,
)
from kaggle_slayer.tools.swarm_tools import (
    CAPABILITIES,
    Dispatcher,
    ToolError,
    dataset_observation,
    registry,
)
from kaggle_slayer.tools.training_tools import run_isolated


class StopRun(Exception):
    pass


class CheckpointError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class SwarmRuntime:
    def __init__(self, engine, state, provider=None):
        self.engine, self.state, self.store = engine, state, engine.store
        self.provider = provider or ActionLLM()
        self.settings = getattr(self.provider, "settings", Settings())
        self.session = SwarmState.model_validate(state.swarm)
        if self.session.counter_version == 1:
            # Old checkpoints charged formatting failures to the invalid-action cap.
            # Reclassify only recorded failures; retain permissions/contract violations.
            formats = max(
                sum(o.get("code") == "response_repair_required" for o in self.session.observations),
                sum(
                    u.get("code") in {"schema_mismatch", "invalid_advice_json"}
                    for u in self.session.usage
                ),
            )
            self.session.schema_repairs += formats
            self.session.invalid_actions = max(0, self.session.invalid_actions - formats)
            self.session.counter_version = 2
        self.dispatcher = Dispatcher(self)
        self.directory = self.store.run_dir(state.run_id)
        self.data = None
        self.stopping = threading.Event()
        self.worker_events = SimpleQueue()
        self.started = time.monotonic()
        self.previous_elapsed = state.elapsed_seconds
        self.pool = ThreadPoolExecutor(max_workers=state.config.agent_concurrency)

    def remaining(self):
        return (
            self.state.config.max_seconds
            - self.previous_elapsed
            - (time.monotonic() - self.started)
        )

    def cancelled(self):
        return self.stopping.is_set() or (self.directory / "cancel.request").exists()

    def guard(self):
        if self.cancelled():
            raise StopRun("cancelled")
        if self.remaining() <= 0:
            raise StopRun("wall_time_exhausted")

    def checkpoint(self, journal=False):
        s = self.state
        s.elapsed_seconds = self.previous_elapsed + time.monotonic() - self.started
        s.swarm = self.session.model_dump(mode="json")
        try:
            if journal:
                sequence = s.metadata.get("swarm_commit_sequence", 0) + 1
                s.metadata["swarm_commit_sequence"] = sequence
                # A local durable receipt closes the worker-result/SQLite commit gap.
                write_json(self.directory / "swarm_receipt.json", s.model_dump(mode="json"))
            self.store.save(s)
        except OSError as exc:
            raise CheckpointError("checkpoint_persistence_failed") from exc

    def observe(self, task, code, **result):
        self.session.observations.append({"task_id": task, "code": code, **result})
        self.session.observations = self.session.observations[-64:]

    def log(self, task, action, result, level="info"):
        self.store.record(
            self.state.run_id,
            Decision(
                agent=self.session.tasks[task].role,
                input_context={
                    "session_id": self.session.session_id,
                    "task_id": task,
                    "mode": "agentic",
                    "protocol": VERSION,
                },
                selected_action=action.kind,
                rationale=action.summary,
                tool_invoked=action.tool if action.kind == "tool" else action.kind,
                tool_result={"action_id": action.action_id, **result},
                next_action="llm_observe",
                level=level,
            ),
        )
        self.engine.live.emit(
            self.state.run_id,
            self.session.tasks[task].role,
            f"{task} | {action.action_id} | {getattr(action, 'tool', action.kind)} | {result.get('status', '')}"
            + (
                f" | measured score={result['score']:.6g}"
                if result.get("score") is not None
                else ""
            )
            + (
                " | " + str(RuntimeOperationError(result["error"], result.get("details")))
                if result.get("error") in OPERATION_ERRORS
                else ""
            ),
        )

    def register_artifact(self, ref, paths):
        files = []
        for path in paths:
            p = Path(path).resolve()
            if not p.is_relative_to(self.directory.resolve()) or not p.is_file():
                raise ToolError("artifact_boundary")
            files.append({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        self.session.artifacts[ref] = {"files": files}

    def verify_artifact(self, ref):
        artifact = self.session.artifacts.get(ref)
        if not artifact:
            raise ToolError("artifact_not_recorded")
        for file in artifact["files"]:
            path = Path(file["path"]).resolve()
            if (
                not path.is_relative_to(self.directory.resolve())
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != file["sha256"]
            ):
                raise ToolError("artifact_integrity")

    def measurement_revision(self):
        return digest(
            {
                "dataset": self.state.report.get("fingerprints") if self.state.report else None,
                "validation": self.session.validation,
                "experiments": [
                    {"id": e.id, "status": e.status, "lineage": e.lineage}
                    for e in self.state.experiments
                ],
            }
        )

    def progress(self):
        """Durable bounded receipts, independent of the rolling dialogue window."""
        s = self.state
        receipts = {}
        for ref, record in self.session.actions.items():
            if record.status != "completed" or record.action.kind != "tool":
                continue
            result, name = record.result, record.action.tool
            if name == "evaluate" and s.best_id and result.get("best_ref") == s.best_id:
                receipts["evaluation"] = {"action_id": ref, "result": result}
            elif name == "generate_predictions" and (
                self.session.finalized
                and result.get("best_ref") == s.best_id
                and result.get("sha256") == s.metadata.get("submission_sha256")
            ):
                receipts["predictions"] = {"action_id": ref, "result": result}
            elif name == "validate_predictions" and (
                result.get("valid")
                and result.get("sha256")
                == s.metadata.get("swarm_submission_validated")
                == s.metadata.get("submission_sha256")
            ):
                receipts["validation"] = {"action_id": ref, "result": result}
        blockers = [
            {
                "action_id": ref,
                "task_id": rec.task_id,
                "kind": rec.action.kind,
                "status": rec.status,
                "depends_on": rec.action.depends_on,
            }
            for ref, rec in self.session.actions.items()
            if rec.status in {"pending", "reserved"} and rec.action.kind != "finish"
        ]
        return {
            "revision": self.measurement_revision(),
            "best_ref": s.best_id,
            "receipts": receipts,
            "experiments": [
                {
                    "ref": e.id,
                    "model": e.model,
                    "status": e.status,
                    "score": e.score,
                    "lineage": e.lineage,
                    "validation_role": e.validation_role,
                }
                for e in s.experiments[-16:]
            ],
            "finish_blockers": blockers,
            "ready_to_finish": bool(
                receipts.get("validation") and receipts.get("evaluation") and not blockers
            ),
        }

    def context(self, task):
        from kaggle_slayer.agents.swarm import role_response_schema

        s, session = self.state, self.session
        # Only locally constructed aggregates/opaque IDs cross the provider boundary.
        observations = (
            session.observations[-16:]
            if task.role == "coordinator"
            else [
                obs
                for obs in session.observations
                if obs["task_id"] == task.id or obs.get("action_id") in task.evidence
            ][-12:]
        )
        evidence_ids = set(task.evidence)
        if task.role == "coordinator":
            evidence_ids.update(list(session.actions)[-16:])
        evidence = {
            ref: session.actions[ref].result for ref in evidence_ids if ref in session.actions
        }
        return {
            "protocol": VERSION,
            "response_schema": role_response_schema(task.role),
            "session_id": session.session_id,
            "task_id": task.id,
            "role": task.role,
            "action_aliases": task.action_aliases,
            "owned_actions": [
                {
                    "id": ref,
                    "kind": record.action.kind,
                    "status": record.status,
                    "tool": getattr(record.action, "tool", None),
                    "task": getattr(record.action, "task", None),
                }
                for ref, record in session.actions.items()
                if record.task_id == task.id
            ][-32:],
            "response_repair": {
                "attempt": task.repair_attempts,
                "issues": task.repair_issues,
                "instruction": "Your previous response was rejected; no actions from it executed. Regenerate a complete AgentResponse JSON envelope matching response_schema. Return only supported typed actions, with short summaries and no extra fields or private reasoning.",
            }
            if task.repair_pending
            else None,
            "goal": task.goal,
            "capabilities": registry(task.role, s),
            "team_capabilities": {
                role: registry(role, s)
                for role in ("data_validation", "pipeline_modeling", "evaluation", "submission")
            }
            if task.role == "coordinator"
            else {},
            "task_requirements": {
                "task": s.competition.task,
                "metric": s.competition.metric,
                "prediction_kind": s.competition.prediction_kind,
                "target_transform": s.competition.target_transform,
            },
            "constraints": {
                key: s.config.model_dump()[key]
                for key in (
                    "models",
                    "folds",
                    "max_experiments",
                    "max_seconds",
                    "experiment_timeout",
                    "n_jobs",
                    "max_rows",
                    "max_features",
                    "max_dense_cells",
                    "ensemble",
                    "confirmation_top_k",
                    "agent_concurrency",
                    "max_delegation_depth",
                )
            },
            "budget": {
                "experiments_remaining": s.config.max_experiments - session.experiment_attempts,
                "seconds_remaining": round(max(0, self.remaining()), 2),
                "requests_remaining": s.config.max_llm_requests - session.requests,
                "tokens_remaining": s.config.max_llm_tokens - session.tokens,
                "turns_remaining": s.config.max_agent_turns - session.turns,
                "tuning_attempts_remaining": s.config.max_iterations * s.config.optuna_trials
                - session.tuning_attempts,
            },
            "dataset": dataset_observation(s) if s.report else {"status": "not_inspected"},
            "validation": session.validation,
            "confirmation_started": session.confirmation_started,
            "finalized": session.finalized,
            "progress": self.progress(),
            "tasks": [
                {
                    "id": t.id,
                    "role": t.role,
                    "status": t.status,
                    "parent": t.parent,
                    "goal": t.goal,
                    "depends_on": t.depends_on,
                    "evidence": t.evidence,
                }
                for t in session.tasks.values()
            ]
            if task.role == "coordinator"
            else [],
            "pending_actions": [
                {
                    "id": ref,
                    "task_id": rec.task_id,
                    "kind": rec.action.kind,
                    "depends_on": rec.action.depends_on,
                    "reused_from": rec.reused_from,
                }
                for ref, rec in session.actions.items()
                if rec.status in {"pending", "reserved"}
            ],
            "observations": observations,
            "evidence": evidence,
        }

    def call_agents(self, tasks):
        futures = {}
        for task in tasks:
            self.guard()
            config, session = self.state.config, self.session
            if session.requests >= config.max_llm_requests:
                raise StopRun("llm_request_budget_exhausted")
            if session.turns >= config.max_agent_turns:
                raise StopRun("agent_turn_budget_exhausted")
            context = self.context(task)
            if len(json.dumps(context).encode()) > config.max_llm_context_bytes:
                raise StopRun("llm_context_limit")
            max_tokens = self.settings.llm_max_tokens
            # UTF-8 byte length is a conservative input-token reservation. Unknown
            # provider usage retains the reservation; failed calls also consume it.
            reservation = (
                len(json.dumps(context).encode())
                + len(self.prompt(task.role).encode())
                + max_tokens
                + len(json.dumps(AgentResponse.model_json_schema()).encode())
                + 256
            )
            if session.tokens + reservation > config.max_llm_tokens:
                raise StopRun("llm_token_budget_exhausted")
            session.requests += 1
            session.turns += 1
            session.tokens += reservation
            task.turns += 1
            self.checkpoint()
            self.engine.live.emit(
                self.state.run_id,
                task.role,
                f"LLM turn | task={task.id} | request={session.requests}/{config.max_llm_requests}"
                f" | experiments remaining={config.max_experiments - session.experiment_attempts}"
                f" | wall remaining={self.remaining():.0f}s",
            )
            call = getattr(self.provider, "decide_isolated", self.provider.decide)
            options = (
                {"cancelled": self.cancelled} if hasattr(self.provider, "decide_isolated") else {}
            )
            future = self.pool.submit(
                call,
                task.role,
                context,
                timeout=min(self.settings.llm_timeout, self.remaining()),
                max_tokens=max_tokens,
                **options,
            )
            futures[future] = (task, reservation)
        self.collect(futures, self.commit_response)

    @staticmethod
    def prompt(role):
        from kaggle_slayer.agents.swarm import role_prompt

        return role_prompt(role)

    def collect(self, futures, commit):
        try:
            while futures:
                self.guard()
                self.drain_events()
                done, _ = wait(futures, timeout=min(0.2, self.remaining()))
                for future in done:
                    info = futures.pop(future)
                    try:
                        value = future.result()
                    except Exception:
                        value = None
                    commit(info, value)
            self.drain_events()
        finally:
            for future in futures:
                future.cancel()

    def drain_events(self):
        while True:
            try:
                task_id, message = self.worker_events.get_nowait()
            except Empty:
                return
            self.engine.live.emit(self.state.run_id, self.session.tasks[task_id].role, message)

    def commit_response(self, info, value):
        task, reservation = info
        if (
            value is None
            or not isinstance(value, tuple)
            or len(value) != 2
            or not isinstance(value[1], dict)
        ):
            self.observe(task.id, "provider_failure")
            self.checkpoint()
            raise StopRun("provider_failure")
        response, diagnostic = value
        allowed = {
            "provider",
            "model",
            "status",
            "code",
            "message",
            "http_status",
            "request_attempts",
            "prompt_tokens",
            "completion_tokens",
            "duration_seconds",
            "validation_issues",
            "action",
        }
        safe = {k: diagnostic[k] for k in allowed if k in diagnostic}
        # Protocol implementers must provide controlled diagnostics. No raw exception/body.
        self.session.usage.append(safe)
        counts = [diagnostic.get("prompt_tokens"), diagnostic.get("completion_tokens")]
        if all(type(v) is int and v >= 0 for v in counts):
            actual = sum(counts)
            self.session.tokens += actual - reservation
        if self.session.tokens > self.state.config.max_llm_tokens:
            raise StopRun("llm_token_budget_exhausted")
        if response is None:
            if diagnostic.get("code") in {"schema_mismatch", "invalid_advice_json"}:
                self.request_repair(
                    task, safe.get("validation_issues") or ["response: invalid_json"]
                )
                return
            self.observe(task.id, "provider_failure", diagnostic=safe)
            self.checkpoint()
            code = diagnostic.get("code", "invalid_response")
            known = {
                "rate_limit",
                "invalid_request",
                "authentication",
                "credits",
                "permission_or_budget",
                "timeout",
                "provider_timeout",
                "connection",
                "configuration",
                "missing_api_key",
                "unsupported_format",
                "invalid_model",
                "not_found",
                "no_available_provider",
                "provider_unavailable",
                "provider_error",
                "output_truncated",
                "refusal",
                "empty_response",
                "deterministic",
                "deadline_or_cancelled",
                "invalid_endpoint",
            }
            raise StopRun("llm_" + (code if code in known else "invalid_response"))
        try:
            response = AgentResponse.model_validate(response)
            # Redact configured secrets before persisting any summaries.
            response = AgentResponse.model_validate_json(
                redact(response.model_dump_json(), self.settings.llm_api_key)
            )
        except ValidationError as exc:
            self.request_repair(task, action_validation_issues(exc))
            return
        except ValueError:
            self.request_repair(task, ["response: invalid_json"])
            return
        try:
            actions = self.scope_actions(task, response.actions)
        except ToolError:
            self.request_repair(
                task, ["actions.action_id: reused_with_different_contract; use a fresh local ID"]
            )
            return
        task.repair_pending, task.repair_attempts, task.repair_issues = False, 0, []
        repeated = False
        for action in actions:
            try:
                repeated = self.accept(task, action, defer_repeat=True) or repeated
            except ToolError as exc:
                self.invalid(task.id, str(exc), action.action_id)
            except ValidationError:
                self.invalid(task.id, "invalid_action_contract", action.action_id)
        if repeated:
            self.fail_repeating_task(task)
        self.checkpoint()

    def request_repair(self, task, issues):
        task.repair_pending = True
        task.repair_attempts += 1
        task.repair_issues = issues[:6]
        self.session.schema_repairs += 1
        self.observe(
            task.id,
            "response_repair_required",
            issues=task.repair_issues,
            repair_attempt=task.repair_attempts,
        )
        self.engine.live.emit(
            self.state.run_id,
            task.role,
            f"{task.id} | "
            + (
                f"schema repair {task.repair_attempts}/2"
                if task.repair_attempts <= 2
                else "schema repair exhausted (2 corrections)"
            )
            + " | "
            + "; ".join(task.repair_issues),
        )
        self.checkpoint()
        if task.repair_attempts > 2:
            raise StopRun("llm_schema_repair_exhausted")

    @staticmethod
    def action_contract(action):
        # Wording/context updates do not make an already accepted operation new work.
        return action.model_dump(exclude={"summary", "evidence"})

    def scope_actions(self, task, actions):
        aliases = task.action_aliases.copy()
        if len({a.action_id for a in actions}) != len(actions):
            raise ToolError("duplicate_batch_id")
        for action in actions:
            local = action.action_id
            if local in aliases:
                continue
            existing = self.session.actions.get(local)
            if (existing and existing.task_id != task.id) or (
                local in self.session.tasks and not existing
            ):
                aliases[local] = "a-" + digest({"task": task.id, "id": local})[:32]
            else:
                aliases[local] = local
        scoped = []
        for action in actions:
            value = action.model_dump()
            value["action_id"] = aliases[action.action_id]
            for field in ("depends_on", "evidence"):
                value[field] = [aliases.get(ref, ref) for ref in value[field]]
            if action.kind == "tool":
                arguments = value["arguments"].copy()
                for key in ("pipeline_ref", "parent_ref"):
                    if key in arguments and isinstance(arguments[key], str):
                        arguments[key] = aliases.get(arguments[key], arguments[key])
                if isinstance(arguments.get("members"), list):
                    arguments["members"] = [
                        aliases.get(ref, ref) if isinstance(ref, str) else ref
                        for ref in arguments["members"]
                    ]
                value["arguments"] = arguments
            normalized = type(action).model_validate(value)
            old = self.session.actions.get(normalized.action_id)
            if old and (
                old.task_id != task.id
                or self.action_contract(old.action) != self.action_contract(normalized)
            ):
                raise ToolError("action_id_collision")
            scoped.append(normalized)
        task.action_aliases = aliases
        return scoped

    def invalid(self, task, code, action_id=None):
        self.session.invalid_actions += 1
        guidance = {}
        if code == "role_permission":
            role = self.session.tasks[task].role
            guidance = {
                "permitted_tools": [
                    name for name, cap in CAPABILITIES.items() if role in cap.roles
                ],
                "instruction": "Use only your permitted tools. The coordinator must delegate dataset inspection to recipient=data_validation with kind=delegate; it cannot call inspect_dataset directly.",
            }
        self.observe(task, code, action_id=action_id, **guidance)
        self.engine.live.emit(
            self.state.run_id,
            self.session.tasks[task].role,
            f"{task} | {action_id or 'action'} | rejected | {code}"
            + (" | " + guidance["instruction"] if guidance else ""),
        )
        if self.session.invalid_actions >= self.state.config.max_invalid_actions:
            raise StopRun("invalid_action_limit")

    def dependency_status(self, ref):
        if ref in self.session.actions:
            return self.session.actions[ref].status
        if ref in self.session.tasks:
            return self.session.tasks[ref].status
        return None

    def repeated_action(self, task, action, *, defer_failure=False):
        """Contain a specialist loop while preserving its measured evidence."""
        if task.role == "coordinator":
            raise StopRun("repeated_action_limit")
        result = {
            "error": "repeated_action_limit",
            "instruction": "Use the recorded evidence. Repeating unchanged work with a fresh ID does not add evidence; the coordinator can continue from completed results.",
        }
        if action.action_id not in self.session.actions:
            self.session.actions[action.action_id] = ActionRecord(task_id=task.id, action=action)
            self.complete(action.action_id, result, "failed")
        else:
            self.observe(task.id, "repeated_action_limit", action_id=action.action_id, **result)
            self.log(task.id, action, {"status": "rejected", **result}, "warning")
        if not defer_failure:
            self.fail_repeating_task(task)
        return True

    def fail_repeating_task(self, task):
        # Accept the whole response before checking for distinct work or a finish.
        if any(
            record.task_id == task.id and record.status in {"pending", "reserved"}
            for record in self.session.actions.values()
        ):
            return
        task.status, task.repair_pending = "failed", False
        evidence = [
            ref
            for ref, record in self.session.actions.items()
            if record.task_id == task.id and record.status == "completed"
        ][-16:]
        if task.parent and task.id in self.session.actions:
            self.complete(
                task.id,
                {
                    "task_id": task.id,
                    "summary": "Specialist stopped after repeated unchanged actions; completed evidence remains available.",
                    "evidence": evidence,
                    "error": "repeated_action_limit",
                    "instruction": "Continue from completed evidence or repair the failed task dependencies.",
                },
                "failed",
            )

    def accept(self, task, action, *, defer_repeat=False):
        session = self.session
        old = session.actions.get(action.action_id)
        if old:
            if old.task_id == task.id and self.action_contract(old.action) == self.action_contract(
                action
            ):
                self.observe(task.id, "deduplicated", action_id=action.action_id, result=old.result)
                if old.status in {"pending", "reserved"}:
                    # Referring to outstanding work is not another execution attempt.
                    return
                signature = "duplicate-" + action.action_id
                session.signatures[signature] = session.signatures.get(signature, 0) + 1
                if session.signatures[signature] >= self.state.config.max_action_repeats:
                    return self.repeated_action(task, action, defer_failure=defer_repeat)
                return
            raise ToolError("action_id_collision")
        if action.action_id in session.tasks:
            raise ToolError("task_id_collision")
        if any(self.dependency_status(ref) is None for ref in action.depends_on):
            raise ToolError("unknown_dependency")
        if any(
            ref in session.actions
            and session.actions[ref].task_id == task.id
            and session.actions[ref].action.kind == "finish"
            for ref in action.depends_on
        ):
            raise ToolError("work_after_finish")
        if any(
            ref not in session.actions or session.actions[ref].status not in {"completed", "failed"}
            for ref in action.evidence
        ):
            raise ToolError("unknown_evidence")
        if isinstance(action, ToolAction):
            self.dispatcher.validate(task.role, action)
        if action.kind == "delegate":
            if task.role not in {"coordinator", "pipeline_modeling", "evaluation"}:
                raise ToolError("delegation_permission")
            if (
                action.recipient == "coordinator"
                or task.depth >= self.state.config.max_delegation_depth
            ):
                raise ToolError("delegation_depth")
            if task.role != "coordinator" and action.recipient not in {
                "data_validation",
                "evaluation",
            }:
                raise ToolError("delegation_permission")
        signature = digest(
            {
                "task": task.id,
                "kind": action.kind,
                "tool": getattr(action, "tool", None),
                "arguments": getattr(action, "arguments", None),
                "recipient": action.recipient,
                "depends_on": action.depends_on,
            }
        )
        if action.kind == "tool":
            if action.tool in {"evaluate", "configure_validation"}:
                signature += str(len(self.state.experiments))
            count = session.signatures.get(signature, 0) + 1
            session.signatures[signature] = count
            if count > self.state.config.max_action_repeats:
                return self.repeated_action(task, action, defer_failure=defer_repeat)
        record = ActionRecord(task_id=task.id, action=action)
        if action.kind == "delegate":
            record.task_scope = digest(
                {
                    "parent": task.id,
                    "recipient": action.recipient,
                    "goal": " ".join(action.task.split()),
                    "dependencies": sorted(action.depends_on),
                    "evidence": sorted(action.evidence),
                    "revision": self.measurement_revision(),
                }
            )
            source = next(
                (
                    ref
                    for ref, existing in session.actions.items()
                    if existing.task_scope == record.task_scope
                    and existing.status != "failed"
                    and not existing.reused_from
                ),
                None,
            )
            if source:
                record.reused_from = source
        session.actions[action.action_id] = record
        self.log(task.id, action, {"status": "accepted"})

    def complete(self, ref, result, status="completed"):
        record = self.session.actions[ref]
        record.status, record.result = status, result
        self.observe(
            record.task_id,
            "tool_result" if record.action.kind == "tool" else record.action.kind,
            action_id=ref,
            status=status,
            result=result,
        )
        self.log(
            record.task_id,
            record.action,
            {"status": status, **result},
            "warning" if status == "failed" else "info",
        )
        self.checkpoint(journal=True)

    def drain_ready(self):
        # A later action can complete a dependency of an earlier pending action.
        # Activate that work before asking the coordinator for another decision.
        while not self.session.termination_reason and self.execute_ready():
            self.guard()

    def execute_ready(self):
        training = {}
        progressed = False
        for ref, record in list(self.session.actions.items()):
            if record.status != "pending":
                continue
            self.guard()
            action, task = record.action, self.session.tasks[record.task_id]
            if record.reused_from:
                source = self.session.actions[record.reused_from]
                if source.status in {"completed", "failed"}:
                    if (
                        source.status == "completed"
                        and action.kind == "tool"
                        and CAPABILITIES[action.tool].experiment
                    ):
                        try:
                            reused = self.dispatcher.reuse_training(record.tool_snapshot)
                            if not reused:
                                raise ToolError("training_reuse_unavailable")
                            source_ref, result = reused
                            self.complete(ref, {**result, "reused_from": source_ref})
                        except ToolError as exc:
                            self.complete(ref, {"error": str(exc)}, "failed")
                    else:
                        self.complete(
                            ref, {**source.result, "reused_from": record.reused_from}, source.status
                        )
                    progressed = True
                continue
            statuses = [self.dependency_status(dep) for dep in action.depends_on]
            if any(status == "failed" for status in statuses):
                self.complete(ref, {"error": "dependency_failed"}, "failed")
                progressed = True
                continue
            if any(status != "completed" for status in statuses):
                continue
            # Finish is a barrier over existing work, not a failed new experiment.
            if action.kind == "finish":
                if any(
                    key != ref
                    and other.task_id == task.id
                    and other.status in {"pending", "reserved"}
                    and other.action.kind != "finish"
                    for key, other in self.session.actions.items()
                ):
                    continue
                if task.role == "coordinator" and any(
                    t.status not in {"completed", "failed"}
                    for t in self.session.tasks.values()
                    if t.id != task.id
                ):
                    continue
            progressed = True
            try:
                if action.kind == "delegate":
                    self.session.tasks[ref] = Task(
                        id=ref,
                        role=action.recipient,
                        parent=task.id,
                        depth=task.depth + 1,
                        goal=action.task,
                        depends_on=action.depends_on,
                        evidence=action.evidence,
                    )
                    # Delegation action completes only when the child task completes.
                    record.status = "reserved"
                    self.checkpoint()
                elif action.kind == "wait":
                    self.complete(ref, {"dependencies": "completed"})
                elif action.kind == "finish":
                    if task.role == "coordinator":
                        valid = self.state.metadata.get("swarm_submission_validated")
                        if action.outcome == "complete" and (
                            not valid or valid != self.state.metadata.get("submission_sha256")
                        ):
                            raise ToolError("verified_predictions_required")
                        if action.outcome == "complete":
                            self.verify_artifact(self.state.best_id)
                            if (
                                hashlib.sha256(
                                    Path(self.state.submission_path).read_bytes()
                                ).hexdigest()
                                != valid
                            ):
                                raise ToolError("prediction_integrity")
                        self.session.termination_reason = (
                            "coordinator_finished"
                            if action.outcome == "complete"
                            else "coordinator_incomplete"
                        )
                    task.status = "completed" if action.outcome == "complete" else "failed"
                    self.complete(ref, {"outcome": action.outcome})
                    if task.parent and task.id in self.session.actions:
                        self.complete(
                            task.id,
                            {
                                "task_id": task.id,
                                "summary": action.summary,
                                "evidence": action.evidence,
                            },
                            "completed" if action.outcome == "complete" else "failed",
                        )
                else:
                    args = self.dispatcher.validate(task.role, action)
                    if CAPABILITIES[action.tool].experiment:
                        candidate = self.dispatcher.candidate(action, args, for_reuse=True)
                        worker_config = self.state.config.model_copy(
                            update={
                                "n_jobs": max(
                                    1,
                                    self.state.config.n_jobs
                                    // min(
                                        self.state.config.agent_concurrency,
                                        self.state.config.n_jobs,
                                    ),
                                ),
                                "seed": candidate.validation_seed
                                if candidate.validation_seed is not None
                                else self.state.config.seed,
                            }
                        )
                        record.tool_snapshot = self.dispatcher.training_key(
                            candidate, worker_config
                        )
                        reused = self.dispatcher.reuse_training(record.tool_snapshot)
                        if reused:
                            source_ref, result = reused
                            record.reused_from = source_ref
                            self.complete(ref, {**result, "reused_from": source_ref})
                            continue
                        source_ref = next(
                            (
                                key
                                for key, other in self.session.actions.items()
                                if key != ref
                                and other.status == "reserved"
                                and other.reservation.get("candidate")
                                and other.tool_snapshot == record.tool_snapshot
                            ),
                            None,
                        )
                        if source_ref:
                            record.reused_from = source_ref
                            self.checkpoint()
                            continue
                        if action.tool == "confirm" and training:
                            continue
                        if training and any(
                            self.session.actions[pending].action.tool == "confirm"
                            for pending in training.values()
                        ):
                            continue
                        if len(training) >= min(
                            self.state.config.agent_concurrency, self.state.config.n_jobs
                        ):
                            continue
                        if self.session.experiment_attempts >= self.state.config.max_experiments:
                            raise ToolError("experiment_budget_exhausted")
                        candidate = self.dispatcher.candidate(action, args)
                        self.session.experiment_attempts += 1
                        if action.tool == "tune_experiment":
                            self.session.tuning_attempts += 1
                        output = self.directory / "experiments" / candidate.id
                        record.status = "reserved"
                        record.reservation = {
                            "candidate": candidate.model_dump(),
                            "competition": self.state.competition.model_dump(),
                            "config": worker_config.model_dump(),
                        }
                        self.checkpoint()
                        future = self.pool.submit(
                            run_isolated,
                            candidate,
                            self.state.competition.model_copy(deep=True),
                            worker_config,
                            output,
                            min(self.remaining(), worker_config.experiment_timeout),
                            progress=lambda message, task_id=record.task_id: self.worker_events.put(
                                (task_id, message)
                            ),
                            cancelled=lambda: self.cancelled() or self.remaining() <= 0,
                        )
                        training[future] = ref
                    else:
                        # All non-worker tools run on the owner thread. Never mutate
                        # validation or selection while a worker uses its snapshot.
                        if training:
                            continue
                        record.tool_snapshot = self.dispatcher.reuse_key(action, args)
                        reused = self.dispatcher.reuse(action, record.tool_snapshot)
                        if reused:
                            source, result = reused
                            self.complete(ref, {**result, "reused_from": source})
                            continue
                        record.status = "reserved"
                        self.checkpoint()
                        result = self.dispatcher.execute(action, args)
                        self.complete(ref, result)
            except DatasetError as exc:
                diagnostic = RuntimeOperationError(exc.code, getattr(exc, "details", None))
                self.complete(ref, diagnostic.diagnostic(), "failed")
                if diagnostic.code in IMMUTABLE_DATASET_ERRORS:
                    self.state.error = str(diagnostic)
                    raise StopRun(diagnostic.code) from exc
            except ToolError as exc:
                self.complete(ref, {"error": str(exc)}, "failed")
                if str(exc) == "metric_requirement_unresolved":
                    self.state.error = str(RuntimeOperationError(str(exc)))
                    raise StopRun(str(exc)) from exc
            except RuntimeOperationError as exc:
                self.complete(ref, exc.diagnostic(), "failed")
                if exc.code in IMMUTABLE_DATASET_ERRORS | {"metric_requirement_unresolved"}:
                    self.state.error = str(exc)
                    raise StopRun(exc.code) from exc
            except CheckpointError:
                raise
            except (ValueError, OSError, RuntimeError) as exc:
                # Local diagnostic only; provider context receives a controlled category.
                with (self.directory / "tool_errors.log").open("a", encoding="utf-8") as log:
                    log.write(redact(traceback.format_exc(), self.settings.llm_api_key))
                self.complete(
                    ref,
                    {"error": "tool_execution_failed", "error_type": type(exc).__name__},
                    "failed",
                )
        self.collect(training, self.commit_experiment)
        for task in self.session.tasks.values():
            if (
                task.role != "coordinator"
                and task.status == "ready"
                and any(
                    record.task_id == task.id
                    and record.result.get("error") == "repeated_action_limit"
                    for record in self.session.actions.values()
                )
            ):
                # A finish may itself fail because it depended on the rejected repeat.
                self.fail_repeating_task(task)
        return progressed

    def commit_experiment(self, ref, result):
        record = self.session.actions[ref]
        candidate = record.reservation["candidate"]
        if result is None:
            result = Experiment(
                id=ref,
                model=candidate["model"],
                params=candidate["params"],
                lineage=candidate.get("lineage"),
                status="failed",
                error="worker_failure",
            )
        if any(e.id == ref for e in self.state.experiments):
            return
        if (
            result.id != ref
            or result.params != candidate["params"]
            or result.lineage != candidate["lineage"]
        ):
            raise ToolError("worker_contract_mismatch")
        if result.status == "completed":
            self.register_artifact(
                ref, [result.checkpoint, result.oof_path, result.diagnostics_path]
            )
        self.state.experiments.append(result)
        # A new primary measurement invalidates previous selection/predictions.
        if result.validation_role == "primary":
            self.state.best_id = None
        diagnostic = (
            RuntimeOperationError(result.error_code, result.error_details)
            if result.status != "completed" and result.error_code in TRAINING_RESOURCE_ERRORS
            else None
        )
        self.complete(
            ref,
            {
                "experiment_ref": ref,
                "status": result.status,
                "score": result.score,
                "std": result.std,
                "fold_scores": result.fold_scores,
                "train_score": result.train_score,
                "model": result.model,
                "lineage": result.lineage,
                "validation_role": result.validation_role,
                "effective_pipeline": {
                    k: result.training_contract.get(k)
                    for k in (
                        "pipeline_config",
                        "effective_params",
                        "validation",
                        "metric",
                        "preprocessing",
                    )
                },
                "artifact_ref": ref if result.status == "completed" else None,
                "error": "training_failed" if result.status != "completed" else None,
                **(diagnostic.diagnostic() if diagnostic else {}),
            },
            "completed" if result.status == "completed" else "failed",
        )
        if diagnostic and diagnostic.code in IMMUTABLE_DATASET_ERRORS:
            self.state.error = str(diagnostic)
            raise StopRun(diagnostic.code)

    def recover(self):
        for ref, record in list(self.session.actions.items()):
            if record.status != "reserved":
                continue
            if record.action.kind == "delegate":
                child = self.session.tasks.get(ref)
                if child and child.status in {"completed", "failed"}:
                    finishes = [
                        r
                        for r in self.session.actions.values()
                        if r.task_id == ref
                        and r.action.kind == "finish"
                        and r.status == "completed"
                    ]
                    finish = finishes[-1].action if finishes else None
                    self.complete(
                        ref,
                        {
                            "task_id": ref,
                            "summary": finish.summary if finish else "Recovered terminal task",
                            "evidence": finish.evidence if finish else [],
                        },
                        child.status,
                    )
                continue
            if record.reservation.get("candidate"):
                output = self.directory / "experiments" / ref
                try:
                    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
                    if any(
                        request[k] != record.reservation[k]
                        for k in ("candidate", "competition", "config")
                    ):
                        raise ValueError("request mismatch")
                    result = Experiment.model_validate_json(
                        (output / "result.json").read_text(encoding="utf-8")
                    )
                    if (
                        result.status == "completed"
                        and result.training_contract.get("dataset_fingerprints")
                        != self.state.report["fingerprints"]
                    ):
                        raise ValueError("fingerprint mismatch")
                    self.commit_experiment(ref, result)
                    continue
                except (ValueError, OSError, ToolError):
                    self.commit_experiment(ref, None)
            else:
                # Local effects without a receipt are uncertain. Do not silently replay.
                self.complete(ref, {"error": "interrupted_tool_outcome_unknown"}, "failed")

    def validate_constraints(self):
        original = self.state.metadata["user_constraints"]
        resolved = self.state.metadata.get("resolved_requirements", {})
        if any(
            key != "validation" and self.state.competition.model_dump()[key] != value
            for key, value in resolved.items()
        ):
            raise StopRun("resolved_requirements_changed")
        for key, value in original["config"].items():
            if key not in {"data_dir", "submit"} and self.state.config.model_dump()[key] != value:
                raise StopRun("immutable_user_constraints_changed")
        directory = original["config"]["data_dir"]
        if directory and Path(directory).resolve() != Path(self.state.config.data_dir).resolve():
            raise StopRun("immutable_user_constraints_changed")
        for key, value in original["competition"].items():
            if key == "validation" and value == "auto":
                continue
            if value is not None and self.state.competition.model_dump()[key] != value:
                raise StopRun("immutable_user_constraints_changed")

    def coordinator_waiting(self):
        return any(
            record.task_id == "coordinator"
            and record.status in {"pending", "reserved"}
            and record.action.kind in {"wait", "finish"}
            for record in self.session.actions.values()
        )

    def release_deadlocked_barriers(self):
        # With no runnable specialist or tool, return a dependency error for the
        # coordinator to resolve. Do not spin on a cyclic wait/finish graph.
        blocked = [
            ref
            for ref, record in self.session.actions.items()
            if record.status == "pending" and record.action.kind in {"wait", "finish"}
        ]
        for ref in blocked:
            self.complete(
                ref, {"error": "coordination_deadlock", "blocked_actions": blocked[:16]}, "failed"
            )
        return bool(blocked)

    def run(self):
        try:
            self.validate_constraints()
            self.state.status, self.state.error, self.state.phase = "running", None, "agent_loop"
            self.session.termination_reason = None
            if self.state.report:
                original = self.state.report["fingerprints"]
                self.data = inspect_dataset(
                    Path(self.state.config.data_dir), self.state.competition, self.state.config
                )
                if self.data.report["fingerprints"] != original:
                    raise StopRun("dataset_fingerprint_changed")
            self.recover()
            self.checkpoint()
            while not self.session.termination_reason:
                self.guard()
                repairs = [task for task in self.session.tasks.values() if task.repair_pending]
                if repairs:
                    self.call_agents(repairs[: self.state.config.agent_concurrency])
                    continue
                self.drain_ready()
                if self.session.termination_reason:
                    break
                ready = sorted(
                    [
                        task
                        for task in self.session.tasks.values()
                        if task.role != "coordinator"
                        and task.status == "ready"
                        and all(
                            self.dependency_status(dep) == "completed" for dep in task.depends_on
                        )
                        and not any(
                            r.task_id == task.id and r.status in {"pending", "reserved"}
                            for r in self.session.actions.values()
                        )
                    ],
                    key=lambda task: task.turns,
                )
                if ready:
                    self.call_agents(ready[: self.state.config.agent_concurrency])
                else:
                    if self.release_deadlocked_barriers():
                        continue
                    coordinator = self.session.tasks["coordinator"]
                    self.call_agents([coordinator])
                # Coordinator receives feedback after every specialist/tool round,
                # even when other specialists have independent work pending.
                self.drain_ready()
                if any(task.repair_pending for task in self.session.tasks.values()):
                    continue
                if not self.session.termination_reason and ready and not self.coordinator_waiting():
                    self.call_agents([self.session.tasks["coordinator"]])
            self.state.status = (
                "completed"
                if self.session.termination_reason == "coordinator_finished"
                else "incomplete"
            )
        except StopRun as exc:
            self.session.termination_reason = str(exc)
            self.state.status = "interrupted" if str(exc) == "cancelled" else "incomplete"
        except KeyboardInterrupt:
            self.session.termination_reason, self.state.status = "interrupted", "interrupted"
        except Exception:
            self.session.termination_reason, self.state.status = "runtime_failure", "failed"
            self.state.error = "Agentic runtime failed; inspect the local structured trace."
            raise
        finally:
            self.stopping.set()
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.state.phase = "finished" if self.state.status == "completed" else "checkpointed"
            self.checkpoint(journal=True)
            self.report()
        return self.state

    def report(self):
        s = self.state
        # Factual report is computed by trusted runtime, never copied from LLM prose.
        write_json(
            self.directory / "summary.json",
            {
                "run_id": s.run_id,
                "execution_mode": "agentic",
                "protocol": VERSION,
                "status": s.status,
                "termination_reason": self.session.termination_reason,
                "error": s.error,
                "best_id": s.best_id,
                "submission_path": s.submission_path,
                "feedback": s.feedback,
                "experiments": [e.model_dump() for e in s.experiments],
                "budget": {
                    "requests": self.session.requests,
                    "tokens_charged": self.session.tokens,
                    "experiment_attempts": self.session.experiment_attempts,
                    "turns": self.session.turns,
                    "schema_repairs": self.session.schema_repairs,
                    "invalid_actions": self.session.invalid_actions,
                    "elapsed_seconds": s.elapsed_seconds,
                },
                "provider_usage": self.session.usage,
                "evidence": {
                    ref: record.result
                    for ref, record in self.session.actions.items()
                    if record.status == "completed" and record.action.kind == "tool"
                },
            },
        )


def new_session(run_id):
    return SwarmState(
        session_id=run_id,
        counter_version=2,
        tasks={
            "coordinator": Task(
                id="coordinator",
                role="coordinator",
                goal=(
                    "Design and execute a supported competition pipeline within the immutable user "
                    "constraints. Delegate inspection/validation, modeling, evaluation and validated "
                    "predictions as needed. Finish with recorded measured evidence or an explicit incomplete outcome."
                ),
            ),
        },
    ).model_dump(mode="json")


def recover_receipt(store, state):
    receipt = store.run_dir(state.run_id) / "swarm_receipt.json"
    if receipt.exists():
        recovered = RunState.model_validate_json(receipt.read_text(encoding="utf-8"))
        if recovered.run_id != state.run_id or recovered.metadata.get(
            "user_constraints"
        ) != state.metadata.get("user_constraints"):
            raise ValueError("Swarm recovery receipt mismatch")
        if recovered.metadata.get("swarm_commit_sequence", 0) > state.metadata.get(
            "swarm_commit_sequence", 0
        ):
            # Current explicit submission authorization wins over a recovery snapshot.
            recovered.config.submit = state.config.submit
            store.save(recovered)
            return recovered
    return state
