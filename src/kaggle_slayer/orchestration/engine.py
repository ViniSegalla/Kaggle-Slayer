import hashlib
import platform
import time
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4

from kaggle_slayer.agents.data_agent import DataAgent
from kaggle_slayer.agents.evaluator import EvaluationAgent
from kaggle_slayer.agents.leaderboard_agent import LeaderboardAgent, field
from kaggle_slayer.agents.modeling_agent import ModelingAgent
from kaggle_slayer.agents.planner import CompetitionPlanner
from kaggle_slayer.orchestration.progress import RunProgress
from kaggle_slayer.schemas import Decision, RunState
from kaggle_slayer.schemas.models import Candidate, now
from kaggle_slayer.storage.store import Store, write_json
from kaggle_slayer.tools.kaggle_client import KaggleClient, KaggleError


class Engine:
    def __init__(self, store: Store, client=None, progress=None, llm=None):
        self.store = store
        self.client = client or KaggleClient()
        self.progress = progress or (lambda message: None)
        self.live = RunProgress(store, self.progress)
        self.planner = CompetitionPlanner(store)
        self.data_agent = DataAgent(store)
        self.modeler = ModelingAgent(store)
        self.evaluator = EvaluationAgent(store)
        self.reviewer = LeaderboardAgent(store)
        self.llm = llm

    def create(self, competition, config):
        run_id = f"{competition.competition}-{uuid4().hex[:12]}"
        state = RunState(run_id=run_id, competition=competition, config=config)
        state.metadata["execution_mode"] = config.execution_mode
        if config.execution_mode == "agentic":
            from kaggle_slayer.orchestration.swarm import new_session

            state.swarm = new_session(run_id)
            state.metadata["user_constraints"] = {
                "competition": competition.model_dump(),
                "config": config.model_dump(),
            }
        state.metadata["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {dist.metadata["Name"]: dist.version for dist in distributions()},
        }
        self.store.save(state)
        return state

    def execute(self, state):
        with self.store.lock(state.run_id):
            for agent in (
                self.planner,
                self.data_agent,
                self.modeler,
                self.evaluator,
                self.reviewer,
            ):
                agent.progress = lambda name, message: self.live.emit(state.run_id, name, message)
            return self._execute(state)

    def _execute(self, state):
        if state.config.execution_mode != state.metadata.get("execution_mode", "deterministic"):
            raise ValueError("Saved execution mode is immutable; create a new run to change mode")
        if state.status == "completed" and not (
            state.config.execution_mode == "agentic"
            and state.submissions
            and state.submissions[-1]["status"] == "intent_recorded"
        ):
            return state
        if state.config.execution_mode == "agentic":
            from kaggle_slayer.orchestration.swarm import SwarmRuntime, recover_receipt

            if not state.swarm:
                raise ValueError("Agentic session missing; create a new agentic run")
            state = recover_receipt(self.store, state)
            runtime = SwarmRuntime(self, state, self.llm)
            if state.swarm.get("termination_reason") == "coordinator_finished":
                self.finish_agentic(runtime)
                runtime.pool.shutdown(wait=False)
                return runtime.state
            result = runtime.run()
            if result.status == "completed":
                self.finish_agentic(runtime)
            return result
        started, previous_elapsed = time.monotonic(), state.elapsed_seconds

        def remaining():
            return state.config.max_seconds - previous_elapsed - (time.monotonic() - started)

        def checkpoint(phase=None):
            if phase:
                state.phase = phase
                self.live.emit(state.run_id, "orchestrator", f"Phase: {phase}")
            state.elapsed_seconds = previous_elapsed + time.monotonic() - started
            self.store.save(state)

        try:
            state.status, state.error = "running", None
            self.live.emit(
                state.run_id,
                "orchestrator",
                f"Starting/resuming | experiment budget={state.config.max_experiments} | "
                f"recorded={len(state.experiments)} | remaining wall budget={max(0, remaining()):.0f}s",
            )
            checkpoint()
            if state.phase == "initialized":
                self.planner.initialize(state)
            data = self.data_agent.prepare(state, self.client)
            if state.plan is None:
                checkpoint("data_prepared")
                self.planner.plan(state)
                checkpoint("planned")
            if state.phase not in {"submission_ready", "submitting", "reviewing"}:
                while True:
                    checkpoint("training")
                    done = {experiment.id for experiment in state.experiments}
                    for candidate in state.plan.queue:
                        if candidate.id in done:
                            self.live.emit(
                                state.run_id,
                                "orchestrator",
                                f"Reusing recorded experiment {candidate.id}",
                            )
                            continue
                        if (
                            remaining() <= 0
                            or len(state.experiments) >= state.config.max_experiments
                        ):
                            break
                        result = self.modeler.train(state, candidate, remaining())
                        self.planner.record_trial(state, candidate, result)
                        checkpoint()
                    checkpoint("evaluating")
                    self.evaluator.evaluate(state)
                    checkpoint()
                    if remaining() < 3 or not self.planner.feedback_plan(state):
                        break
                    checkpoint("planned")
                if remaining() <= 0:
                    state.warnings.append(
                        "Run wall budget exhausted; using strongest completed checkpoint"
                    )
                self.live.emit(
                    state.run_id,
                    "competition_planner",
                    f"Primary search complete | experiments recorded={len(state.experiments)} | "
                    f"tuning rounds={state.iteration}; checking ensemble and confirmation budgets",
                )
                ensemble = self.planner.ensemble_candidate(state)
                if ensemble and remaining() > 3:
                    self.modeler.train(state, ensemble, remaining())
                    checkpoint("evaluating")
                    self.evaluator.evaluate(state)
                    checkpoint()
                if len(state.experiments) < 3:
                    state.warnings.append("Budget ended before three model families completed")
                if not any(e.id.startswith("tune-") for e in state.experiments):
                    state.warnings.append(
                        "No optimization trial completed; increase experiment/time/iteration budget"
                    )
                # Persist the frozen audit queue so recovery never changes finalists.
                if "confirmation_queue" not in state.metadata:
                    state.metadata["confirmation_queue"] = [
                        c.model_dump() for c in self.planner.confirmation_candidates(state)
                    ]
                checkpoint("confirming")
                for raw in state.metadata["confirmation_queue"]:
                    candidate = Candidate.model_validate(raw)
                    if any(e.id == candidate.id for e in state.experiments):
                        continue
                    if remaining() < 3 or len(state.experiments) >= state.config.max_experiments:
                        break
                    self.modeler.train(state, candidate, remaining())
                    checkpoint()
                self.evaluator.review_stability(state)
                self.evaluator.create_submission(state, data)
                checkpoint("submission_ready")
            self.upload_authorized(state, checkpoint)
            if not state.config.submit:
                self.live.emit(
                    state.run_id,
                    "orchestrator",
                    "No upload authorized; keeping the validated submission locally",
                )
            if state.submissions and state.submissions[-1]["status"] == "intent_recorded":
                state.submissions[-1]["status"] = "outcome_unknown"
                state.warnings.append(
                    "Interrupted upload: inspect Kaggle history. Automatic resubmission is disabled."
                )
            self.reviewer.review(state, self.client)
            self.planner.leaderboard_followup(state)
            state.status = "completed"
            checkpoint("finished")
            write_json(
                self.store.run_dir(state.run_id) / "summary.json",
                {
                    "run_id": state.run_id,
                    "competition": state.competition.competition,
                    "data_source": state.metadata.get("data_source"),
                    "provenance": state.metadata.get("provenance"),
                    "best_id": state.best_id,
                    "feedback": state.feedback,
                    "submission_path": state.submission_path,
                    "warnings": state.warnings,
                    "experiments": [e.model_dump() for e in state.experiments],
                },
            )
        except (Exception, KeyboardInterrupt) as exc:
            state.status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            state.error = f"{type(exc).__name__}: {exc}"
            self.store.record(
                state.run_id,
                Decision(
                    agent="orchestrator",
                    input_context={"phase": state.phase},
                    selected_action="checkpoint_failure",
                    rationale="Preserve completed work for an explicit resume.",
                    tool_invoked="engine",
                    tool_result={"error": state.error},
                    next_action="inspect_or_resume",
                    level="error",
                ),
            )
            checkpoint()
            raise
        return state

    def upload_authorized(self, state, checkpoint, remaining=None, cancelled=lambda: False):
        if state.config.submit and state.config.max_submissions and not state.submissions:
            if state.config.execution_mode == "agentic" and (
                state.metadata.get("swarm_submission_validated")
                != state.metadata.get("submission_sha256")
                or not state.metadata.get("swarm_submission_validated")
            ):
                raise ValueError("Agentic predictions require recorded validation before upload")
            self.live.emit(
                state.run_id,
                "orchestrator",
                "Submission authorized: validating integrity before one Kaggle upload",
            )
            provenance = state.metadata.get("provenance", {})
            if provenance.get("kind") in {"synthetic_fixture", "public_dataset_demo"}:
                raise ValueError(
                    "Demo/fixture datasets cannot be submitted to Kaggle; use actual competition files in a new run"
                )
            attempt = {
                # Submission bytes must match the evaluator's validated artifact.
                "status": "intent_recorded",
                "message": f"KaggleSlayer run {state.run_id}",
                "timestamp": now(),
                "public_score": None,
                "reference": None,
            }
            expected_hash = state.metadata.get("submission_sha256")
            if (
                not expected_hash
                or hashlib.sha256(Path(state.submission_path).read_bytes()).hexdigest()
                != expected_hash
            ):
                raise ValueError(
                    "Submission integrity check failed; regenerate and validate predictions before uploading"
                )
            # Persist intent before making a non-idempotent API call.
            state.submissions.append(attempt)
            checkpoint("submitting")
            try:
                if remaining is not None:
                    from kaggle_slayer.tools.runtime_worker import run_bounded

                    response = run_bounded(
                        "upload",
                        state,
                        self.store.run_dir(state.run_id) / "operations" / "upload",
                        remaining(),
                        cancelled,
                    )["response"]
                else:
                    response = self.client.submit(
                        state.competition.competition,
                        Path(state.submission_path),
                        attempt["message"],
                        authorized=True,
                    )
                reference = field(response, "ref", "id", "submissionId", "submission_id")
                attempt.update(
                    status="submitted" if reference else "outcome_unknown",
                    reference=reference,
                    response=response,
                )
            except (KaggleError, RuntimeError) as exc:
                attempt.update(status="outcome_unknown", error=str(exc))
                state.warnings.append(str(exc))
            checkpoint("reviewing")

    def finish_agentic(self, runtime):
        """Trusted terminal I/O; LLM actions cannot enter the upload path."""
        state = runtime.state
        state.status = "running"
        try:
            runtime.validate_constraints()

            def cancelled():
                return (runtime.directory / "cancel.request").exists()

            if cancelled():
                state.status, runtime.session.termination_reason = "interrupted", "cancelled"
                return
            if state.config.submit and not state.submissions and runtime.remaining() <= 0:
                state.status = "incomplete"
                runtime.session.termination_reason = "wall_time_exhausted_before_upload"
                return
            self.upload_authorized(
                state,
                lambda phase=None: runtime.checkpoint(journal=True),
                remaining=runtime.remaining,
                cancelled=cancelled,
            )
            if state.submissions and state.submissions[-1]["status"] == "intent_recorded":
                state.submissions[-1]["status"] = "outcome_unknown"
                state.warnings.append("Interrupted upload; automatic resubmission disabled")
            if cancelled():
                state.status, runtime.session.termination_reason = "interrupted", "cancelled"
                return
            if state.submissions and runtime.remaining() > 0:
                from kaggle_slayer.tools.runtime_worker import run_bounded

                try:
                    feedback = run_bounded(
                        "feedback",
                        state,
                        runtime.directory / "operations" / "final-feedback",
                        runtime.remaining(),
                        cancelled,
                    )
                    state.feedback, state.submissions, state.warnings = (
                        feedback["feedback"],
                        feedback["submissions"],
                        feedback["warnings"],
                    )
                except RuntimeError:
                    state.warnings.append("Feedback unavailable within the remaining run deadline")
            elif not state.submissions:
                self.reviewer.review(state, self.client)
            state.status = "completed"
        except (ValueError, RuntimeError):
            state.status = "failed"
            runtime.session.termination_reason = "terminal_policy_or_integrity_failure"
            raise
        finally:
            state.phase = "finished" if state.status == "completed" else "checkpointed"
            runtime.checkpoint(journal=True)
            runtime.report()
