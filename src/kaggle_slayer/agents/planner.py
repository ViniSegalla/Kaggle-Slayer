import optuna
from optuna.trial import TrialState

from kaggle_slayer.agents.base import Agent
from kaggle_slayer.llm.provider import PlannerLLM
from kaggle_slayer.schemas import Plan
from kaggle_slayer.schemas.models import Candidate
from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.evaluation_tools import direction, rank_experiments


class CompetitionPlanner(Agent):
    name = "competition_planner"

    def initialize(self, state):
        self.log(
            state,
            "request_data_inspection",
            "Scope is tabular CSV; resolve target, template and metric before training.",
            "competition_profile",
            state.competition.model_dump(),
            "data_agent.inspect",
        )

    def plan(self, state):
        advice, warning = PlannerLLM().advise(
            {
                "task": state.competition.task,
                "metric": state.competition.metric,
                "quality": {
                    key: state.report[key]
                    for key in ("train_rows", "test_rows", "dtypes", "missing_fraction", "warnings")
                },
                "allowed_models": state.config.models,
            }
        )
        if warning:
            state.warnings.append(warning)
        models = list(dict.fromkeys((advice.priorities if advice else []) + state.config.models))
        models = [model for model in models if model in state.config.models]
        # Always establish a simple linear baseline before complex candidates.
        if "linear" in models:
            models.remove("linear")
            models.insert(0, "linear")
        queue = [
            Candidate(
                id=f"baseline-{model}",
                model=model,
                rationale="Measure a distinct model family on fixed folds",
            )
            for model in models[: state.config.max_experiments - int(state.config.ensemble)]
        ]
        strategy = (
            advice.rationale
            if advice
            else (
                "Compare a linear baseline with nonlinear models on identical fixed folds, "
                "then tune the leading family with seeded Optuna trials. Fit every transform within training folds."
            )
        )
        state.plan = Plan(
            strategy=strategy,
            queue=queue,
            validation=state.competition.validation,
            direction=direction(state.competition.metric),
            source="llm" if advice else "deterministic",
        )
        if advice:
            state.warnings.extend(advice.cautions)
        self.log(
            state,
            "schedule_baselines",
            strategy,
            "model_registry",
            state.plan.model_dump(),
            "modeling_agent.train",
        )

    def study(self, state):
        path = self.store.run_dir(state.run_id) / "optimization.sqlite"
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return optuna.create_study(
            study_name="optimization",
            storage=f"sqlite:///{path.as_posix()}",
            load_if_exists=True,
            direction=state.plan.direction,
            sampler=optuna.samplers.TPESampler(seed=state.config.seed + state.iteration),
        )

    def feedback_plan(self, state) -> bool:
        remaining = (
            state.config.max_experiments - len(state.experiments) - int(state.config.ensemble)
        )
        if remaining <= 0 or state.iteration >= state.config.max_iterations:
            return False
        ranking = rank_experiments(state.experiments, state.competition.metric)
        if not ranking:
            return False
        best = ranking[0]
        study = self.study(state)
        queue = []
        # Resume an already asked but uncheckpointed trial instead of abandoning it.
        pending = [trial for trial in study.get_trials() if trial.state == TrialState.RUNNING]
        for _ in range(min(remaining, state.config.optuna_trials)):
            trial = optuna.trial.Trial(study, pending.pop(0)._trial_id) if pending else study.ask()
            if best.model == "linear":
                key = "C" if state.competition.task == "classification" else "alpha"
                params = {key: trial.suggest_float(key, 0.01, 100, log=True)}
            elif best.model == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 240, step=20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                    "max_depth": trial.suggest_int("max_depth", 3, 18),
                    "max_features": trial.suggest_float("max_features", 0.5, 1),
                }
            elif best.model == "hist_gradient":
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31),
                    "l2_regularization": trial.suggest_float(
                        "l2_regularization", 0.001, 10, log=True
                    ),
                }
            else:
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
                }
                if best.model == "lightgbm":
                    params["num_leaves"] = trial.suggest_int("num_leaves", 7, 31)
                    params["reg_lambda"] = trial.suggest_float("reg_lambda", 0.01, 10, log=True)
                else:
                    params["depth" if best.model == "catboost" else "max_depth"] = (
                        trial.suggest_int("depth", 3, 7)
                    )
            queue.append(
                Candidate(
                    id=f"tune-{trial.number}",
                    model=best.model,
                    params=params,
                    parent_id=best.id,
                    trial_number=trial.number,
                    rationale=state.feedback.get("recommendation", "Tune the best CV family"),
                )
            )
        state.plan.queue = queue
        state.iteration += 1
        self.log(
            state,
            "revise_experiment_queue",
            "Evaluator evidence selects the tuning family; public leaderboard is diagnostic only.",
            "optuna.ask",
            {"feedback": state.feedback, "queue": [x.model_dump() for x in queue]},
            "modeling_agent.train",
        )
        return bool(queue)

    def record_trial(self, state, candidate, experiment):
        if candidate.trial_number is not None:
            study = self.study(state)
            trial = next(t for t in study.get_trials() if t.number == candidate.trial_number)
            if trial.state == TrialState.RUNNING:
                if experiment.status == "completed":
                    study.tell(candidate.trial_number, experiment.score)
                else:
                    study.tell(candidate.trial_number, state=TrialState.FAIL)

    def ensemble_candidate(self, state):
        if not state.config.ensemble or len(state.experiments) >= state.config.max_experiments:
            return None
        if any(e.id == "ensemble-soft" for e in state.experiments):
            return None
        members = []
        for experiment in rank_experiments(state.experiments, state.competition.metric):
            if experiment.model not in {item["model"] for item in members}:
                members.append(
                    {
                        "model": experiment.model,
                        "params": experiment.effective_params or experiment.params,
                    }
                )
            if len(members) == 2:
                break
        if len(members) < 2:
            return None
        candidate = Candidate(
            id="ensemble-soft",
            model="ensemble",
            params={"members": members},
            rationale="Test equal-weight voting/averaging of the strongest two distinct validated families on the same folds; select only if it beats existing candidates.",
        )
        self.log(
            state,
            "propose_ensemble",
            candidate.rationale,
            "validated_family_registry",
            candidate.model_dump(),
            "modeling_agent.train",
        )
        return candidate

    def leaderboard_followup(self, state):
        feedback = state.feedback.get("leaderboard", {})
        if feedback.get("public_score") is None:
            return
        overrides = {}
        priorities = [
            "Keep fixed local validation as the selection criterion; never optimize on repeated public submissions"
        ]
        if feedback.get("unexpected_degradation"):
            priorities.append(
                "Audit train/test distribution shift and compare a fresh, stricter validation strategy before changing model complexity"
            )
            if state.competition.feature_profile == "titanic":
                overrides = {"validation": "group", "group_column": "Ticket"}
                priorities.append(
                    "Compare group CV by Ticket with the current stratified CV; related passengers may make random folds optimistic"
                )
        if state.report.get("duplicate_rows", 0):
            priorities.append(
                f"Inspect {state.report['duplicate_rows']} duplicate feature rows and their validation group assignments"
            )
        priorities.append(
            "Test stronger regularization in the leading model only after the validation audit"
        )
        followup = {
            "status": "proposed_only",
            "evidence": feedback,
            "priorities": priorities,
            "suggested_competition_overrides": overrides,
            "experiment_budget_remaining": max(
                0, state.config.max_experiments - len(state.experiments)
            ),
            "submission_budget_remaining": max(
                0, state.config.max_submissions - len(state.submissions)
            ),
            "execute_automatically": False,
            "reason": "This run's experiment/submission limits are retained; diagnostics are for a separately budgeted run",
        }
        state.feedback["planner_followup"] = followup
        path = self.store.run_dir(state.run_id) / "next_run_plan.json"
        write_json(path, followup)
        self.log(
            state,
            "accept_leaderboard_diagnostic_feedback",
            "Measured public/local discrepancy informs a validation audit, not public-score optimization.",
            "leaderboard_to_planner",
            followup,
            "finish_with_deferred_plan",
            [str(path)],
        )
