from kaggle_slayer.agents.base import Agent
from kaggle_slayer.tools.training_tools import run_isolated


class ModelingAgent(Agent):
    name = "modeler"

    def train(self, state, candidate, remaining_seconds):
        output = self.store.run_dir(state.run_id) / "experiments" / candidate.id
        self.log(
            state,
            "start_experiment",
            candidate.rationale,
            "isolated_cv_worker",
            {
                "candidate": candidate.model_dump(),
                "timeout": min(remaining_seconds, state.config.experiment_timeout),
            },
            "evaluator.assess",
        )
        result = run_isolated(
            candidate,
            state.competition,
            state.config,
            output,
            min(remaining_seconds, state.config.experiment_timeout),
        )
        state.experiments.append(result)
        self.log(
            state,
            "experiment_completed" if result.status == "completed" else "experiment_failed",
            "Record measured validation and train scores; preserve failures for review.",
            "isolated_cv_worker",
            result.model_dump(),
            "evaluator.assess",
            [
                x
                for x in (
                    result.checkpoint,
                    result.oof_path,
                    result.importance_path,
                    str(output / "worker.log"),
                )
                if x
            ],
            level="info" if result.status == "completed" else "warning",
        )
        return result
