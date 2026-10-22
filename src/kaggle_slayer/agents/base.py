from kaggle_slayer.orchestration.progress import decision_summary
from kaggle_slayer.schemas import Decision


class Agent:
    name = "agent"

    def __init__(self, store, progress=None):
        self.store = store
        self.progress = progress or (lambda agent, message: None)

    def activity(self, state, message):
        self.progress(self.name, message)

    def log(
        self,
        state,
        action,
        rationale,
        tool,
        result,
        next_action,
        artifacts=None,
        level="info",
        context=None,
    ):
        self.store.record(
            state.run_id,
            Decision(
                agent=self.name,
                input_context=context
                or {
                    "phase": state.phase,
                    "iteration": state.iteration,
                    "competition": state.competition.competition,
                },
                selected_action=action,
                rationale=rationale,
                tool_invoked=tool,
                tool_result=result,
                artifacts=artifacts or [],
                next_action=next_action,
                level=level,
            ),
        )
        self.activity(state, decision_summary(action, result))
