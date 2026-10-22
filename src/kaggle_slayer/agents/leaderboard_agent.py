import math
import time

from kaggle_slayer.agents.base import Agent
from kaggle_slayer.tools.evaluation_tools import direction
from kaggle_slayer.tools.kaggle_client import KaggleError


def field(row, *names):
    return next((row[name] for name in names if row.get(name) is not None), None)


class LeaderboardAgent(Agent):
    name = "leaderboard_reviewer"

    def review(self, state, client, *, refresh=False):
        if not state.submissions:
            state.feedback["leaderboard"] = {
                "status": "not_submitted",
                "public_score": None,
                "rank": None,
                "top_20_percent": None,
            }
            self.log(
                state,
                "review_local_evidence",
                "No Kaggle feedback exists; performance target is unmeasured.",
                "submission_history",
                state.feedback["leaderboard"],
                "finish",
            )
            return
        attempt = state.submissions[-1]
        for index in range(max(1 if refresh else 0, state.config.poll_attempts)):
            try:
                rows = client.submissions(state.competition.competition)
                matches = [
                    row
                    for row in rows
                    if str(field(row, "ref", "id", "submissionId", "submission_id"))
                    == str(attempt.get("reference"))
                    or field(row, "description", "submissionDescription", "submission_description")
                    == attempt["message"]
                ]
                if matches:
                    actual = matches[0]
                    raw_score = field(actual, "publicScore", "public_score")
                    try:
                        public_score = float(raw_score)
                        if not math.isfinite(public_score):
                            public_score = None
                    except (ValueError, TypeError):
                        public_score = None
                    attempt.update(
                        {
                            "feedback": actual,
                            "public_score": public_score,
                            "status": str(field(actual, "status") or "pending"),
                        }
                    )
                    if public_score is not None:
                        break
            except KaggleError as exc:
                state.warnings.append(str(exc))
                break
            if index + 1 < state.config.poll_attempts:
                time.sleep(state.config.poll_interval)
        public_score = attempt.get("public_score")
        report = {
            "status": attempt["status"],
            "public_score": public_score,
            "rank": None,
            "top_20_percent": None,
            "recommendation": "Keep local CV as the selection criterion; inspect distribution shift before any new run.",
        }
        if public_score is not None:
            sign = 1 if direction(state.competition.metric) == "maximize" else -1
            report["validation_minus_public"] = sign * (
                state.feedback["local_score"] - public_score
            )
            report["unexpected_degradation"] = report["validation_minus_public"] > max(
                abs(state.feedback["local_score"]) * 0.1, 0.02
            )
        # API leaderboard pages are partial. Never infer a percentile from a partial page.
        state.feedback["leaderboard"] = report
        self.log(
            state,
            "review_kaggle_feedback",
            "Associate only this run's submission; unavailable rank and percentile remain unknown.",
            "kaggle.competition_submissions",
            report,
            "planner_feedback_for_next_run",
        )
