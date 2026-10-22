import math
import time
from datetime import datetime, timezone

from kaggle_slayer.agents.base import Agent
from kaggle_slayer.tools.evaluation_tools import direction
from kaggle_slayer.tools.kaggle_client import KaggleError


def field(row, *names):
    return next((row[name] for name in names if row.get(name) is not None), None)


class LeaderboardAgent(Agent):
    name = "leaderboard_reviewer"

    def review(self, state, client, *, refresh=False):
        self.activity(state, "Checking submission feedback and available account standing")
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
        # Account standing can reflect a different run's best submission. Do not
        # estimate rank by matching rounded scores on a partial leaderboard page.
        try:
            metadata = client.metadata(state.competition.competition)
            rank = field(metadata, "userRank", "user_rank")
            teams = field(metadata, "teamCount", "team_count")
            valid_rank = isinstance(rank, int) and not isinstance(rank, bool) and rank > 0
            valid_total = isinstance(teams, int) and not isinstance(teams, bool) and teams > 0
            report.update(
                rank=rank if valid_rank else None,
                team_count=teams if valid_total else None,
                standing_scope="current_account_best",
                standing_source="kaggle.competitions_list",
                standing_checked_at=datetime.now(timezone.utc).isoformat(),
            )
            if valid_rank and valid_total and rank <= teams:
                report["top_percent"] = 100 * rank / teams
                report["top_20_percent"] = rank / teams <= 0.2
            else:
                report["standing_note"] = (
                    "Kaggle did not return a usable account rank and team count; "
                    "a percentile cannot be established."
                )
        except KaggleError as exc:
            report["standing_note"] = str(exc)
        state.feedback["leaderboard"] = report
        self.log(
            state,
            "review_kaggle_feedback",
            "Associate this run's score separately from the account's current competition standing.",
            "kaggle.competition_submissions + kaggle.competitions_list",
            report,
            "planner_feedback_for_next_run",
        )
