"""Trusted fixed operations. Child memory is isolated; this worker never opens SQLite."""

import json
import sys
from pathlib import Path

from kaggle_slayer.agents.data_agent import DataAgent
from kaggle_slayer.agents.leaderboard_agent import LeaderboardAgent
from kaggle_slayer.schemas import RunState
from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.kaggle_client import KaggleClient


class LocalArtifacts:
    def __init__(self, directory):
        self.directory = directory

    def run_dir(self, _):
        return self.directory

    def record(self, *_):
        pass


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    state = RunState.model_validate(request["state"])
    local = LocalArtifacts(Path(request["output"]))
    client = KaggleClient()
    try:
        if request["operation"] == "prepare":
            DataAgent(local).prepare(state, client)
            result = {
                "competition": state.competition.model_dump(),
                "config": state.config.model_dump(),
                "metadata": state.metadata,
                "report": state.report,
                "warnings": state.warnings,
            }
        elif request["operation"] == "feedback":
            LeaderboardAgent(local).review(state, client)
            result = {
                "feedback": state.feedback,
                "submissions": state.submissions,
                "warnings": state.warnings,
            }
        elif request["operation"] == "upload":
            if (
                not state.config.submit
                or len(state.submissions) != 1
                or state.submissions[0]["status"] != "intent_recorded"
            ):
                raise ValueError("authorization")
            result = {
                "response": client.submit(
                    state.competition.competition,
                    Path(state.submission_path),
                    state.submissions[0]["message"],
                    authorized=True,
                )
            }
        else:
            raise ValueError("operation")
    except Exception:
        result = {"error": "bounded_operation_failed"}
    write_json(Path(sys.argv[2]), result)


if __name__ == "__main__":
    main()
