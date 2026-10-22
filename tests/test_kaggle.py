import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kaggle_slayer.agents.leaderboard_agent import LeaderboardAgent
from kaggle_slayer.schemas import RunState
from kaggle_slayer.storage.store import Store
from kaggle_slayer.tools.kaggle_client import KaggleClient, KaggleError


def test_auth_missing_is_actionable(monkeypatch):
    monkeypatch.setattr("kaggle_slayer.tools.kaggle_client.credentials_present", lambda: False)
    with pytest.raises(KaggleError, match="credentials unavailable"):
        KaggleClient().metadata("titanic")


def test_mock_official_api_and_authorization(tmp_path):
    api = Mock()
    api.competitions_list.return_value = [
        SimpleNamespace(ref="titanic", evaluationMetric="Accuracy")
    ]
    api.competition_submit.return_value = {"ref": 123}
    api.competition_submissions.return_value = [{"ref": 123, "publicScore": "0.8"}]
    client = KaggleClient(api)
    assert client.metadata("titanic")["evaluationMetric"] == "Accuracy"
    with pytest.raises(KaggleError, match="explicit"):
        client.submit("titanic", tmp_path / "submission.csv", "test", authorized=False)
    assert not api.competition_submit.called
    assert (
        client.submit("titanic", tmp_path / "submission.csv", "test", authorized=True)["ref"] == 123
    )
    assert client.submissions("titanic")[0]["publicScore"] == "0.8"


def test_current_sdk_competition_response_wrapper():
    # Importing kagglesdk types does not authenticate or contact Kaggle.
    from kagglesdk.competitions.types.competition_api_service import (
        ApiCompetition,
        ApiListCompetitionsResponse,
    )

    competition = ApiCompetition()
    competition.ref = "https://www.kaggle.com/competitions/titanic"
    competition.evaluation_metric = "Accuracy"
    response = ApiListCompetitionsResponse()
    response.competitions = [competition]
    api = Mock()
    api.competitions_list.return_value = response
    assert KaggleClient(api).metadata("titanic")["evaluationMetric"] == "Accuracy"


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"evaluationMetric": "Categorization Accuracy"}, "accuracy"),
        ({"evaluationMetric": "Roc Auc Score"}, "roc_auc"),
        ({"evaluationMetric": "ROC-AUC"}, "roc_auc"),
        ({"evaluation_metric": "roc_auc_score"}, "roc_auc"),
        ({"evaluationMetric": "Area Under The ROC Curve"}, "roc_auc"),
    ],
)
def test_actual_metric_name_inference(fixture_data, metadata, expected):
    from kaggle_slayer.agents.data_agent import DataAgent

    directory, c, config = fixture_data
    c.metric, config.data_dir = None, None
    state = RunState(run_id="metadata-inference", competition=c, config=config)
    client = Mock()
    client.metadata.return_value = metadata
    client.download.return_value = directory
    DataAgent(Store(config.artifact_root)).prepare(state, client)
    assert state.competition.metric == expected


def test_unknown_official_metric_requires_explicit_profile(fixture_data):
    from kaggle_slayer.agents.data_agent import DataAgent
    from kaggle_slayer.tools.data_tools import DatasetError

    directory, c, config = fixture_data
    c.metric, config.data_dir = None, None
    config.execution_mode = "agentic"
    state = RunState(run_id="unknown-metric", competition=c, config=config)
    client = Mock()
    client.metadata.return_value = {"evaluationMetric": "Unrecognized Metric"}
    client.download.return_value = directory
    with pytest.raises(DatasetError) as caught:
        DataAgent(Store(config.artifact_root)).prepare(state, client)
    assert caught.value.code == "metric_requirement_unresolved"
    assert state.competition.metric is None


def test_download_extracts_csv_and_rejects_escape(tmp_path):
    def archive(slug, path, quiet):
        with zipfile.ZipFile(Path(path) / f"{slug}.zip", "w") as bundle:
            bundle.writestr("train.csv", "x,y\n1,0\n")

    api = Mock()
    api.competition_download_files.side_effect = archive
    client = KaggleClient(api)
    assert (client.download("safe", tmp_path / "safe") / "train.csv").exists()

    def unsafe(slug, path, quiet):
        with zipfile.ZipFile(Path(path) / f"{slug}.zip", "w") as bundle:
            bundle.writestr("../escape.csv", "x\n1\n")

    api.competition_download_files.side_effect = unsafe
    with pytest.raises(KaggleError, match="Unsafe"):
        client.download("unsafe", tmp_path / "unsafe")


def test_upload_never_retries(tmp_path):
    api = Mock()
    api.competition_submit.side_effect = ConnectionError("connection dropped")
    with pytest.raises(KaggleError, match="uncertain"):
        KaggleClient(api).submit("titanic", tmp_path / "submission.csv", "run", authorized=True)
    assert api.competition_submit.call_count == 1


def test_permission_error_does_not_retry():
    api = Mock()
    error = RuntimeError("forbidden")
    error.status = 403
    api.competitions_list.side_effect = error
    with pytest.raises(KaggleError, match="manually accept"):
        KaggleClient(api).metadata("titanic")
    assert api.competitions_list.call_count == 1


def test_feedback_matches_this_run_only(fixture_data, tmp_path):
    _, c, config = fixture_data
    store = Store(tmp_path / "review")
    state = RunState(
        run_id="review-test",
        competition=c,
        config=config,
        submissions=[{"status": "submitted", "reference": 12, "message": "run review-test"}],
        feedback={"local_score": 0.85},
    )
    state.config.poll_attempts = 1
    api = Mock()
    api.competition_submissions.return_value = [
        {"ref": 1, "publicScore": 0.99, "description": "other"},
        {"ref": 12, "publicScore": 0.8, "status": "complete"},
    ]
    api.competitions_list.return_value = []
    LeaderboardAgent(store).review(state, KaggleClient(api))
    assert state.feedback["leaderboard"]["public_score"] == 0.8
    assert state.feedback["leaderboard"]["top_20_percent"] is None


def test_planner_consumes_realistic_discrepancy_without_training(fixture_data):
    from kaggle_slayer.agents.planner import CompetitionPlanner

    _, c, config = fixture_data
    state = RunState(
        run_id="feedback-plan",
        competition=c,
        config=config,
        report={"duplicate_rows": 12},
        feedback={
            "leaderboard": {
                "public_score": 0.75,
                "validation_minus_public": 0.09,
                "unexpected_degradation": True,
            }
        },
    )
    store = Store(config.artifact_root)
    CompetitionPlanner(store).leaderboard_followup(state)
    plan = state.feedback["planner_followup"]
    assert plan["suggested_competition_overrides"]["group_column"] == "Ticket"
    assert not plan["execute_automatically"] and not state.experiments
    assert (store.run_dir(state.run_id) / "next_run_plan.json").exists()


@pytest.mark.parametrize(
    "metadata,rank,top_percent",
    [
        ({"userRank": 20, "teamCount": 100}, 20, 20.0),
        ({"user_rank": 50, "team_count": 100}, 50, 50.0),
        ({"userRank": 0, "teamCount": 100}, None, None),
        ({"userRank": 5}, 5, None),
        ({"userRank": 101, "teamCount": 100}, 101, None),
        ({"userRank": True, "teamCount": 100}, None, None),
        ({}, None, None),
    ],
)
def test_account_standing_separate_from_run_score(fixture_data, metadata, rank, top_percent):
    _, c, config = fixture_data
    config.poll_attempts = 1
    state = RunState(
        run_id="standing-test",
        competition=c,
        config=config,
        submissions=[{"status": "COMPLETE", "reference": 12, "message": "this run"}],
        feedback={"local_score": 0.85},
    )
    client = Mock()
    client.submissions.return_value = [{"ref": 12, "publicScore": 0.8, "status": "COMPLETE"}]
    client.metadata.return_value = metadata
    LeaderboardAgent(Store(config.artifact_root)).review(state, client, refresh=True)
    report = state.feedback["leaderboard"]
    assert report["public_score"] == 0.8
    assert report["rank"] == rank
    assert report.get("top_percent") == top_percent
    assert report["top_20_percent"] == (top_percent <= 20 if top_percent is not None else None)
    assert report["standing_scope"] == "current_account_best"
    client.leaderboard.assert_not_called()


def test_standing_failure_preserves_submission_feedback(fixture_data):
    _, c, config = fixture_data
    config.poll_attempts = 1
    state = RunState(
        run_id="standing-error",
        competition=c,
        config=config,
        submissions=[{"status": "COMPLETE", "reference": 12, "message": "this run"}],
        feedback={"local_score": 0.85},
    )
    client = Mock()
    client.submissions.return_value = [{"ref": 12, "publicScore": 0.8, "status": "COMPLETE"}]
    client.metadata.side_effect = KaggleError("Kaggle denied access")
    LeaderboardAgent(Store(config.artifact_root)).review(state, client, refresh=True)
    report = state.feedback["leaderboard"]
    assert report["public_score"] == 0.8
    assert report["rank"] is None
    assert report["standing_note"] == "Kaggle denied access"
