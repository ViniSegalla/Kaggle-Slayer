import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from kaggle_slayer.orchestration.engine import Engine
from kaggle_slayer.storage.store import Store


def test_fixture_cannot_be_uploaded(fixture_data):
    _, competition, config = fixture_data
    config.submit = True
    store = Store(config.artifact_root)
    client = Mock()
    client.metadata.return_value = {}
    engine = Engine(store, client)
    state = engine.create(competition, config)
    state.phase, state.submission_path = (
        "submission_ready",
        str(Path(config.artifact_root) / "submission.csv"),
    )
    # A plan prevents training when recovering a ready-to-submit fixture run.
    from kaggle_slayer.schemas import Plan

    state.plan = Plan(strategy="fixture", queue=[], validation="auto", direction="maximize")
    with pytest.raises(ValueError, match="cannot be submitted"):
        engine.execute(state)
    assert not client.submit.called


def test_interrupted_submission_is_never_reuploaded(fixture_data):
    _, competition, config = fixture_data
    config.submit, config.poll_attempts = True, 0
    store = Store(config.artifact_root)
    client = Mock()
    client.metadata.return_value = {}
    engine = Engine(store, client)
    state = engine.create(competition, config)
    from kaggle_slayer.schemas import Plan

    state.plan = Plan(strategy="fixture", queue=[], validation="auto", direction="maximize")
    state.phase = "submitting"
    state.submissions = [
        {"status": "intent_recorded", "message": "previous upload", "reference": None}
    ]
    result = engine.execute(state)
    assert result.submissions[-1]["status"] == "outcome_unknown"
    assert not client.submit.called


@pytest.mark.parametrize("tamper", [False, True])
def test_ready_artifact_upload_integrity_and_single_attempt(fixture_data, tamper):
    directory, competition, config = fixture_data
    # This is a mocked network test with generated data, not a real Kaggle upload.
    (directory / "provenance.json").write_text(json.dumps({"kind": "mocked_api_test"}))
    config.submit, config.poll_attempts = True, 1
    store = Store(config.artifact_root)
    client = Mock()
    client.metadata.return_value = {}
    client.submit.return_value = {"ref": 123}
    client.submissions.return_value = [{"ref": 123, "status": "complete", "publicScore": 0.8}]
    engine = Engine(store, client)
    state = engine.create(competition, config)
    from kaggle_slayer.schemas import Plan

    state.plan = Plan(strategy="mock", queue=[], validation="auto", direction="maximize")
    path = store.run_dir(state.run_id) / "submission.csv"
    path.write_text("PassengerId,Survived\n1,0\n")
    state.metadata["submission_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    state.phase, state.submission_path = "submission_ready", str(path)
    state.feedback["local_score"] = 0.85
    if tamper:
        path.write_text("changed after validation")
        with pytest.raises(ValueError, match="integrity"):
            engine.execute(state)
        assert not client.submit.called
    else:
        result = engine.execute(state)
        assert result.submissions[0]["reference"] == 123
        assert result.feedback["leaderboard"]["public_score"] == 0.8
        assert store.load(state.run_id).submissions
        engine.execute(result)
        assert client.submit.call_count == 1
