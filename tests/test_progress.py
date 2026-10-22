import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kaggle_slayer.cli import app
from kaggle_slayer.orchestration.progress import WorkerEvents, decision_summary
from kaggle_slayer.schemas.models import Candidate
from kaggle_slayer.tools.training_tools import monitor_worker, run_isolated


def test_worker_reader_handles_partial_invalid_and_untrusted_fields(tmp_path):
    path = tmp_path / "progress.jsonl"
    events = WorkerEvents(path, 3, "mae", "baseline-linear")
    path.write_text('{"stage":"fit_fold","fold":1}', encoding="utf-8")
    assert events.read() == []
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\nnot-json\n[]\n")
        stream.write(
            json.dumps(
                {"stage": "fold_complete", "fold": 1, "score": 12.5, "raw_response": "secret-token"}
            )
            + "\n"
        )
        stream.write(json.dumps({"stage": "unknown", "raw_response": "secret-token"}) + "\n")
    messages = events.read()
    assert len(messages) == 2
    assert "fold 1/3" in messages[0]
    assert "mae=12.5" in messages[1]
    assert "secret-token" not in str(messages)
    assert events.read() == []


def test_decision_summary_does_not_render_provider_or_error_bodies():
    assert "secret" not in decision_summary(
        "llm_planning_failed", {"status": "failed", "code": "authentication", "message": "secret"}
    )
    assert "secret" not in decision_summary(
        "experiment_failed",
        {"id": "baseline-linear", "status": "failed", "duration": 2, "error": "secret"},
    )


def test_real_worker_streams_folds_refit_and_checkpoints(fixture_data, tmp_path):
    _, competition, config = fixture_data
    messages = []
    result = run_isolated(
        Candidate(id="live-linear", model="linear", rationale="test"),
        competition,
        config,
        tmp_path / "live",
        30,
        progress=messages.append,
    )
    assert result.status == "completed"
    assert len([m for m in messages if "validation complete" in m]) == config.folds
    assert any("refitting on all training rows" in m for m in messages)
    assert not any("selecting stopping iterations" in m for m in messages)
    assert "checkpoint and diagnostics saved" in messages[-1]
    assert (tmp_path / "live/worker.log").exists()


def test_heartbeat_and_timeout_kill_worker(fixture_data, tmp_path, monkeypatch):
    _, competition, config = fixture_data
    clock = [0.0]
    monkeypatch.setattr("kaggle_slayer.tools.training_tools.time.monotonic", lambda: clock[0])

    class Process:
        killed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self, timeout=None):
            if self.killed:
                return -1
            clock[0] += 5
            raise subprocess.TimeoutExpired("worker", timeout)

        def kill(self):
            self.killed = True

    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    messages = []
    with (tmp_path / "worker.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            monitor_worker(
                ["worker"],
                log,
                {},
                16,
                tmp_path,
                Candidate(id="slow", model="linear", rationale="test"),
                competition,
                config,
                messages.append,
            )
    assert process.killed
    assert any("still running" in m and "elapsed=10s" in m for m in messages)


def test_cli_shows_agent_experiment_and_fold_progress(fixture_data, tmp_path):
    directory, competition, config = fixture_data
    profile = tmp_path / "competition.yaml"
    profile.write_text(json.dumps(competition.model_dump()), encoding="utf-8")
    run_profile = tmp_path / "run.yaml"
    run_profile.write_text(json.dumps(config.model_dump()), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--competition",
            competition.competition,
            "--profile",
            str(profile),
            "--config",
            str(run_profile),
            "--data-dir",
            str(directory),
        ],
    )
    assert result.exit_code == 0, result.output
    for agent in (
        "competition_planner",
        "data_engineer",
        "modeler",
        "evaluator",
        "leaderboard_reviewer",
    ):
        assert f"[{agent}]" in result.output
    assert "Experiment 1/4" in result.output
    assert "fold 1/3" in result.output
    assert "Queue:" in result.output
    assert "Best:" in result.output
    assert "No upload authorized" in result.output
    logs = list(Path(config.artifact_root).glob("runs/*/progress.log"))
    assert len(logs) == 1
    assert "validation complete" in logs[0].read_text(encoding="utf-8")
