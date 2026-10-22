from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from kaggle_slayer.cli import app
from kaggle_slayer.config import cli_run_profile, load_config


@pytest.mark.parametrize(
    "options,mode,experiments",
    [
        ([], "agentic", 8),
        (["--execution-mode", "deterministic"], "deterministic", 12),
        (["--config", "configs/default.yaml"], "deterministic", 12),
        (["--config", "configs/agentic.yaml", "--max-experiments", "2"], "agentic", 2),
    ],
)
def test_cli_default_agentic_and_explicit_overrides(monkeypatch, options, mode, experiments):
    engine = Mock()

    def execute(state):
        assert state.config.execution_mode == mode
        assert state.config.max_experiments == experiments
        return SimpleNamespace(
            status="completed",
            run_id="fake",
            swarm={},
            best_id=None,
            feedback={},
            submission_path=None,
            warnings=[],
        )

    engine.create.side_effect = lambda competition, config: SimpleNamespace(
        run_id="fake", config=config
    )
    engine.execute.side_effect = execute
    monkeypatch.setattr("kaggle_slayer.cli.Engine", Mock(return_value=engine))
    monkeypatch.setattr("kaggle_slayer.cli.Store", Mock())
    response = CliRunner().invoke(app, ["run", "--competition", "titanic", *options])
    assert response.exit_code == 0, response.output


def test_cli_profile_resolves_outside_repository_and_library_defaults_stay_compatible(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    profile = cli_run_profile()
    assert profile.is_file() and profile.name == "agentic.yaml"
    assert load_config("titanic", run_profile=profile)[1].execution_mode == "agentic"
    assert load_config("titanic")[1].execution_mode == "deterministic"
    assert cli_run_profile("deterministic").name == "default.yaml"
