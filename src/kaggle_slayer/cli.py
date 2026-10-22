"""All user-facing operations run through uv run (module or console script)."""

import json
from pathlib import Path
from typing import Annotated

import typer

from kaggle_slayer.config import Settings, load_config
from kaggle_slayer.orchestration.engine import Engine
from kaggle_slayer.storage.store import Store
from kaggle_slayer.tools.kaggle_client import KaggleClient, KaggleError, credentials_present

app = typer.Typer(no_args_is_help=True, help="KaggleSlayer — autonomous tabular competition team")


def emit(value):
    typer.echo(json.dumps(value, indent=2, default=str))


@app.command()
def doctor(competition: str = "titanic", require_kaggle: bool = False):
    """Check local environment and, when configured, read Kaggle competition metadata."""
    from importlib.metadata import version

    result = {
        "kaggle_credentials_present": credentials_present(),
        "llm_provider": Settings().llm_provider,
        "dependencies": {
            name: version(name)
            for name in (
                "pandas",
                "scikit-learn",
                "lightgbm",
                "catboost",
                "xgboost",
                "kaggle",
                "optuna",
                "streamlit",
            )
        },
    }
    if result["kaggle_credentials_present"]:
        try:
            result["competition_metadata"] = KaggleClient().metadata(competition)
            result["kaggle_status"] = "authenticated_read_succeeded"
        except KaggleError as exc:
            result["kaggle_status"] = str(exc)
    else:
        result["kaggle_status"] = (
            "blocked: configure Kaggle credentials and accept competition rules"
        )
    emit(result)
    if require_kaggle and result["kaggle_status"] != "authenticated_read_succeeded":
        raise typer.Exit(1)


@app.command("llm-check")
def llm_check():
    """Test one LLM planning request with synthetic context; may use provider credits."""
    from kaggle_slayer.llm.provider import PlannerLLM

    llm = PlannerLLM()
    llm.advise(
        {
            "task": "classification",
            "metric": "accuracy",
            "quality": {
                "train_rows": 100,
                "test_rows": 20,
                "dtypes": {"feature": "float64"},
                "missing_fraction": {"feature": 0},
                "warnings": [],
            },
            "allowed_models": ["linear", "random_forest", "lightgbm"],
        }
    )
    diagnostic = llm.last_diagnostic
    emit(diagnostic.model_dump(mode="json"))
    if diagnostic.status == "failed":
        raise typer.Exit(1)


@app.command()
def run(
    competition: Annotated[str, typer.Option(help="Kaggle competition slug")],
    profile: Path | None = None,
    config_file: Annotated[
        Path | None, typer.Option("--config", help="YAML run budgets and model roster")
    ] = None,
    data_dir: Annotated[
        Path | None, typer.Option(help="Local CSV directory; skips download")
    ] = None,
    max_experiments: int | None = None,
    max_iterations: int | None = None,
    max_seconds: int | None = None,
    folds: int | None = None,
    n_jobs: int | None = None,
    max_rows: int | None = None,
    artifact_root: Path | None = None,
    submit: Annotated[
        bool, typer.Option("--submit/--no-submit", help="Authorize one Kaggle upload")
    ] = False,
):
    """Analyze, plan, train, optimize, evaluate and generate a validated submission."""
    try:
        c, config = load_config(
            competition,
            profile,
            run_profile=config_file,
            data_dir=str(data_dir) if data_dir else None,
            max_experiments=max_experiments,
            max_iterations=max_iterations,
            max_seconds=max_seconds,
            folds=folds,
            n_jobs=n_jobs,
            max_rows=max_rows,
            artifact_root=str(artifact_root) if artifact_root else None,
            submit=submit,
        )
        store = Store(config.artifact_root)
        engine = Engine(store, progress=typer.echo)
        state = engine.create(c, config)
        typer.echo(f"Run ID: {state.run_id}")
        result = engine.execute(state)
        emit(
            {
                "run_id": result.run_id,
                "status": result.status,
                "best_model": result.best_id,
                "feedback": result.feedback,
                "submission": result.submission_path,
                "warnings": result.warnings,
            }
        )
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def resume(run_id: str, artifact_root: Path | None = None, submit: bool = False):
    """Recover a checkpoint; submission authorization must be renewed explicitly."""
    try:
        store = Store(artifact_root or Settings().artifact_root)
        state = store.load(run_id)
        state.config.submit = submit
        result = Engine(store, progress=typer.echo).execute(state)
        emit(
            {"run_id": result.run_id, "status": result.status, "submission": result.submission_path}
        )
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def runs(artifact_root: Path | None = None):
    """List persisted runs."""
    rows = Store(artifact_root or Settings().artifact_root).rows("runs")
    emit([{key: value for key, value in row.items() if key != "state"} for row in rows])


@app.command("inspect")
def inspect_run(run_id: str, artifact_root: Path | None = None):
    """Print one run's complete configuration, decisions and experiment references."""
    emit(Store(artifact_root or Settings().artifact_root).load(run_id).model_dump())


@app.command("submit")
def submit_saved(run_id: str, artifact_root: Path | None = None):
    """Explicitly authorize one upload of a completed run's validated submission."""
    try:
        store = Store(artifact_root or Settings().artifact_root)
        state = store.load(run_id)
        if state.status != "completed" or not state.submission_path:
            raise ValueError("Complete or resume the run before submitting")
        if state.submissions:
            raise ValueError(
                "This run already has a submission attempt; use refresh and inspect Kaggle history"
            )
        state.status, state.phase, state.config.submit = "running", "submission_ready", True
        result = Engine(store, progress=typer.echo).execute(state)
        emit(
            {
                "run_id": result.run_id,
                "submissions": result.submissions,
                "feedback": result.feedback,
            }
        )
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def refresh(run_id: str, artifact_root: Path | None = None):
    """Read available Kaggle feedback for a prior upload; never submit."""
    from kaggle_slayer.agents.leaderboard_agent import LeaderboardAgent
    from kaggle_slayer.agents.planner import CompetitionPlanner

    try:
        store = Store(artifact_root or Settings().artifact_root)
        with store.lock(run_id):
            state = store.load(run_id)
            from kaggle_slayer.orchestration.progress import RunProgress

            live = RunProgress(store, typer.echo)

            def progress(agent, message):
                live.emit(run_id, agent, message)

            LeaderboardAgent(store, progress=progress).review(state, KaggleClient(), refresh=True)
            CompetitionPlanner(store, progress=progress).leaderboard_followup(state)
            store.save(state)
            emit(state.feedback)
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
