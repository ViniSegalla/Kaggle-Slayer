"""Serializable contracts shared by agents; no private reasoning is recorded."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Competition(Schema):
    competition: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    target: str | None = None
    id_column: str | None = None
    task: Literal["classification", "regression"] | None = None
    metric: (
        Literal["accuracy", "f1", "roc_auc", "log_loss", "rmse", "rmsle", "log_rmse", "mae", "r2"]
        | None
    ) = None
    prediction_kind: Literal["label", "probability", "value"] | None = None
    target_transform: Literal["none", "log1p"] = "none"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    sample_file: str | None = None
    drop_columns: list[str] = Field(default_factory=list)
    feature_profile: Literal["generic", "titanic"] = "generic"
    validation: Literal["auto", "stratified", "kfold", "group", "time"] = "auto"
    group_column: str | None = None
    time_column: str | None = None

    @model_validator(mode="after")
    def relationships(self):
        if self.validation == "group" and not self.group_column:
            raise ValueError("group validation requires group_column")
        if self.validation == "time" and not self.time_column:
            raise ValueError("time validation requires time_column")
        for name in (self.train_file, self.test_file, self.sample_file):
            if name and (Path(name).is_absolute() or ".." in Path(name).parts):
                raise ValueError("Data files must be relative paths inside the dataset directory")
        return self


class RunConfig(Schema):
    seed: int = 42
    folds: int = Field(default=5, ge=2, le=20)
    max_experiments: int = Field(default=12, ge=3, le=100)
    max_iterations: int = Field(default=3, ge=1, le=20)
    max_seconds: int = Field(default=1800, ge=5, le=86400)
    experiment_timeout: int = Field(default=300, ge=2, le=14400)
    n_jobs: int = Field(default=2, ge=1, le=16)
    optuna_trials: int = Field(default=2, ge=1, le=50)
    max_submissions: int = Field(default=1, ge=0, le=1)
    poll_attempts: int = Field(default=3, ge=0, le=10)
    poll_interval: int = Field(default=10, ge=1, le=60)
    models: list[
        Literal["linear", "random_forest", "hist_gradient", "lightgbm", "xgboost", "catboost"]
    ] = Field(
        default_factory=lambda: ["linear", "random_forest", "lightgbm", "catboost", "xgboost"]
    )
    submit: bool = False
    data_dir: str | None = None
    artifact_root: str = "artifacts"
    max_rows: int = Field(default=1000000, ge=20)
    max_features: int = Field(default=10000, ge=10)
    max_dense_cells: int = Field(default=25000000, ge=1000)
    ensemble: bool = True
    tuning_families: int = Field(default=3, ge=1, le=6)
    confirmation_top_k: int = Field(default=2, ge=0, le=3)
    min_relative_improvement: float = Field(default=0.001, ge=0, le=0.1)

    @model_validator(mode="after")
    def unique_models(self):
        if len(set(self.models)) < 3:
            raise ValueError("Configure at least three distinct baseline model families")
        return self


class Candidate(Schema):
    id: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    parent_id: str | None = None
    trial_number: int | None = None
    validation_role: Literal["primary", "confirmation"] = "primary"
    validation_seed: int | None = None
    study_family: str | None = None


class Plan(Schema):
    strategy: str
    queue: list[Candidate]
    validation: str
    direction: Literal["maximize", "minimize"]
    source: str = "deterministic"


class Experiment(Schema):
    id: str
    model: str
    params: dict[str, Any]
    status: Literal["completed", "failed", "timeout"]
    score: float | None = None
    train_score: float | None = None
    fold_scores: list[float] = Field(default_factory=list)
    std: float | None = None
    duration: float = 0
    error: str | None = None
    checkpoint: str | None = None
    oof_path: str | None = None
    importance_path: str | None = None
    effective_params: dict[str, Any] = Field(default_factory=dict)
    validation_role: Literal["primary", "confirmation"] = "primary"
    validation_seed: int | None = None
    parent_id: str | None = None
    diagnostics_path: str | None = None
    training_contract: dict[str, Any] = Field(default_factory=dict)


class Decision(Schema):
    agent: str
    timestamp: str = Field(default_factory=now)
    input_context: dict[str, Any]
    selected_action: str
    rationale: str
    tool_invoked: str
    tool_result: dict[str, Any]
    artifacts: list[str] = Field(default_factory=list)
    next_action: str
    level: Literal["info", "warning", "error"] = "info"


class RunState(Schema):
    run_id: str
    competition: Competition
    config: RunConfig
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)
    status: Literal["running", "completed", "failed", "interrupted"] = "running"
    phase: str = "initialized"
    iteration: int = 0
    elapsed_seconds: float = 0
    plan: Plan | None = None
    report: dict[str, Any] = Field(default_factory=dict)
    feedback: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    experiments: list[Experiment] = Field(default_factory=list)
    best_id: str | None = None
    submission_path: str | None = None
    submissions: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
