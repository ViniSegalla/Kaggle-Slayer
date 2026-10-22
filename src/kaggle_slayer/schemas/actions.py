"""Swarm wire protocol v1. JSON only; descriptions are never executable code."""

from typing import Annotated, Literal

from pydantic import Field

from kaggle_slayer.schemas.models import Schema

VERSION = "swarm-v1"
Role = Literal["coordinator", "data_validation", "pipeline_modeling", "evaluation", "submission"]
Id = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]


class PipelineConfig(Schema):
    numeric_imputation: Literal["median", "mean", "most_frequent"] = "median"
    scaling: Literal["auto", "standard", "none"] = "auto"
    missing_indicators: bool = True
    max_categories: int = Field(default=64, ge=2, le=64)
    min_frequency: int = Field(default=2, ge=1, le=100)
    features: list[Literal["missing_count", "numeric_square", "numeric_signed_log"]] = Field(
        default_factory=list, max_length=3
    )


class EmptyArgs(Schema):
    pass


class ValidationArgs(Schema):
    method: Literal["stratified", "kfold", "group", "time"]


class ConfigureArgs(Schema):
    model: Literal["linear", "random_forest", "hist_gradient", "lightgbm", "xgboost", "catboost"]
    params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


class ExperimentArgs(Schema):
    pipeline_ref: Id


class TuneArgs(Schema):
    parent_ref: Id
    params: dict[str, float | int | str | bool | None]


class EnsembleArgs(Schema):
    members: list[Id] = Field(min_length=2, max_length=2)


class ConfirmationArgs(Schema):
    parent_ref: Id


class ToolAction(Schema):
    kind: Literal["tool"]
    action_id: Id
    recipient: Literal["runtime"] = "runtime"
    tool: str = Field(max_length=64)
    # Validated against the named capability's typed schema before scheduling.
    arguments: dict = Field(default_factory=dict)
    depends_on: list[Id] = Field(default_factory=list, max_length=16)
    evidence: list[Id] = Field(default_factory=list, max_length=16)
    summary: str = Field(max_length=400)


class DelegateAction(Schema):
    kind: Literal["delegate"]
    action_id: Id
    recipient: Role
    task: str = Field(min_length=1, max_length=600)
    depends_on: list[Id] = Field(default_factory=list, max_length=16)
    evidence: list[Id] = Field(default_factory=list, max_length=16)
    summary: str = Field(max_length=400)


class WaitAction(Schema):
    kind: Literal["wait"]
    action_id: Id
    recipient: Literal["runtime"] = "runtime"
    depends_on: list[Id] = Field(min_length=1, max_length=16)
    evidence: list[Id] = Field(default_factory=list, max_length=16)
    summary: str = Field(max_length=400)


class FinishAction(Schema):
    kind: Literal["finish"]
    action_id: Id
    recipient: Literal["runtime"] = "runtime"
    depends_on: list[Id] = Field(default_factory=list, max_length=16)
    evidence: list[Id] = Field(default_factory=list, max_length=16)
    summary: str = Field(max_length=400)
    outcome: Literal["complete", "incomplete"] = "complete"


Action = Annotated[
    ToolAction | DelegateAction | WaitAction | FinishAction, Field(discriminator="kind")
]


class AgentResponse(Schema):
    version: Literal["swarm-v1"]
    actions: list[Action] = Field(min_length=1, max_length=8)


class Task(Schema):
    id: Id
    role: Role
    parent: Id | None = None
    depth: int = 0
    goal: str
    depends_on: list[Id] = Field(default_factory=list)
    evidence: list[Id] = Field(default_factory=list)
    status: Literal["ready", "waiting", "completed", "failed"] = "ready"
    turns: int = 0
    repair_pending: bool = False
    repair_attempts: int = 0
    repair_issues: list[str] = Field(default_factory=list)
    action_aliases: dict[str, str] = Field(default_factory=dict)


class ActionRecord(Schema):
    task_id: Id
    action: Action
    status: Literal["pending", "reserved", "completed", "failed"] = "pending"
    result: dict = Field(default_factory=dict)
    reservation: dict = Field(default_factory=dict)


class SwarmState(Schema):
    version: Literal["swarm-v1"] = VERSION
    session_id: str
    tasks: dict[str, Task] = Field(default_factory=dict)
    actions: dict[str, ActionRecord] = Field(default_factory=dict)
    pipelines: dict[str, dict] = Field(default_factory=dict)
    observations: list[dict] = Field(default_factory=list)
    artifacts: dict[str, dict] = Field(default_factory=dict)
    requests: int = 0
    tokens: int = 0
    turns: int = 0
    invalid_actions: int = 0
    schema_repairs: int = 0
    counter_version: int = 1
    experiment_attempts: int = 0
    tuning_attempts: int = 0
    signatures: dict[str, int] = Field(default_factory=dict)
    validation: dict | None = None
    confirmation_started: bool = False
    finalized: bool = False
    termination_reason: str | None = None
    usage: list[dict] = Field(default_factory=list)
