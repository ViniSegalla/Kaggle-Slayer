"""Bounded capability registry and dispatcher. Called only by the session owner."""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kaggle_slayer.schemas.actions import (
    ConfigureArgs,
    ConfirmationArgs,
    EmptyArgs,
    EnsembleArgs,
    ExperimentArgs,
    TuneArgs,
    ValidationArgs,
)
from kaggle_slayer.schemas.models import Candidate
from kaggle_slayer.tools.data_tools import fingerprint, inspect_dataset
from kaggle_slayer.tools.evaluation_tools import direction, validate_submission


class ToolError(ValueError):
    """Controlled code only; never echo untrusted parameters or local exceptions."""


@dataclass(frozen=True)
class Capability:
    arguments: type
    roles: tuple[str, ...]
    description: str
    experiment: bool = False


CAPABILITIES = {
    "inspect_dataset": Capability(
        EmptyArgs, ("data_validation",), "Bounded aggregate dataset report"
    ),
    "configure_validation": Capability(
        ValidationArgs, ("data_validation",), "Propose/freeze CV lineage"
    ),
    "configure_pipeline": Capability(
        ConfigureArgs, ("pipeline_modeling",), "Create supported fold-fit pipeline"
    ),
    "run_experiment": Capability(
        ExperimentArgs, ("pipeline_modeling",), "Train CV/OOF and full-data refit", True
    ),
    "tune_experiment": Capability(
        TuneArgs, ("pipeline_modeling",), "One bounded explicit-parameter trial", True
    ),
    "ensemble": Capability(
        EnsembleArgs, ("pipeline_modeling",), "Measure equal-weight two-member ensemble", True
    ),
    "confirm": Capability(
        ConfirmationArgs,
        ("pipeline_modeling", "evaluation"),
        "Terminal diagnostic second-seed CV",
        True,
    ),
    "evaluate": Capability(
        EmptyArgs, ("evaluation",), "Trusted primary eligibility, ranking and OOF diagnostics"
    ),
    "generate_predictions": Capability(
        EmptyArgs, ("submission",), "Predict using trusted measured winner"
    ),
    "validate_predictions": Capability(
        EmptyArgs, ("submission",), "Verify serialized predictions and artifact hash"
    ),
    "retrieve_feedback": Capability(
        EmptyArgs, ("submission",), "Read feedback after finalization; never upload"
    ),
}

# Inclusive bounds or enumerations. Resource controls/objectives cannot be overridden.
BOUNDS = {
    "linear": {
        "C": (0.0001, 1000.0),
        "alpha": (0.0001, 1000.0),
        "class_weight": (None, "balanced"),
    },
    "random_forest": {
        "n_estimators": (10, 500),
        "max_depth": (2, 32),
        "min_samples_leaf": (1, 32),
        "max_features": (0.1, 1.0),
    },
    "hist_gradient": {
        "max_iter": (10, 500),
        "max_leaf_nodes": (2, 128),
        "learning_rate": (0.001, 0.5),
        "l2_regularization": (0.0, 100.0),
        "min_samples_leaf": (2, 100),
    },
    "lightgbm": {
        "n_estimators": (10, 500),
        "num_leaves": (2, 128),
        "learning_rate": (0.001, 0.5),
        "max_depth": (2, 16),
        "reg_alpha": (0.0, 100.0),
        "reg_lambda": (0.0, 100.0),
        "min_child_samples": (2, 100),
    },
    "xgboost": {
        "n_estimators": (10, 500),
        "max_depth": (2, 12),
        "learning_rate": (0.001, 0.5),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (0.0, 100.0),
        "reg_lambda": (0.0, 100.0),
    },
    "catboost": {
        "iterations": (10, 500),
        "depth": (2, 10),
        "learning_rate": (0.001, 0.5),
        "l2_leaf_reg": (0.0, 100.0),
    },
}
INTEGER_PARAMS = {
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_iter",
    "max_leaf_nodes",
    "num_leaves",
    "min_child_samples",
    "iterations",
    "depth",
}


def parameter_bounds(model, task):
    values = BOUNDS[model].copy()
    if model == "linear":
        for key in {"alpha"} if task == "classification" else {"C", "class_weight"}:
            values.pop(key)
    return values


def validate_parameters(model, task, params):
    allowed = parameter_bounds(model, task)
    for key, value in params.items():
        if key not in allowed:
            raise ToolError("unsupported_parameter")
        bounds = allowed[key]
        if key == "class_weight":
            if value not in bounds:
                raise ToolError("parameter_bounds")
        elif (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or not bounds[0] <= value <= bounds[1]
        ):
            raise ToolError("parameter_bounds")
        elif key in INTEGER_PARAMS and type(value) is not int:
            raise ToolError("parameter_type")


def registry(role, state):
    return {
        name: {"arguments": cap.arguments.model_json_schema(), "description": cap.description}
        for name, cap in CAPABILITIES.items()
        if role in cap.roles
    } | (
        {
            "model_parameter_bounds": {
                m: parameter_bounds(m, state.competition.task) for m in state.config.models
            }
        }
        if role == "pipeline_modeling"
        else {}
    )


def dataset_observation(state):
    r = state.report
    distribution = r.get("target_distribution", {})
    target = (
        {"class_counts": sorted(distribution.values(), reverse=True)[:100]}
        if state.competition.task == "classification"
        else {
            key: value
            for key, value in distribution.items()
            if key in {"count", "mean", "std", "min", "50%", "max"}
        }
    )
    return {
        "train_rows": r.get("train_rows"),
        "test_rows": r.get("test_rows"),
        "feature_count": len(r.get("features", [])),
        "target_summary": target,
        "duplicate_rows": r.get("duplicate_rows"),
        "fingerprint": hashlib.sha256(
            json.dumps(r.get("fingerprints", {}), sort_keys=True).encode()
        ).hexdigest(),
        "features": [
            {
                "ref": f"feature-{i}",
                "kind": "numeric" if col in r.get("numeric_features", []) else "categorical",
                "missing_fraction": r.get("missing_fraction", {}).get(col),
                "cardinality": r.get("categorical_cardinality", {}).get(col),
            }
            for i, col in enumerate(r.get("features", [])[:32])
        ],
        "features_truncated": len(r.get("features", [])) > 32,
        "structural_validation": {
            "group_available": bool(state.competition.group_column),
            "time_available": bool(state.competition.time_column),
        },
        "warning_count": len(r.get("warnings", [])),
    }


class Dispatcher:
    def __init__(self, runtime):
        self.runtime = runtime

    @property
    def state(self):
        return self.runtime.state

    @property
    def session(self):
        return self.runtime.session

    def validate(self, role, action):
        cap = CAPABILITIES.get(action.tool)
        if cap is None:
            raise ToolError("unknown_tool")
        if role not in cap.roles:
            raise ToolError("role_permission")
        return cap.arguments.model_validate(action.arguments)

    def reuse_key(self, action, args):
        """Identify repeatable local work by measured inputs, never by task prose."""
        if action.tool not in {"evaluate", "generate_predictions", "validate_predictions"}:
            return None
        s, session = self.state, self.session
        if not s.report or not session.validation:
            return None
        if any(
            record.status in {"pending", "reserved"}
            and record.action.kind == "tool"
            and CAPABILITIES[record.action.tool].experiment
            for record in session.actions.values()
        ):
            return None
        paths = [
            Path(s.config.data_dir) / s.competition.train_file,
            Path(s.config.data_dir) / s.competition.test_file,
        ]
        if s.competition.sample_file:
            paths.append(Path(s.config.data_dir) / s.competition.sample_file)
        if fingerprint(paths) != s.report["fingerprints"]:
            raise ToolError("dataset_fingerprint_changed")
        experiments = []
        for experiment in s.experiments:
            if experiment.lineage != session.validation["lineage"]:
                continue
            if experiment.status == "completed":
                if (
                    experiment.training_contract.get("dataset_fingerprints")
                    != s.report["fingerprints"]
                ):
                    raise ToolError("dataset_fingerprint_changed")
                self.runtime.verify_artifact(experiment.id)
            experiments.append(
                {
                    "experiment": experiment.model_dump(
                        exclude={
                            "checkpoint",
                            "oof_path",
                            "diagnostics_path",
                            "importance_path",
                            "error",
                        }
                    ),
                    "hashes": [
                        f["sha256"]
                        for f in session.artifacts.get(experiment.id, {}).get("files", [])
                    ],
                }
            )
        snapshot = {
            "tool": action.tool,
            "arguments": args.model_dump(),
            "dataset": s.report["fingerprints"],
            "requirements": s.competition.model_dump(),
            "validation": session.validation,
            "experiments": experiments,
            "confirmation_started": session.confirmation_started,
        }
        if action.tool != "evaluate":
            if not s.best_id:
                return None
            self.experiment(s.best_id)
            snapshot["best_ref"] = s.best_id
        if action.tool == "validate_predictions":
            if not session.finalized or not s.submission_path:
                return None
            current = hashlib.sha256(Path(s.submission_path).read_bytes()).hexdigest()
            if current != s.metadata.get("submission_sha256"):
                raise ToolError("prediction_integrity")
            snapshot["submission_sha256"] = current
        return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()

    def reuse(self, action, key):
        """Return verified evidence for the same snapshot without repeating effects."""
        if key is None:
            return None
        args = CAPABILITIES[action.tool].arguments.model_validate(action.arguments)
        if self.reuse_key(action, args) != key:
            return None
        s, session = self.state, self.session
        for ref, record in session.actions.items():
            if (
                record.status != "completed"
                or record.action.kind != "tool"
                or record.action.tool != action.tool
                or record.tool_snapshot != key
            ):
                continue
            result = record.result
            if action.tool == "evaluate":
                if result.get("best_ref") != s.best_id:
                    continue
            elif action.tool == "generate_predictions":
                if (
                    not session.finalized
                    or not s.submission_path
                    or result.get("best_ref") != s.best_id
                    or result.get("sha256") != s.metadata.get("submission_sha256")
                ):
                    continue
                self.runtime.verify_artifact(result["prediction_ref"])
                if (
                    hashlib.sha256(Path(s.submission_path).read_bytes()).hexdigest()
                    != result["sha256"]
                ):
                    raise ToolError("prediction_integrity")
            elif (
                not result.get("valid")
                or result.get("sha256") != s.metadata.get("swarm_submission_validated")
                or result.get("sha256") != s.metadata.get("submission_sha256")
            ):
                continue
            return ref, deepcopy(result) | {"reused_from": ref}
        return None

    def candidate(self, action, args):
        s, session = self.state, self.session
        if not session.validation:
            raise ToolError("validation_required")
        if session.finalized:
            raise ToolError("selection_sealed")
        lineage = session.validation["lineage"]
        candidate = None
        if action.tool == "run_experiment":
            contract = session.pipelines.get(args.pipeline_ref)
            if not contract:
                raise ToolError("unknown_pipeline_ref")
            candidate = Candidate(id=action.action_id, rationale=action.summary, **contract)
        elif action.tool in {"tune_experiment", "confirm"}:
            parent = self.experiment(args.parent_ref)
            if parent.model == "ensemble" or parent.validation_role != "primary":
                raise ToolError("ineligible_parent")
            candidate = Candidate(
                id=action.action_id,
                model=parent.model,
                params=parent.params,
                pipeline=parent.training_contract.get("pipeline_config", {}),
                parent_id=parent.id,
                rationale=action.summary,
            )
            if action.tool == "tune_experiment":
                if session.tuning_attempts >= s.config.max_iterations * s.config.optuna_trials:
                    raise ToolError("tuning_budget")
                validate_parameters(parent.model, s.competition.task, args.params)
                candidate.params = parent.params | args.params
            else:
                if not s.best_id or s.competition.validation in {"group", "time"}:
                    raise ToolError("confirmation_not_available")
                count = sum(
                    r.reservation.get("candidate", {}).get("validation_role") == "confirmation"
                    for r in session.actions.values()
                )
                if count >= s.config.confirmation_top_k:
                    raise ToolError("confirmation_budget")
                session.confirmation_started = True
                candidate.validation_role = "confirmation"
                candidate.validation_seed = s.config.seed + 1009
        elif action.tool == "ensemble":
            if not s.config.ensemble or len(set(args.members)) != 2:
                raise ToolError("ensemble_not_available")
            members = [self.experiment(ref) for ref in args.members]
            if any(m.model == "ensemble" or m.validation_role != "primary" for m in members):
                raise ToolError("ineligible_member")
            candidate = Candidate(
                id=action.action_id,
                model="ensemble",
                rationale=action.summary,
                params={
                    "members": [
                        {
                            "model": m.model,
                            "params": m.params,
                            "pipeline": m.training_contract.get("pipeline_config", {}),
                        }
                        for m in members
                    ]
                },
            )
        if session.confirmation_started and candidate.validation_role == "primary":
            raise ToolError("confirmation_is_terminal")
        candidate.lineage = lineage
        return candidate

    def experiment(self, ref):
        found = next((e for e in self.state.experiments if e.id == ref), None)
        if (
            not found
            or found.status != "completed"
            or found.lineage != self.session.validation["lineage"]
        ):
            raise ToolError("invalid_experiment_ref")
        self.runtime.verify_artifact(ref)
        return found

    def execute(self, action, args):
        s, session, engine = self.state, self.session, self.runtime.engine
        name = action.tool
        if name == "inspect_dataset":
            if s.config.data_dir and not s.competition.metric:
                raise ToolError("metric_requirement_unresolved")
            from kaggle_slayer.schemas import Competition, RunConfig
            from kaggle_slayer.tools.runtime_worker import run_bounded

            result = run_bounded(
                "prepare",
                s,
                self.runtime.directory / "operations" / action.action_id,
                self.runtime.remaining(),
                self.runtime.cancelled,
            )
            s.competition = Competition.model_validate(result["competition"])
            s.config = RunConfig.model_validate(result["config"])
            s.metadata, s.report, s.warnings = (
                result["metadata"],
                result["report"],
                result["warnings"],
            )
            s.metadata.setdefault("resolved_requirements", s.competition.model_dump())
            self.runtime.data = inspect_dataset(Path(s.config.data_dir), s.competition, s.config)
            return dataset_observation(s)
        if not s.report:
            raise ToolError("dataset_inspection_required")
        if name == "configure_validation":
            if session.finalized or session.confirmation_started:
                raise ToolError("selection_sealed")
            original = s.metadata["user_constraints"]["competition"]
            if original["validation"] != "auto" and args.method != original["validation"]:
                raise ToolError("immutable_validation_constraint")
            if (
                s.competition.group_column
                and args.method != "group"
                or s.competition.time_column
                and args.method != "time"
            ):
                raise ToolError("structural_separation_required")
            updated = s.competition.model_copy(update={"validation": args.method})
            data = inspect_dataset(Path(s.config.data_dir), updated, s.config)
            lineage = hashlib.sha256(
                json.dumps(
                    {
                        "fingerprints": data.report["fingerprints"],
                        "method": args.method,
                        "competition": updated.model_dump(),
                        "folds": s.config.folds,
                        "seed": s.config.seed,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:20]
            s.competition, self.runtime.data = updated, data
            s.report = data.report
            session.validation = {
                "method": args.method,
                "lineage": lineage,
                "folds": s.config.folds,
                "seed": s.config.seed,
            }
            s.best_id, s.submission_path = None, None
            return session.validation
        if name == "configure_pipeline":
            if session.finalized or session.confirmation_started:
                raise ToolError("selection_sealed")
            if args.model not in s.config.models:
                raise ToolError("model_not_enabled")
            if args.model == "catboost" and (
                args.pipeline.scaling != "auto" or args.pipeline.numeric_imputation != "median"
            ):
                raise ToolError("native_categories_incompatible_options")
            validate_parameters(args.model, s.competition.task, args.params)
            if len(set(args.pipeline.features)) != len(args.pipeline.features):
                raise ToolError("duplicate_feature_operation")
            session.pipelines[action.action_id] = args.model_dump()
            return {"pipeline_ref": action.action_id, "contract": args.model_dump()}
        if name == "evaluate":
            if not session.validation:
                raise ToolError("validation_required")
            for experiment in s.experiments:
                if (
                    experiment.status == "completed"
                    and experiment.lineage == session.validation["lineage"]
                ):
                    self.runtime.verify_artifact(experiment.id)
            # Evaluator reads the session snapshot for the active comparison lineage.
            s.swarm = session.model_dump(mode="json")
            best = engine.evaluator.evaluate(s)
            if session.confirmation_started:
                engine.evaluator.review_stability(s)
            ranked = {e.id: e for e in s.experiments}
            analysis = s.feedback.get("error_analysis", {})
            features = {name: f"feature-{i}" for i, name in enumerate(s.report["features"])}
            return {
                "best_ref": best.id,
                "metric": s.competition.metric,
                "direction": direction(s.competition.metric),
                "ranking": [
                    {
                        "ref": e.id,
                        "score": e.score,
                        "std": e.std,
                        "train_score": e.train_score,
                        "fold_scores": e.fold_scores,
                    }
                    for e in [ranked[ref] for ref in s.feedback["eligible_ids"]]
                ],
                "gap": s.feedback["train_validation_gap"],
                "diagnostics": {
                    "overall_error": analysis.get("overall_error"),
                    "measure": analysis.get("measure"),
                    "slice_count": len(analysis.get("slices", [])),
                    "slices": [
                        {
                            "ref": f"slice-{i}",
                            "feature_ref": features.get(row["feature"]),
                            "rows": row["rows"],
                            "error": row["error"],
                        }
                        for i, row in enumerate(analysis.get("slices", [])[:8])
                    ],
                },
            }
        if name == "generate_predictions":
            if any(
                r.status in {"pending", "reserved"}
                and r.action.kind == "tool"
                and CAPABILITIES[r.action.tool].experiment
                for r in session.actions.values()
            ):
                raise ToolError("pending_experiments")
            if not s.best_id:
                raise ToolError("evaluation_required")
            self.experiment(s.best_id)
            from kaggle_slayer.tools.prediction_worker import run_predictions

            result = run_predictions(
                s,
                self.runtime.directory,
                min(self.runtime.remaining(), s.config.experiment_timeout),
                self.runtime.cancelled,
            )
            s.submission_path = result["path"]
            s.metadata["submission_sha256"] = result["sha256"]
            session.finalized = True
            self.runtime.register_artifact(action.action_id, [s.submission_path])
            return {
                "prediction_ref": action.action_id,
                "best_ref": s.best_id,
                "sha256": s.metadata["submission_sha256"],
                "rows": s.report["test_rows"],
            }
        if name == "validate_predictions":
            if not session.finalized or not s.submission_path:
                raise ToolError("predictions_required")
            path = Path(s.submission_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != s.metadata["submission_sha256"]:
                raise ToolError("prediction_integrity")
            c, data = s.competition, self.runtime.data
            types = {c.id_column: str} if c.id_column else {}
            if c.prediction_kind == "label" and isinstance(data.report["classes"][0], str):
                types.update({col: str for col in pd.read_csv(path, nrows=0) if col != c.id_column})
            result = validate_submission(
                pd.read_csv(path, dtype=types), data.test, data.sample, c, data.report["classes"]
            )
            s.metadata["swarm_submission_validated"] = s.metadata["submission_sha256"]
            return {
                "valid": result["valid"],
                "rows": result["rows"],
                "sha256": s.metadata["submission_sha256"],
            }
        if name == "retrieve_feedback":
            if not session.finalized:
                raise ToolError("feedback_selection_barrier")
            from kaggle_slayer.tools.runtime_worker import run_bounded

            if s.submissions:
                result = run_bounded(
                    "feedback",
                    s,
                    self.runtime.directory / "operations" / action.action_id,
                    self.runtime.remaining(),
                    self.runtime.cancelled,
                )
                s.feedback, s.submissions, s.warnings = (
                    result["feedback"],
                    result["submissions"],
                    result["warnings"],
                )
            else:
                s.feedback["leaderboard"] = {
                    "status": "not_submitted",
                    "public_score": None,
                    "rank": None,
                    "top_percent": None,
                    "top_20_percent": None,
                }
            feedback = s.feedback["leaderboard"]
            return {
                k: feedback.get(k)
                for k in ("public_score", "rank", "top_percent", "top_20_percent")
            }
        raise ToolError("unknown_tool")
