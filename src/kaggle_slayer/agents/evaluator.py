import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px

from kaggle_slayer.agents.base import Agent
from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.evaluation_tools import direction, rank_experiments, validate_submission
from kaggle_slayer.tools.training_tools import load_predictions


class EvaluationAgent(Agent):
    name = "evaluator"

    def evaluate(self, state):
        self.activity(state, "Comparing primary CV results and auditing train/validation gaps")
        metric = state.competition.metric
        ranking = rank_experiments(state.experiments, metric)
        if not ranking:
            raise ValueError("All experiments failed; inspect worker logs and dataset report")
        singles = [e for e in ranking if e.model != "ensemble"]
        rejected_ensembles = []
        if singles:
            sign = 1 if direction(metric) == "maximize" else -1
            threshold = max(abs(singles[0].score), 1e-9) * state.config.min_relative_improvement
            rejected_ensembles = [
                e.id
                for e in ranking
                if e.model == "ensemble" and sign * (e.score - singles[0].score) <= threshold
            ]
            ranking = [e for e in ranking if e.id not in rejected_ensembles]
        state.best_id = ranking[0].id
        best = ranking[0]
        sign = 1 if direction(metric) == "maximize" else -1
        gap = sign * (best.train_score - best.score)
        warnings = []
        if metric in {"accuracy", "roc_auc", "f1", "r2"} and best.score > 0.995:
            warnings.append(
                "Near-perfect CV score; audit identifiers, duplicates and target leakage before submitting"
            )
        if gap > max(abs(best.score) * 0.15, 0.05):
            warnings.append("Large train/CV gap; favor regularization or revise validation")
        state.warnings = list(dict.fromkeys(state.warnings + warnings))
        rejected = [
            e.id for e in ranking[1:] if sign * (best.score - e.score) > max(best.std or 0, 0.001)
        ]
        state.feedback = {
            "best_id": best.id,
            "local_score": best.score,
            "metric": metric,
            "train_validation_gap": gap,
            "rejected_for_final": rejected,
            "recommendation": "Tune regularization and model complexity using the same fixed folds; keep the baseline for comparison.",
            "warnings": warnings,
            "ensemble_below_improvement_threshold": rejected_ensembles,
        }
        if best.diagnostics_path:
            analysis = json.loads(Path(best.diagnostics_path).read_text(encoding="utf-8"))
            state.feedback["error_analysis"] = analysis
            if analysis["slices"]:
                worst = analysis["slices"][0]
                state.feedback["recommendation"] = (
                    f"Audit {worst['feature']} slice {worst['slice']} ({worst['rows']} held-out rows): "
                    f"{analysis['measure']}={worst['error']:.5g} versus overall "
                    f"{analysis['overall_error']:.5g}. Compare model families and a fixed blend; "
                    "slice differences are exploratory, not proof of a defect."
                )
        output = self.store.run_dir(state.run_id)
        write_json(
            output / "evaluation.json",
            state.feedback | {"ranking": [e.model_dump() for e in ranking]},
        )
        chart = pd.DataFrame(
            [
                {"experiment": e.id, "score": e.score, "std": e.std, "model": e.model}
                for e in ranking
            ]
        )
        px.bar(
            chart,
            x="experiment",
            y="score",
            color="model",
            error_y="std",
            template="plotly_dark",
            title=f"{metric}: {direction(metric)} (fixed CV folds)",
        ).write_html(output / "comparison.html", include_plotlyjs=True)
        self.log(
            state,
            "select_best_validated_model",
            "Rank completed experiments by metric direction; weak candidates are excluded from final selection.",
            "rank_experiments",
            state.feedback,
            "planner.feedback_or_submission",
            [str(output / "evaluation.json"), str(output / "comparison.html")],
        )
        return best

    def review_stability(self, state):
        confirmations = [e for e in state.experiments if e.validation_role == "confirmation"]
        valid = [e for e in confirmations if e.status == "completed"]
        sign = 1 if direction(state.competition.metric) == "maximize" else -1
        primary = {e.id: e for e in state.experiments if e.validation_role == "primary"}
        report = {
            "status": "completed" if len(valid) >= 2 else "unavailable",
            "selection_changed": False,
            "reason": "A second split seed is diagnostic; it is not an untouched holdout and does not trigger further tuning.",
            "results": [
                {
                    "parent_id": e.parent_id,
                    "status": e.status,
                    "score": e.score,
                    "seed": e.validation_seed,
                    "primary_score": primary[e.parent_id].score,
                }
                for e in confirmations
                if e.parent_id in primary
            ],
        }
        if len(valid) >= 2:
            winner = max(valid, key=lambda e: sign * e.score)
            report["same_winner"] = winner.parent_id == state.best_id
            if not report["same_winner"]:
                warning = "Finalist ordering changed on the confirmation split; performance gains are split-sensitive"
                state.warnings = list(dict.fromkeys(state.warnings + [warning]))
        elif state.competition.validation in {"group", "time"}:
            report["reason"] = (
                "Group/time validation retained; random-seed confirmation is not applicable."
            )
        else:
            report["reason"] = (
                "Fewer than two confirmations completed within the configured experiment/time budget."
            )
        state.feedback["stability"] = report
        output = self.store.run_dir(state.run_id) / "stability.json"
        write_json(output, report)
        self.log(
            state,
            "check_finalist_stability",
            report["reason"],
            "confirmation_cv",
            report,
            "create_submission",
            [str(output)],
        )

    def create_submission(self, state, data):
        self.activity(state, "Generating predictions and validating the submission template")
        best = next(e for e in state.experiments if e.id == state.best_id)
        values = load_predictions(best.checkpoint, data, state.competition)
        c = state.competition
        submission = (
            data.sample.copy()
            if data.sample is not None
            else pd.DataFrame(
                {c.id_column: data.test[c.id_column]} if c.id_column else {}, index=data.test.index
            )
        )
        columns = [column for column in submission if column != c.id_column]
        columns = columns or [c.target]
        if c.prediction_kind == "probability":
            if len(columns) == 1 and len(data.report["classes"]) == 2:
                submission[columns[0]] = values[:, -1]
            elif len(columns) == len(data.report["classes"]) and set(columns) == set(
                map(str, data.report["classes"])
            ):
                for index, label in enumerate(data.report["classes"]):
                    submission[str(label)] = values[:, index]
            else:
                raise ValueError(
                    "Probability submission columns must identify every class, or have one column for binary positive-class probability"
                )
        else:
            if len(columns) != 1:
                raise ValueError("Label/value submission requires one prediction column")
            submission[columns[0]] = values
        output = self.store.run_dir(state.run_id) / "submissions"
        output.mkdir(exist_ok=True)
        path = output / "submission.csv"
        # Validate both in memory and after CSV serialization.
        report = validate_submission(submission, data.test, data.sample, c, data.report["classes"])
        submission.to_csv(path, index=False)
        reloaded = pd.read_csv(
            path, dtype={c.id_column: data.test[c.id_column].dtype} if c.id_column else None
        )
        if c.prediction_kind == "label" and isinstance(data.report["classes"][0], str):
            reloaded[columns[0]] = reloaded[columns[0]].astype(str)
        validate_submission(reloaded, data.test, data.sample, c, data.report["classes"])
        write_json(output / "validation.json", report)
        state.submission_path = str(path)
        state.metadata["submission_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.log(
            state,
            "validate_submission",
            "Best model has been refitted on all training rows; match template and test identifiers exactly.",
            "validate_submission",
            report,
            "leaderboard_review",
            [str(path), str(output / "validation.json")],
        )
