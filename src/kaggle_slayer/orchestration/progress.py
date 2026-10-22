"""Concise live output from controlled fields, never raw prompts or tool bodies."""

import json
import math
from datetime import datetime, timezone


class RunProgress:
    def __init__(self, store, callback):
        self.store, self.callback = store, callback

    def emit(self, run_id, agent, message):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[{run_id}] [{timestamp}] [{agent}] {message}"
        with (self.store.run_dir(run_id) / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        self.callback(line)


def decision_summary(action, result):
    """Allowlist summaries rather than rendering arbitrary provider or tool text."""
    if action in {"schedule_baselines", "revise_experiment_queue"}:
        queue = result.get("queue", [])
        return "Queue: " + ", ".join(f"{c['id']} ({c['model']})" for c in queue)
    if action == "start_experiment":
        c = result["candidate"]
        return (
            f"Experiment {result.get('experiment_index', '?')}/{result.get('max_experiments', '?')}: "
            f"starting {c['id']} | model={c['model']} | "
            f"role={c.get('validation_role', 'primary')} | timeout={result['timeout']:.0f}s"
            f" | folds={result.get('folds', '?')} | metric={result.get('metric', '?')}"
        )
    if action in {"experiment_completed", "experiment_failed"}:
        value = result.get("score")
        score = f" | CV={value:.6g}" if value is not None else ""
        return f"{result['id']}: {result['status']}{score} | duration={result['duration']:.1f}s" + (
            " | inspect worker.log" if result["status"] != "completed" else ""
        )
    if action == "select_best_validated_model":
        return (
            f"Best: {result['best_id']} | {result['metric']}={result['local_score']:.6g} | "
            f"train/CV gap={result['train_validation_gap']:.6g} | "
            f"warnings={len(result.get('warnings', []))}"
        )
    if action == "prepare_fold_safe_data":
        return (
            f"Prepared {result['train_rows']} training / {result['test_rows']} test rows | "
            f"features={len(result['features'])} | folds={len(result['fold_sizes'])}"
        )
    if action in {"llm_planning_status", "llm_planning_failed"}:
        return f"LLM planning: {result['status']} | category={result.get('category', result.get('code', 'unknown'))}"
    if action == "propose_ensemble":
        return "Evaluate equal-weight ensemble: " + ", ".join(
            m["model"] for m in result["params"]["members"]
        )
    if action == "check_finalist_stability":
        return f"Finalist confirmation: {result['status']} | same winner={result.get('same_winner', 'unavailable')}"
    if action == "review_kaggle_feedback":
        return f"Submission status={result.get('status')} | public score={result.get('public_score')} | rank={result.get('rank')}"
    return action.replace("_", " ").capitalize()


def worker_event(output, stage, **values):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": stage, **values}, allow_nan=False) + "\n")


class WorkerEvents:
    """Incrementally read only structured worker events, not worker stdout."""

    stages = {
        "inspect": "inspecting dataset and fixed folds",
        "prepare_fold": "preparing fold-specific preprocessing and training settings",
        "early_stopping": "selecting stopping iterations on inner training data",
        "fit_fold": "fitting training fold",
        "fold_complete": "validation complete",
        "refit": "refitting on all training rows",
        "saved": "checkpoint and diagnostics saved",
    }

    def __init__(self, path, folds, metric, candidate_id):
        self.path, self.folds, self.metric, self.candidate_id = path, folds, metric, candidate_id
        self.offset = 0
        self.stage = "starting worker"

    def read(self):
        if not self.path.exists():
            return []
        messages = []
        with self.path.open(encoding="utf-8") as stream:
            stream.seek(self.offset)
            while True:
                line = stream.readline(4096)
                if not line or not line.endswith("\n"):
                    break
                self.offset = stream.tell()
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or event.get("stage") not in self.stages:
                        continue
                    description = self.stages[event["stage"]]
                    fold = event.get("fold")
                    prefix = (
                        f"fold {fold}/{self.folds}: "
                        if type(fold) is int and 1 <= fold <= self.folds
                        else ""
                    )
                    value = event.get("score")
                    self.stage = prefix + description
                    suffix = (
                        f" | {self.metric}={value:.6g}"
                        if type(value) in {int, float} and math.isfinite(value)
                        else ""
                    )
                    messages.append(f"{self.candidate_id} | {self.stage}{suffix}")
                except (ValueError, TypeError):
                    continue
        return messages
