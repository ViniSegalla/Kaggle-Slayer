"""Export only public/synthetic demo evidence; never private competition datasets/models.

uv run python scripts/export_evidence.py
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import plotly.express as px

from kaggle_slayer.storage.store import Store, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-run",
        action="append",
        default=[],
        help="Explicitly include aggregate evidence for a completed official run; excludes dataset rows and its submission CSV",
    )
    args = parser.parse_args()
    store = Store("artifacts")
    destination = Path("evidence")
    destination.mkdir(exist_ok=True)
    states = [json.loads(row["state"]) for row in store.rows("runs")]
    selected = {}
    for state in sorted(states, key=lambda item: item["updated_at"]):
        kind = state["metadata"].get("provenance", {}).get("kind")
        if state["status"] == "completed" and kind in {"synthetic_fixture", "public_dataset_demo"}:
            selected[state["competition"]["competition"]] = state
        elif (
            state["status"] == "completed"
            and state["run_id"] in args.official_run
            and state["metadata"].get("data_source") == "official_kaggle_api"
        ):
            selected[state["competition"]["competition"]] = state
    index = []
    workspace = str(Path.cwd().resolve())

    def sanitize(text):
        return text.replace(workspace.replace("\\", "\\\\"), "<workspace>").replace(
            workspace, "<workspace>"
        )

    for slug, state in selected.items():
        official = state["metadata"].get("data_source") == "official_kaggle_api"
        provenance = state["metadata"].get("provenance") or {
            "kind": "official_kaggle_api",
            "description": "Authenticated competition download; dataset rows and submission predictions excluded from curated evidence",
        }
        source = store.run_dir(state["run_id"])
        output = destination / slug
        output.mkdir(exist_ok=True)
        for name in (
            "summary.json",
            "evaluation.json",
            "decisions.jsonl",
            "public_holdout_evidence.json",
            "next_run_plan.json",
        ):
            if (source / name).exists():
                (output / name).write_text(
                    sanitize((source / name).read_text(encoding="utf-8")), encoding="utf-8"
                )
        for name in ("validation.json",) if official else ("submission.csv", "validation.json"):
            if (source / "submissions" / name).exists():
                shutil.copy2(source / "submissions" / name, output / name)
        write_json(output / "environment.json", state["metadata"]["environment"])
        write_json(output / "provenance.json", provenance)
        (output / "dataset_report.json").write_text(
            sanitize(json.dumps(state["report"], indent=2)), encoding="utf-8"
        )
        frame = pd.DataFrame(state["experiments"])
        frame[
            [
                "id",
                "model",
                "status",
                "score",
                "std",
                "train_score",
                "duration",
                "params",
                "effective_params",
                "error",
            ]
        ].to_csv(output / "experiments.csv", index=False)
        completed = frame[frame.status == "completed"]
        px.bar(
            completed,
            x="id",
            y="score",
            error_y="std",
            color="model",
            template="plotly_dark",
            title=f"Actual {slug} CV · {state['competition']['metric']}",
        ).write_html(output / "comparison.html", include_plotlyjs="cdn")
        index.append(
            {
                "competition": slug,
                "run_id": state["run_id"],
                "provenance": provenance["kind"],
                "best_id": state["best_id"],
                "score": state["feedback"]["local_score"],
                "metric": state["competition"]["metric"],
                "public_score": state["feedback"].get("leaderboard", {}).get("public_score"),
                "submission_status": state["feedback"].get("leaderboard", {}).get("status"),
                "submission_reference": state["submissions"][-1].get("reference")
                if state["submissions"]
                else None,
                "experiments": len(state["experiments"]),
                "successful_experiments": sum(
                    e["status"] == "completed" for e in state["experiments"]
                ),
            }
        )
    write_json(destination / "index.json", index)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
