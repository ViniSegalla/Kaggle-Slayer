from pathlib import Path

from kaggle_slayer.agents.base import Agent
from kaggle_slayer.tools.data_tools import inspect_dataset, save_eda
from kaggle_slayer.tools.kaggle_client import KaggleError


class DataAgent(Agent):
    name = "data_engineer"

    def prepare(self, state, client):
        config, competition = state.config, state.competition
        if not config.data_dir:
            try:
                state.metadata["kaggle"] = client.metadata(competition.competition)
            except KaggleError as exc:
                state.warnings.append(str(exc))
            config.data_dir = str(
                client.download(
                    competition.competition, Path("data") / competition.competition
                ).resolve()
            )
            state.metadata["data_source"] = "official_kaggle_api"
            self.log(
                state,
                "download_dataset",
                "Obtain competition files using authorized Kaggle access.",
                "kaggle.competition_download_files",
                {"data_dir": config.data_dir},
                "inspect_csv",
            )
        else:
            config.data_dir = str(Path(config.data_dir).resolve())
            manifest = Path(config.data_dir) / "provenance.json"
            if manifest.exists():
                import json

                state.metadata["provenance"] = json.loads(manifest.read_text(encoding="utf-8"))
            state.metadata.setdefault("data_source", "local_user_supplied")
        if not competition.metric:
            raw = state.metadata.get("kaggle", {})
            metric = (
                str(raw.get("evaluationMetric", raw.get("evaluation_metric", "")))
                .lower()
                .replace(" ", "")
            )
            mapping = {
                "accuracy": "accuracy",
                "categorizationaccuracy": "accuracy",
                "auc": "roc_auc",
                "logloss": "log_loss",
                "rmse": "rmse",
                "rootmeansquarederror": "rmse",
                "rmsle": "rmsle",
                "rootmeansquaredlogarithmicerror": "rmsle",
                "mae": "mae",
                "meanabsoluteerror": "mae",
            }
            competition.metric = mapping.get(metric)
        data = inspect_dataset(Path(config.data_dir), competition, config)
        if (
            state.report.get("fingerprints")
            and state.report["fingerprints"] != data.report["fingerprints"]
        ):
            raise ValueError("Dataset fingerprints changed; start a new run instead of resuming")
        state.report = data.report
        state.warnings = list(dict.fromkeys(state.warnings + data.report["warnings"]))
        artifacts = save_eda(data, self.store.run_dir(state.run_id) / "eda")
        self.log(
            state,
            "prepare_fold_safe_data",
            "Resolve schema, audit leakage risks and fix validation splits before model selection.",
            "inspect_dataset",
            data.report,
            "planner.plan",
            artifacts,
        )
        return data
