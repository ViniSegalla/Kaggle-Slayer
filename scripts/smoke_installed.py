"""Run from an isolated uv environment with the built wheel to verify package resources."""

import os
import tempfile

from kagglesdk.competitions.types.competition_api_service import (
    ApiCompetition,
    ApiListCompetitionsResponse,
)

from kaggle_slayer.config import load_config
from kaggle_slayer.tools.kaggle_client import KaggleClient

original_directory = os.getcwd()
with tempfile.TemporaryDirectory(prefix="kaggle-slayer-wheel-") as directory:
    try:
        os.chdir(directory)
        titanic, config = load_config("titanic")
        house, _ = load_config("house-prices-advanced-regression-techniques")
        assert titanic.target == "Survived" and config.max_experiments == 12
        assert {"catboost", "xgboost"} <= set(config.models)
        assert house.metric == "log_rmse" and house.target == "SalePrice"
        item = ApiCompetition()
        item.ref, item.evaluation_metric = "titanic", "Accuracy"
        response = ApiListCompetitionsResponse()
        response.competitions = [item]

        class FakeApi:
            def competitions_list(self, **kwargs):
                return response

        assert KaggleClient(FakeApi()).metadata("titanic")["evaluationMetric"] == "Accuracy"
        print(
            "Installed wheel profile discovery: passed (Titanic + House Prices outside repository)"
        )
    finally:
        os.chdir(original_directory)

print("Installed wheel Kaggle SDK response adapter: passed (mocked; no network)")
