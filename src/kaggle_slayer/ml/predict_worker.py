"""Fixed prediction RPC, never generated code. No shared state or SQLite access."""

import json
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

from kaggle_slayer.schemas import Competition, RunConfig
from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.data_tools import inspect_dataset
from kaggle_slayer.tools.prediction_tools import generate_predictions


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    competition = Competition.model_validate(request["competition"])
    config = RunConfig.model_validate(request["config"])
    with threadpool_limits(limits=config.n_jobs):
        data = inspect_dataset(Path(config.data_dir), competition, config)
        result = generate_predictions(
            request["checkpoint"], data, competition, Path(request["output"])
        )
    write_json(Path(sys.argv[2]), result)


if __name__ == "__main__":
    main()
