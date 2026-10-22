"""uv run python scripts/demo.py [--task regression]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.fixture import make_fixture
from kaggle_slayer.orchestration.engine import Engine
from kaggle_slayer.schemas import Competition, RunConfig
from kaggle_slayer.storage.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["classification", "regression"], default="classification"
    )
    parser.add_argument("--ensemble", action="store_true")
    args = parser.parse_args()
    classification = args.task == "classification"
    directory = make_fixture(Path("data") / f"fixture-{args.task}", args.task)
    c = Competition(
        competition=f"fixture-{args.task}" + ("-ensemble" if args.ensemble else ""),
        target="Survived" if classification else "SalePrice",
        id_column="PassengerId" if classification else "Id",
        task=args.task,
        metric="accuracy" if classification else "rmsle",
        sample_file="sample_submission.csv",
        target_transform="none" if classification else "log1p",
        feature_profile="titanic" if classification else "generic",
    )
    config = RunConfig(
        data_dir=str(directory),
        folds=3,
        max_experiments=5,
        n_jobs=2,
        max_seconds=300,
        ensemble=args.ensemble,
    )
    engine = Engine(Store(config.artifact_root), progress=print)
    state = engine.execute(engine.create(c, config))
    print(f"Completed {state.run_id}; {state.feedback}; submission={state.submission_path}")


if __name__ == "__main__":
    main()
