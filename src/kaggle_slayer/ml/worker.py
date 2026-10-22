"""Internal worker; the parent enforces a hard wall timeout for this process."""

import json
import sys
import traceback
from pathlib import Path

from kaggle_slayer.schemas import Experiment
from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.data_tools import DatasetError
from kaggle_slayer.tools.runtime_worker import TRAINING_RESOURCE_ERRORS, RuntimeOperationError
from kaggle_slayer.tools.training_tools import train_candidate


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        result = train_candidate(request)
    except Exception as exc:
        diagnostic = (
            RuntimeOperationError(exc.code, getattr(exc, "details", None))
            if isinstance(exc, DatasetError) and exc.code in TRAINING_RESOURCE_ERRORS
            else None
        )
        if diagnostic:
            traceback.print_tb(exc.__traceback__)
            print(str(diagnostic), file=sys.stderr)
        else:
            traceback.print_exc()
        candidate = request["candidate"]
        result = Experiment(
            id=candidate["id"],
            model=candidate["model"],
            params=candidate["params"],
            status="failed",
            error=str(diagnostic) if diagnostic else f"{type(exc).__name__}: {exc}",
            error_code=diagnostic.code if diagnostic else None,
            error_details=diagnostic.details if diagnostic else {},
            validation_role=candidate.get("validation_role", "primary"),
            validation_seed=request["config"]["seed"],
            parent_id=candidate.get("parent_id"),
            lineage=candidate.get("lineage"),
        ).model_dump()
    write_json(Path(sys.argv[2]), result)


if __name__ == "__main__":
    main()
