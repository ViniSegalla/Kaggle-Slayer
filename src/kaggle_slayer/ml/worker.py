"""Internal worker; the parent enforces a hard wall timeout for this process."""

import json
import sys
import traceback
from pathlib import Path

from kaggle_slayer.schemas import Experiment
from kaggle_slayer.storage.store import write_json
from kaggle_slayer.tools.training_tools import train_candidate


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        result = train_candidate(request)
    except Exception as exc:
        traceback.print_exc()
        candidate = request["candidate"]
        result = Experiment(
            id=candidate["id"],
            model=candidate["model"],
            params=candidate["params"],
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            validation_role=candidate.get("validation_role", "primary"),
            validation_seed=request["config"]["seed"],
            parent_id=candidate.get("parent_id"),
            lineage=candidate.get("lineage"),
        ).model_dump()
    write_json(Path(sys.argv[2]), result)


if __name__ == "__main__":
    main()
