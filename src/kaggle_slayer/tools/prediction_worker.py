"""Central timeout/cancellation boundary for numerical prediction generation."""

import json
import os
import subprocess
import sys
import time

from kaggle_slayer.storage.store import write_json


def run_predictions(state, directory, timeout, cancelled):
    request, result = directory / "prediction_request.json", directory / "prediction_result.json"
    best = next(e for e in state.experiments if e.id == state.best_id)
    result.unlink(missing_ok=True)
    write_json(
        request,
        {
            "competition": state.competition.model_dump(),
            "config": state.config.model_dump(),
            "checkpoint": best.checkpoint,
            "output": str(directory / "submissions"),
        },
    )
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(state.config.n_jobs)
    with (directory / "prediction_worker.log").open("w") as log:
        with subprocess.Popen(
            [sys.executable, "-m", "kaggle_slayer.ml.predict_worker", str(request), str(result)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        ) as process:
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if cancelled() or time.monotonic() >= deadline:
                        raise RuntimeError("prediction_timeout_or_cancelled")
                    try:
                        process.wait(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode or not result.exists():
                    raise RuntimeError("prediction_worker_failed")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
    return json.loads(result.read_text(encoding="utf-8"))
