"""Fixed local RPC with hard deadline/cancellation, including external API latency."""

import json
import os
import subprocess
import sys
import time

from kaggle_slayer.storage.store import write_json

OPERATION_ERRORS = {
    "metric_requirement_unresolved": (
        "Competition metric could not be resolved from the profile or official metadata.",
        "Set a supported metric in a competition YAML profile with --profile and start a new run.",
    ),
    "dataset_row_limit": (
        "CSV exceeds the configured max_rows guard.",
        "Reduce the dataset or set --max-rows explicitly for a new run.",
    ),
    "dataset_dense_limit": (
        "Encoded dataset exceeds the configured max_dense_cells memory guard.",
        "Reduce data/features or set --max-dense-cells explicitly for a new run.",
    ),
    "dataset_validation_failed": (
        "Dataset validation failed.",
        "Inspect the local operation worker.log and review the competition profile and CSV files.",
    ),
    "kaggle_operation_failed": (
        "The Kaggle operation failed.",
        "Check network access, credentials and competition rules with doctor --require-kaggle.",
    ),
    "bounded_operation_failed": (
        "The isolated runtime operation failed.",
        "Inspect the local operation worker.log for the failing code location.",
    ),
}


class RuntimeOperationError(RuntimeError):
    """Only controlled descriptions cross the worker/provider boundary."""

    def __init__(self, code):
        self.code = code if code in OPERATION_ERRORS else "bounded_operation_failed"
        self.message, self.action = OPERATION_ERRORS[self.code]
        super().__init__(f"{self.code}: {self.message} {self.action}")

    def diagnostic(self):
        return {"error": self.code, "message": self.message, "action": self.action}


def run_bounded(operation, state, directory, timeout, cancelled=lambda: False):
    if operation not in {"prepare", "feedback", "upload"}:
        raise ValueError("Unknown runtime operation")
    if timeout <= 0:
        raise RuntimeError("runtime_deadline")
    directory.mkdir(parents=True, exist_ok=True)
    request, result = directory / "request.json", directory / "result.json"
    result.unlink(missing_ok=True)
    write_json(
        request, {"operation": operation, "state": state.model_dump(), "output": str(directory)}
    )
    environment = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[key] = str(state.config.n_jobs)
    with (directory / "worker.log").open("w") as log:
        with subprocess.Popen(
            [sys.executable, "-m", "kaggle_slayer.ml.runtime_worker", str(request), str(result)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        ) as process:
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if cancelled() or time.monotonic() >= deadline:
                        raise RuntimeError("runtime_deadline_or_cancelled")
                    try:
                        process.wait(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode or not result.exists():
                    raise RuntimeError("runtime_worker_failed")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
    value = json.loads(result.read_text(encoding="utf-8"))
    if value.get("error"):
        raise RuntimeOperationError(value["error"])
    return value
