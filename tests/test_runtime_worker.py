"""Dataset failures survive process isolation without exposing exception values."""

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from test_swarm import setup

from kaggle_slayer.schemas import RunState
from kaggle_slayer.tools.kaggle_client import KaggleError
from kaggle_slayer.tools.runtime_worker import RuntimeOperationError, run_bounded


def widen_source_data(directory):
    # Exceed the source-data guard itself, independently of any chosen encoding.
    for name in ("train.csv", "test.csv"):
        path = Path(directory) / name
        frame = pd.read_csv(path)
        for index in range(20):
            frame[f"extra_{index}"] = range(len(frame))
        frame.to_csv(path, index=False)


@pytest.mark.parametrize(
    "change,code,option",
    [
        ("metric", "metric_requirement_unresolved", "--profile"),
        ("rows", "dataset_row_limit", "--max-rows"),
        ("dense", "dataset_dense_limit", "--max-dense-cells"),
    ],
)
def test_real_prepare_worker_reports_dataset_guard(fixture_data, tmp_path, change, code, option):
    _, competition, config = fixture_data
    if change == "metric":
        competition.metric = None
    elif change == "rows":
        config.max_rows = 20
    else:
        config.max_dense_cells = 1000
        widen_source_data(config.data_dir)
    state = RunState(run_id="guard-test", competition=competition, config=config)
    output = tmp_path / "operation"
    with pytest.raises(RuntimeOperationError) as caught:
        run_bounded("prepare", state, output, 30)
    assert caught.value.code == code
    assert option in str(caught.value)
    assert json.loads((output / "result.json").read_text()) == caught.value.diagnostic()
    assert code in (output / "worker.log").read_text()


@pytest.mark.parametrize(
    "error_type,code",
    [
        (KaggleError, "kaggle_operation_failed"),
        (ValueError, "dataset_validation_failed"),
        (RuntimeError, "bounded_operation_failed"),
    ],
)
def test_worker_diagnostics_exclude_raw_exception_values(
    fixture_data, tmp_path, monkeypatch, capsys, error_type, code
):
    from kaggle_slayer.ml.runtime_worker import main

    _, competition, config = fixture_data
    state = RunState(run_id="safe-errors", competition=competition, config=config)
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(
        json.dumps({"operation": "prepare", "state": state.model_dump(), "output": str(tmp_path)}),
        encoding="utf-8",
    )
    private = "private-credential-and-raw-row-value"

    def fail(*_):
        raise error_type(private)

    monkeypatch.setattr("kaggle_slayer.ml.runtime_worker.DataAgent.prepare", fail)
    monkeypatch.setattr("sys.argv", ["runtime_worker", str(request), str(result)])
    main()
    diagnostic = json.loads(result.read_text())
    assert diagnostic["error"] == code
    assert private not in json.dumps(diagnostic) + capsys.readouterr().err
    assert private not in str(RuntimeOperationError(private))


@pytest.mark.parametrize(
    "change,code,option",
    [
        ("metric", "metric_requirement_unresolved", "--profile"),
        ("dense", "dataset_dense_limit", "--max-dense-cells"),
    ],
)
def test_immutable_dataset_guard_stops_before_more_llm_calls(tmp_path, change, code, option):
    engine, state = setup(tmp_path, **({"max_dense_cells": 1000} if change == "dense" else {}))
    if change == "metric":
        state.competition.metric = None
        state.metadata["user_constraints"]["competition"]["metric"] = None
    else:
        widen_source_data(state.config.data_dir)
    engine.live.emit = Mock()
    result = engine.execute(state)
    assert result.status == "incomplete"
    assert result.swarm["termination_reason"] == code
    assert result.swarm["requests"] == 2
    assert option in result.error
    failed = result.swarm["actions"]["data-inspect_dataset"]
    assert failed["status"] == "failed"
    assert failed["result"]["error"] == code
    assert any(option in call.args[2] for call in engine.live.emit.call_args_list)
    assert not result.experiments
    engine.client.submit.assert_not_called()
