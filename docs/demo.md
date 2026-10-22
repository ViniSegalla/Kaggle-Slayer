# Demo walkthrough

## Recorded official competition

Use the [Home Data audit package](../demo/home-data-for-ml-course-8ed561cbed5e/README.md) for evaluation. It preserves the live agentic run with public MAE **14,252.33305**, submission **57535510**, status **COMPLETE**.

From the repository root:

```bash
uv sync --locked --extra dev
uv run --locked python demo/home-data-for-ml-course-8ed561cbed5e/audit.py verify
uv run --locked python demo/home-data-for-ml-course-8ed561cbed5e/audit.py dashboard --port 8502
```

Open [the dashboard](http://localhost:8502). It selects the recorded run automatically. Inspect specialist handoffs, all six experiments, the selected ensemble, prediction validation and official Kaggle feedback. Viewing the recorded run does not train or upload again.

For the standalone report, run this in a separate terminal:

```bash
uv run --locked python demo/home-data-for-ml-course-8ed561cbed5e/audit.py serve --port 8766
```

Open [the report](http://localhost:8766). The report links to original checkpoints, decisions, validation reports and upload receipts. Account standing is shown separately from the submission-specific public score.

## Offline integration checks

These examples use scripted fake LLM responses and synthetic data with real training and prediction validation:

```bash
uv run --locked python -m examples.scripted_swarm --output artifacts/swarm-example
uv run --locked python -m examples.scripted_swarm --task regression --output artifacts/swarm-regression
```

For deterministic baselines and development datasets, see [offline workflows](offline-baselines.md). Historical automated checks are catalogued in the [engineering evidence index](../evidence/README.md).

## Run a new competition

Configure the provider/model/key in `.env`, configure Kaggle credentials and accept the competition rules if required. Then run:

```bash
uv run --locked python -m kaggle_slayer doctor --competition home-data-for-ml-course --require-kaggle
uv run --locked python -m kaggle_slayer run --competition home-data-for-ml-course --no-submit
```

Review that new run's predictions and validation report. To authorize one upload, use its actual run ID:

```bash
uv run --locked python -m kaggle_slayer submit RUN_ID
uv run --locked python -m kaggle_slayer refresh RUN_ID
```

The saved audit run has already used its upload allowance. New executions create their own run IDs and local evidence. Ambiguous upload outcomes are not retried automatically.

See [reproducibility](reproducibility.md) for dependency and environment setup, and [agentic usage](agentic-usage.md) for budgets and recovery.
