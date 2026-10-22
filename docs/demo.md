# Demo walkthrough

## Recorded official competition

Use the [Home Data audit package](../demo/home-data-for-ml-course-8ed561cbed5e/README.md) for evaluation. It preserves the live agentic run with public MAE **14,252.33305**, submission **57535510**, status **COMPLETE**.

From the repository root:

```bash
uv sync --extra dev
uv run python demo/home-data-for-ml-course-8ed561cbed5e/audit.py verify
```

Expected verification output:

```json
{
  "run_id": "home-data-for-ml-course-8ed561cbed5e",
  "verified_files": 70,
  "verified_bytes": 13311505,
  "local_only_files_skipped": 7,
  "status": "completed",
  "local_cv_mae": 15257.827780355281,
  "public_mae": 14252.33305,
  "submission_reference": 57535510
}
```

The seven skipped files contain per-row training labels and are excluded from a fresh clone. The 70 shareable files include all six checkpoints.

Start the recorded-run dashboard:

```bash
uv run python demo/home-data-for-ml-course-8ed561cbed5e/audit.py dashboard --port 8502
```

Open [the dashboard](http://localhost:8502). It selects the recorded run automatically. Viewing the recorded run does not train or upload again.

Follow this inspection path in the dashboard or the linked evidence:

1. Compare the [six experiments and their CV scores](../demo/home-data-for-ml-course-8ed561cbed5e/metrics.json), then inspect the [agent decisions](../demo/home-data-for-ml-course-8ed561cbed5e/runs/home-data-for-ml-course-8ed561cbed5e/decisions.jsonl).
2. Check the evaluator's [selection of `ens_lgbm_cat`](../demo/home-data-for-ml-course-8ed561cbed5e/runs/home-data-for-ml-course-8ed561cbed5e/evaluation.json), the equal-weight LightGBM + CatBoost winner with local CV MAE **15,257.82778**.
3. Inspect the [submission validation](../demo/home-data-for-ml-course-8ed561cbed5e/runs/home-data-for-ml-course-8ed561cbed5e/submissions/validation.json): 1,459 rows, `Id`/`SalePrice` columns and verified identifier alignment.
4. Compare the [upload receipt](../demo/home-data-for-ml-course-8ed561cbed5e/runs/home-data-for-ml-course-8ed561cbed5e/operations/upload/result.json) with [official Kaggle feedback](../demo/home-data-for-ml-course-8ed561cbed5e/runs/home-data-for-ml-course-8ed561cbed5e/operations/final-feedback/result.json): submission **57535510**, status **COMPLETE**, public MAE **14,252.33305**.

## Standalone report

GitHub displays the HTML file as source. To view the standalone report, run this in a separate terminal:

```bash
uv run python demo/home-data-for-ml-course-8ed561cbed5e/audit.py serve --port 8766
```

Open [the report](http://localhost:8766). The report links to original checkpoints, decisions, validation reports and upload receipts. Account standing is shown separately from the submission-specific public score.

## Offline integration checks

These examples use scripted fake LLM responses and synthetic data with real training and prediction validation:

```bash
uv run python -m examples.scripted_swarm --output artifacts/swarm-example
uv run python -m examples.scripted_swarm --task regression --output artifacts/swarm-regression
```

For deterministic baselines and development datasets, see [offline workflows](offline-baselines.md). Historical automated checks are catalogued in the [engineering evidence index](../evidence/README.md).

## Run a new competition

Configure the provider/model/key in `.env`, configure Kaggle credentials and accept the competition rules if required. Then run:

```bash
uv run kaggle-slayer doctor --competition home-data-for-ml-course --require-kaggle
uv run kaggle-slayer run --competition home-data-for-ml-course --no-submit
```

Review that new run's predictions and validation report. To authorize one upload, use its actual run ID:

```bash
uv run kaggle-slayer submit RUN_ID
uv run kaggle-slayer refresh RUN_ID
```

The saved audit run has already used its upload allowance. New executions create their own run IDs and local evidence. Ambiguous upload outcomes are not retried automatically.

See [reproducibility](reproducibility.md) for dependency and environment setup, [competition profiles](experiments.md#configure-another-competition) for custom datasets, and [agentic usage](agentic-usage.md) for budgets and recovery.
