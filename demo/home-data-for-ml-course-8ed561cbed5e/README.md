# Personal-best Kaggle demo

Run **home-data-for-ml-course-8ed561cbed5e** is the featured audit demo, recorded on October 22, 2026. The user reported a new personal best; the saved Kaggle response records **public MAE 14,252.33305**, submission **57535510**, status **COMPLETE**.

The [offline report](index.html) contains results and evidence links; [metrics.json](metrics.json) provides exact values.

## Verify and display

After installing dependencies with <code>uv sync --extra dev</code>, run from the repository root:

~~~bash
uv run python demo/home-data-for-ml-course-8ed561cbed5e/audit.py verify
~~~

Start each viewer in a separate terminal:

~~~bash
uv run python demo/home-data-for-ml-course-8ed561cbed5e/audit.py serve --port 8766
~~~

~~~bash
uv run python demo/home-data-for-ml-course-8ed561cbed5e/audit.py dashboard --port 8502
~~~

Open [the report](http://localhost:8766) or [the one-run dashboard](http://localhost:8502). These commands inspect saved evidence without training, LLM requests or another upload. The derived SQLite index is rebuilt from the original state and decisions, with paths rebased for the current checkout.

## Recorded result

| Measure | Value |
|---|---|
| Model | Equal-weight LightGBM + CatBoost ensemble, <code>ens_lgbm_cat</code> |
| Validation | Five-fold KFold, seed 42; MAE, lower is better |
| Local CV MAE | **15,257.82778**, fold SD **1,719.32771** |
| Kaggle public MAE | **14,252.33305** |
| Submission | **57535510**, COMPLETE, October 22, 2026 at 15:59 São Paulo time |
| Dataset | 1,460 train / 1,459 test rows, 79 input features |
| Experiments | Six completed; no training timeouts |
| Provider | OpenRouter / <code>anthropic/claude-sonnet-5.5</code> |
| Requests | 19 total: 16 successful, 3 failed responses; 3 schema repairs |
| Recorded usage | 190,932 charged tokens; zero invalid actions |
| Wall time | 292.406 seconds |

The ensemble improved on the strongest single model, <code>tune_lgbm_b</code>, by 293.00253 local CV MAE (1.88%) on the same folds. The large train/CV gap warning is retained; no independent holdout result is claimed.

The account standing is **304 / 3,876 teams (top 7.84%)**, scoped to <code>current_account_best</code>. Account metadata is recorded separately from the submission-specific score.

## Audit evidence

- [State](runs/home-data-for-ml-course-8ed561cbed5e/state.json), [summary](runs/home-data-for-ml-course-8ed561cbed5e/summary.json), [swarm receipt](runs/home-data-for-ml-course-8ed561cbed5e/swarm_receipt.json), [decisions](runs/home-data-for-ml-course-8ed561cbed5e/decisions.jsonl) and [progress](runs/home-data-for-ml-course-8ed561cbed5e/progress.log).
- [Experiment comparison](runs/home-data-for-ml-course-8ed561cbed5e/comparison.html) and [evaluation](runs/home-data-for-ml-course-8ed561cbed5e/evaluation.json), with fold scores and diagnostics.
- [Upload receipt](runs/home-data-for-ml-course-8ed561cbed5e/operations/upload/result.json) and [Kaggle feedback](runs/home-data-for-ml-course-8ed561cbed5e/operations/final-feedback/result.json).
- [Selected checkpoint](runs/home-data-for-ml-course-8ed561cbed5e/experiments/ens_lgbm_cat/model.joblib), [submitted CSV](runs/home-data-for-ml-course-8ed561cbed5e/submissions/submission.csv) and [validation](runs/home-data-for-ml-course-8ed561cbed5e/submissions/validation.json).
- [SHA-256 manifest](manifest.json), covering 77 original files, including all six trained checkpoints.

## Preservation and portability

The completed run was moved from <code>artifacts/runs/</code> into this permanent directory without changing its files. The old ignored path is a junction so existing local history still resolves. Use the audit helper for inspection; execution or CLI feedback refresh can rewrite evidence and invalidate its original checksums.

The 70 shareable original files are verified by default. The 7 per-row label outputs (OOF arrays and the original target chart) stay local and Git-ignored. Verify those too when available:

~~~bash
uv run python demo/home-data-for-ml-course-8ed561cbed5e/audit.py verify --full
~~~

Raw input CSVs and credentials are excluded. Original snapshots retain historical paths; only the disposable dashboard database rebases them. The report, checkpoints, receipts, configuration, fingerprints and environment snapshot are available for audit from a checkout. Hash verification and display do not contact Kaggle or the LLM provider.
