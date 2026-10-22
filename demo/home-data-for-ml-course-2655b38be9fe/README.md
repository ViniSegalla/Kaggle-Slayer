# Auditable live Kaggle demo

This package preserves the completed `home-data-for-ml-course-2655b38be9fe` run from October 22, 2026. A live LLM swarm inspected official Kaggle data, configured validation, compared models, measured an ensemble, validated its predictions and completed one official submission. Promoting this run did not train another model or upload another CSV.

Open the [offline audit report](index.html) for the results, artifact links and embedded experiment comparison. [metrics.json](metrics.json) provides the same headline facts in a compact machine-readable form.

## Inspect and verify

Run these commands from the repository root after `uv sync --locked --extra dev`:

```bash
uv run --locked python demo/home-data-for-ml-course-2655b38be9fe/audit.py verify
uv run --locked python demo/home-data-for-ml-course-2655b38be9fe/audit.py serve --port 8766
uv run --locked python demo/home-data-for-ml-course-2655b38be9fe/audit.py dashboard --port 8502
```

The static report is available at [localhost:8766](http://localhost:8766), and the dashboard at [localhost:8502](http://localhost:8502). The report and its comparison plot use local assets and work without Kaggle or an LLM connection. The default verifier checks the shareable files against the SHA-256 [manifest](manifest.json). To also check retained local files containing individual training targets or out-of-fold labels, use:

```bash
uv run --locked python demo/home-data-for-ml-course-2655b38be9fe/audit.py verify --full
```

## Recorded result

| Measure | Recorded value |
|---|---|
| Competition | Housing Prices Competition for Kaggle Learn Users (`home-data-for-ml-course`) |
| Execution | Agentic, `swarm-v1`; `completed` / `coordinator_finished` |
| Validation | Five-fold KFold, seed 42; MAE, lower is better |
| Selected model | Equal-weight LightGBM + HistGradient ensemble, `measure_lgbm_hist_blend` |
| Local CV MAE | **15,839.30481**, fold standard deviation **1,630.37763** |
| Official public MAE | **14,840.83581** |
| Official submission | **57534881**, `COMPLETE`; October 22, 2026 at 15:33 São Paulo time |
| Dataset | 1,460 training rows, 1,459 test rows, 79 input features |
| Model attempts | Four completed; one Random Forest timeout at 180 seconds |
| Live provider | OpenRouter / `openai/gpt-6-luna`; 32 successful requests |
| Recorded usage | 247,349 charged tokens; one schema repair; zero invalid actions |
| Wall time | 397.985 seconds |

The ensemble improved the primary LightGBM result from MAE **16,064.99304** by **225.68823** (about **1.40%**) on the same frozen folds. The evaluator retained a warning about the large training/CV gap; these results do not establish generalization beyond the recorded validation and Kaggle public score.

Kaggle also returned account standing **352 / 3,876 teams (9.08%)**, checked at 15:33 São Paulo time. Its recorded scope is `current_account_best`: it can reflect a different submission. **The account percentile is not a rank attributed to submission 57534881.**

## Evidence trail

The original files are under [`runs/home-data-for-ml-course-2655b38be9fe/`](runs/home-data-for-ml-course-2655b38be9fe/).

- [State snapshot](runs/home-data-for-ml-course-2655b38be9fe/state.json), [swarm receipt](runs/home-data-for-ml-course-2655b38be9fe/swarm_receipt.json) and [summary](runs/home-data-for-ml-course-2655b38be9fe/summary.json): configuration, data fingerprints, validation lineage, actions, results and provider usage.
- [Decision trace](runs/home-data-for-ml-course-2655b38be9fe/decisions.jsonl) and [progress log](runs/home-data-for-ml-course-2655b38be9fe/progress.log): delegation, model choices, timeout, evaluation and prediction validation.
- [Evaluation](runs/home-data-for-ml-course-2655b38be9fe/evaluation.json) and [comparison plot](runs/home-data-for-ml-course-2655b38be9fe/comparison.html): measured experiments, selection and warnings.
- [Upload response](runs/home-data-for-ml-course-2655b38be9fe/operations/upload/result.json) and [feedback receipt](runs/home-data-for-ml-course-2655b38be9fe/operations/final-feedback/result.json): actual Kaggle upload and subsequent feedback. The decision trace ends before upload; inspect these receipts for submission proof.
- [Selected checkpoint](runs/home-data-for-ml-course-2655b38be9fe/experiments/measure_lgbm_hist_blend/model.joblib), [submission CSV](runs/home-data-for-ml-course-2655b38be9fe/submissions/submission.csv) and [validation report](runs/home-data-for-ml-course-2655b38be9fe/submissions/validation.json): the saved model and its 1,459 validated predictions.

## Provenance and portability

The original run directory was moved here from `artifacts/runs/` without changing its files. The manifest records file hashes so that the preserved execution can be audited independently of the current dashboard database. The state snapshot and swarm receipt are distinct recorded snapshots; their hashes differ because the state includes a later feedback refresh.

On the original workstation, the previous ignored run path is a filesystem junction to this directory so existing local history still resolves. Use the audit helper for inspection: ordinary execution or feedback-refresh commands can rewrite archived run files and will then fail the original checksum verification.

Original snapshots retain their historical absolute paths. `audit.py dashboard` builds an ignored `metadata.sqlite` for display and rebases artifact paths in that derived database; it does not rewrite the original snapshots. The generated database is disposable, and a fresh clone can recreate it from the preserved state.

Raw competition input CSVs and credentials are not copied into this package. Individual out-of-fold labels/predictions (`oof.npz`) and the original target chart containing per-row target values remain local and Git-ignored; a fresh clone does not contain those private files. The default verifier covers shareable outputs, while `--full` additionally checks these files when they are available locally.

The selected Joblib checkpoint is retained for optional inference with the matching official competition data and recorded Python/package environment. Match the original data using the SHA-256 fingerprints in [metrics.json](metrics.json) or the state snapshot. Loading a checkpoint requires the project dependencies; the static audit and hash verification do not require retraining, a paid API or live Kaggle access.
