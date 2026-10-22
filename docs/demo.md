# Practical demo walkthrough

For the featured live competition evidence, start with the [recorded Home Data demo](../demo/home-data-for-ml-course-8ed561cbed5e/README.md). The development walkthrough below uses compatibility workflows; see [offline baselines](offline-baselines.md) for their configuration. The current `.env.example` selects OpenRouter for agentic runs.

## Five-minute offline demonstration

From the repository root:

```powershell
uv python install
uv sync --locked --extra dev
$env:KS_LLM_PROVIDER = 'deterministic'
uv run python -m kaggle_slayer doctor
uv run python scripts/demo.py
uv run python scripts/demo.py --task regression
uv run streamlit run dashboard/app.py
```

Open [the console](http://localhost:8501). Select a fixture run. Explain the provenance notice first: these are synthetic integration demonstrations. The overview shows actual dataset sizes, measured best CV and budget use. In Agent Activity, inspect the Data Engineer's quality report, Planner queue, Modeling Agent results, Evaluator rejection feedback and Reviewer record stating no upload happened.

In Experiments, show the linear baseline, two distinct nonlinear families and two Optuna trials. Tuning need not beat the baseline; the evaluator retains the strongest measured candidate. Inspect effective stopping parameters, fold scores and the saved checkpoint reference. In Artifacts, preview the generated CSV and validation JSON. Every ID must follow test-file order and every prediction must meet the configured output contract.

## Real public-data demonstration

```bash
uv run python scripts/public_titanic_demo.py
```

This downloads public example observations from the seaborn data repository, not the authenticated Kaggle API. It uses 712 training rows and an untouched 179-row holdout. The holdout is evaluated once after selection, and its score is written to `public_holdout_evidence.json`. It is not used by the planner. The demo slug, provenance and upload rejection prevent confusion with the official Titanic test set.

Actual recorded metrics, charts, logs and screenshots are in [evidence](../evidence/README.md). The public dataset URL can change upstream; the recorded SHA-256 identifies the bytes used for the evidence. Private Kaggle datasets and large models are intentionally absent from the repository.

## Official Titanic completion after authentication

1. Configure official Kaggle credentials using `.env` or the Kaggle configuration directory. This historical walkthrough uses deterministic execution; set `KS_LLM_PROVIDER=deterministic` in the current shell to disable optional LLM advice, as described in the offline baseline guide.
2. Open [Titanic](https://www.kaggle.com/competitions/titanic) and accept its rules manually, if required.
3. Verify access, download and train:

```bash
uv run python -m kaggle_slayer doctor --competition titanic --require-kaggle
uv run python -m kaggle_slayer run --competition titanic --execution-mode deterministic --max-experiments 5 --no-submit
```

4. Review the generated CSV, validation report and competition rules. If you intend to upload, run the following with the actual run ID:

```bash
uv run python -m kaggle_slayer submit RUN_ID
uv run python -m kaggle_slayer refresh RUN_ID
```

The dedicated submit command is explicit authorization and does not retrain. Alternatively, start the workflow with `--submit` to authorize one upload after generation. Outcome-unknown uploads cannot be repeated automatically. Review the Kaggle website if the request was interrupted. Public scores may arrive later; `refresh` reads without uploading.

The build completed official Titanic training using existing credentials: 891 training rows, 418 test predictions, five experiments and CV accuracy 0.84060. After checking participation and remaining quota, one explicit upload was completed (reference 57504220). Kaggle returned 0.75358 public accuracy. The local/public gap triggered a proposed stricter-validation plan; no second upload was made. Do not submit this run again: its upload budget is spent. Use `uv run python -m kaggle_slayer refresh titanic-441c742e80e5` to read its feedback.

## Portability, recovery and verification

```bash
uv run python -m kaggle_slayer run --competition house-prices-advanced-regression-techniques --execution-mode deterministic --no-submit
uv run python -m kaggle_slayer runs
uv run python -m kaggle_slayer inspect RUN_ID
uv run python -m kaggle_slayer resume RUN_ID
uv run pytest
```

House Prices changes target, metric, target transformation and identifiers through YAML. An actual official House Prices run also completed: five experiments, 1,459 validated test predictions, best local natural-log RMSE 0.13105, and no upload. The separate offline regression demo uses generated data. Resume skips recorded models and rejects changed data; interrupted unrecorded work is retrained. All-model failures retain worker logs.

To export updated shareable demo evidence, run `uv run python scripts/export_evidence.py`. It selects marked public/synthetic runs by default. `--official-run RUN_ID` explicitly includes an official run's aggregate metrics, decisions and environment, while excluding its dataset rows, submission predictions and model binaries. The standard dashboard reads the live local store, not curated evidence snapshots.
