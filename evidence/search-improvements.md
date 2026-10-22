# General search improvements: measured local benchmark

The updated workflow ran on cached official Home Data CSVs with deterministic
planning, five folds and the default twelve-experiment budget. No API requests,
LLM calls or Kaggle uploads were made by this benchmark.

| Measure | MAE (lower is better) |
|---|---:|
| Previous workflow selected LightGBM | 16,334.14869 |
| Updated workflow best single model | 15,771.87758 |
| Updated workflow selected ensemble | 15,539.69267 |
| Ensemble, confirmation seed 1051 | 15,813.39020 |
| Single LightGBM, confirmation seed 1051 | 16,092.95640 |

The selected ensemble improved primary CV MAE by 4.86% relative to the previous
selected model. Dataset fingerprints and primary split indices were checked and
matched exactly between runs. The ensemble also won the separate split-seed
comparison; confirmation did not alter model selection or trigger more tuning.
The large training/CV gap remains flagged. This comparison does not isolate the
contribution of each code change and does not establish a public-score gain,
statistical significance, or an unbiased estimate on untouched data.

The old run is `home-data-for-ml-course-10b1360767ac`; the updated run is
`home-data-for-ml-course-8a27281cf635`. Machine-generated scores, statuses,
durations and confirmation results are in [search-improvements.json](search-improvements.json).
Private datasets, checkpoints and full run traces remain under ignored `data/`
and `artifacts/runs/` directories.

Reproduce after downloading/accepting rules through the normal workflow:

```powershell
uv sync --locked --extra dev
$env:KS_LLM_PROVIDER = 'deterministic'
uv run python -m kaggle_slayer run --competition home-data-for-ml-course --data-dir data/home-data-for-ml-course --no-submit
```

The bundled Home Data profile fixes the known target and MAE metric, permitting
this local cached-data benchmark without a metadata API request. New run IDs and
training times vary; numerical results may vary with platform/library versions.
This benchmark does not modify the private `.env` file.

Validation performed: the full offline suite passed 130 tests (six upstream
LightGBM deprecation warnings). Follow-up targeted checks passed 38 tests,
covering small-budget handling, fold guards and multi-family Optuna allocation.
The wheel built successfully and passed isolated installation/profile discovery.
Ruff checks and formatting passed. These are local executions; no new GitHub
Actions run or Kaggle evaluation is claimed.
