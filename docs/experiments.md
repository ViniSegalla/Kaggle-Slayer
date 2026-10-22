# Experiment protocol

## Configure another competition

Complete the [LLM setup](../README.md#llm-configuration) and [Kaggle authentication and rules](../README.md#kaggle-authentication-and-rules) first. For a competition without a built-in profile, save a YAML file at `configs/competitions/my-competition.yaml`:

```yaml
competition: my-competition
target: outcome
id_column: record_id
task: classification
metric: roc_auc
prediction_kind: probability
train_file: train.csv
test_file: test.csv
sample_file: sample_submission.csv
validation: stratified
```

Replace `my-competition` with the Kaggle URL slug and adapt the column names, filenames, task, metric and prediction format to its data and rules. This example assumes a binary target and a submission template with one probability column. Then run from the repository root:

```bash
uv run kaggle-slayer run --competition my-competition --profile configs/competitions/my-competition.yaml --no-submit
```

The profile's `competition` value must match `--competition`. Profiles stored at `configs/competitions/<slug>.yaml` are also discovered automatically; the explicit `--profile` form works with other paths. Data filenames are relative to the downloaded dataset directory or the directory supplied with `--data-dir`.

`--profile` configures competition schema, metric and validation. `--config` instead selects run budgets, model choices and execution mode; omitting it uses `configs/agentic.yaml`. A custom run config must set `execution_mode: agentic` or be paired with `--execution-mode agentic` for live agents.

Without explicit overrides, the runner infers a single train-only target and template identifier when unambiguous. Task inference uses type/cardinality; explicitly configure integer-valued regression. Metrics require a known profile, recognized API metadata or YAML. Ambiguous schemas and unresolved metrics stop the run with a diagnostic. Choose group or time validation explicitly when rows are related or ordered; set `group_column` or `time_column` accordingly.

## Validation

Classification uses shuffled stratified CV by default; regression uses shuffled K-fold. Group and strictly ordered time splits are explicit profile choices. Group/time columns and IDs are excluded from features. Every classification training/validation fold must contain all classes. Time validation rejects tied timestamps to avoid ambiguous boundaries; use grouped periods when needed. No test labels or public leaderboard values enter fitting, trial scoring or selection.

Preprocessing is model-specific. Linear models use numeric median imputation, scaling and capped one-hot encoding. Tree models use unscaled numeric imputation with missingness indicators and the same bounded one-hot encoding. CatBoost instead receives DataFrames with native categorical columns, a distinct missing token and numeric NaNs. Unseen categories remain valid native inputs. Every dtype schema, imputer, scaler and encoder is fitted only on training-fold rows; no category frequencies are learned from validation rows. Numeric infinities become missing values. The Titanic profile adds row-local family size, traveling-alone and fare-per-person features.

Each fold computes the metric on its held-out rows. The reported score is the mean of fold scores; the standard deviation reports fold variability, not a confidence interval. Training-fold scores provide an overfitting diagnostic. OOF arrays include a validation mask, which excludes the initial training-only window for time splits. Selection across repeated CV trials can itself overfit the folds; use an untouched external holdout for stronger estimates.

## Models and optimization

The default agentic profile, `configs/agentic.yaml`, allows linear models (logistic regression/Ridge), Random Forest, histogram gradient boosting, LightGBM, CatBoost and XGBoost. All are installed by `uv sync --extra dev`. Agents choose which families to train; one family and one experiment are valid. Edit `models` in the selected run profile to change the allowed roster.

The CLI selects agentic execution when `--config` is omitted. Configure `KS_LLM_PROVIDER`, `KS_LLM_MODEL` and the endpoint's credentials in `.env` before running. For a custom `--config`, include `execution_mode: agentic` or pass `--execution-mode agentic`; a config without a mode retains the schema's deterministic default. See [agentic setup](agentic-usage.md).

The agentic eight-experiment budget is a ceiling. The LLM selects baselines, bounded tuning trials, ensembles and terminal confirmations through validated tools. It can stop before exhausting the budget; Python does not reserve optional-work slots or enforce a baseline sequence. Failed training attempts consume budget. Request, token, turn and wall-time limits can also end execution. Saved runs retain their configuration.

LightGBM, XGBoost and CatBoost choose stopping iterations with an inner holdout of the **outer training fold**, then refit on the entire outer training fold at that iteration count. The final full-data fit uses the median chosen iteration count. Group/time validation disables random inner early stopping, including histogram gradient boosting's internal random holdout. For very small folds (<80 rows), boosters use fixed iteration budgets. Ensemble members also relearn stopping counts within each outer training fold rather than copying stopping counts selected with other folds' labels.

Each experiment includes CV and a full-training-data refit. This makes all completed checkpoints immediately usable if a later trial exhausts the budget. Model parameters, effective stopping parameters, elapsed time, fold scores, OOF predictions and available feature importances are retained. Optional log1p regression transforms are applied inside the estimator and inverse-transformed at prediction time. RMSLE predictions are clipped to zero, consistently for validation and output.

The deterministic compatibility profile, `configs/default.yaml`, uses five baseline families, excluding histogram gradient boosting. Its planner uses seeded Optuna TPE studies and bounded search spaces, giving the leading three distinct families a trial before exploiting the strongest family. Its twelve-experiment budget normally allows five baselines, four tuning trials, one ensemble and two confirmation fits. Five slots fund its five baselines without optional work. These scheduling rules do not apply to agentic runs. See [offline setup](offline-baselines.md) for the explicit execution-mode and provider overrides.

`ensemble: true` enables ensemble work in both profiles. Deterministic scheduling reserves a slot when the budget permits; agentic scheduling requires an explicit LLM-selected action and reserves none automatically. An agentic ensemble combines two distinct validated primary experiments; the deterministic planner selects the strongest two distinct families. Members retain independent fold-fit preprocessing. Soft voting combines classification probabilities; regression averages inverse-transformed predictions in original target units. It receives full CV evaluation, checkpointing and a full-data refit. Promotion requires a relative primary-CV improvement greater than `min_relative_improvement` (default 0.001) over the strongest single model. There is no blend-weight search or public-score feedback into selection.

## Training objectives and final checks

For MAE regression, supported tree families train with absolute-error losses; other regression metrics use squared error. Ridge remains a squared-error surrogate. Classification retains each library's default classification surrogate, not a direct optimizer of accuracy, F1 or AUC. Log-target fitting can change the effective objective relative to the original-unit metric. Each experiment's `training_contract` records preprocessing, explicit objective parameters and alignment limitations; final ranking always uses the competition metric.

`confirmation_top_k: 2` allows up to two diagnostic confirmations. In deterministic mode, a persisted frozen queue selects the winner and strongest other distinct family when resources permit. In agentic mode, agents explicitly request confirmation of a measured primary experiment; starting confirmation prevents further primary training or tuning. Confirmations freeze hyperparameters and use folds with `seed + 1009`. They count toward the total experiment budget, have role `confirmation`, and are excluded from primary ranking. They flag split-sensitive results without replacing the winner or starting another tuning cycle. Insufficient time, failure or small budgets produce an unavailable diagnostic rather than invented stability. Group/time splits skip this random-seed check. A second CV seed reuses training labels and is not an independent holdout, nested CV or an unbiased final generalization estimate.

Workers persist exact fold indices, predictions and aggregate `error_analysis.json`. Classification slices report misclassification rate; regression slices report MAE in original target units even when the ranking metric differs. Up to twelve features and thirty sufficiently populated slices are reported. Missingness, category and numeric-bin slices are exploratory, overlap, and do not establish significance. The evaluator uses the worst observed slice to propose a concrete audit and the dashboard exposes the evidence. LLM planning cautions remain separate from measured execution warnings.

## Metrics and output

Higher is better: accuracy, binary F1, binary ROC AUC and R². Lower is better: log loss, RMSE, RMSLE, natural-log RMSE and MAE. Binary probability output uses the last class in sorted class order as the positive class. Multiclass log loss uses the complete probability matrix. A multiclass probability template must name its prediction columns with the class labels.

Output validation checks exact column order, row count, non-missing finite values, label membership, probability range/sums and identifier alignment against both test data and the sample. IDs are read verbatim to preserve leading zeros. CSV is read back and revalidated after serialization.

House Prices uses `log_rmse`: RMSE on natural-log sale prices, with strictly positive predictions. Its log1p fitting transform is an independent modeling choice; scoring uses the actual natural logarithm. Generic `rmsle` instead scores log1p values and permits zero targets.
