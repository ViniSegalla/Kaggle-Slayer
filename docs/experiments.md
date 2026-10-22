# Experiment protocol

## Validation

Classification uses shuffled stratified CV by default; regression uses shuffled K-fold. Group and strictly ordered time splits are explicit profile choices. Group/time columns and IDs are excluded from features. Every classification training/validation fold must contain all classes. Time validation rejects tied timestamps to avoid ambiguous boundaries; use grouped periods when needed. No test labels or public leaderboard values enter fitting, trial scoring or selection.

The preprocessing pipeline performs row-local features, numeric median imputation, standard scaling, categorical missing-value handling and capped one-hot encoding. Imputers, scalers and encoders fit only the current training fold. Categorical expansion is capped at 64 values per field; unseen values are handled safely. Numeric infinities become missing values. The Titanic profile adds family size, traveling-alone and fare-per-person features.

Each fold computes the metric on its held-out rows. The reported score is the mean of fold scores; the standard deviation reports fold variability, not a confidence interval. Training-fold scores provide an overfitting diagnostic. OOF arrays include a validation mask, which excludes the initial training-only window for time splits. Selection across repeated CV trials can itself overfit the folds; use an untouched external holdout for stronger estimates.

## Models and optimization

The default roster is logistic regression/Ridge, Random Forest, LightGBM, CatBoost and XGBoost. All are installed by the standard `uv sync --locked --extra dev`. Histogram gradient boosting is also available. Change `models` in `configs/default.yaml` to choose at least three distinct families; a separate run profile is not required for the five defaults.

LightGBM, XGBoost and CatBoost choose stopping iterations with an inner holdout of the **outer training fold**, then refit on the entire outer training fold at that iteration count. The final full-data fit uses the median chosen iteration count. Group/time validation disables random inner early stopping for these boosters. Histogram gradient boosting uses its internal validation fraction. For very small folds (<80 rows), boosters use fixed iteration budgets.

Each experiment includes CV and a full-training-data refit. This makes all completed checkpoints immediately usable if a later trial exhausts the budget. Model parameters, effective stopping parameters, elapsed time, fold scores, OOF predictions and available feature importances are retained. Optional log1p regression transforms are applied inside the estimator and inverse-transformed at prediction time. RMSLE predictions are clipped to zero, consistently for validation and output.

Optuna uses a seeded TPE sampler, a persistent local study and bounded model-specific search spaces. The strongest completed family gets the next trials. Failed/time-limited trials are marked failed. Optimization trials count toward the same experiment budget. Defaults allow ten experiments: five baselines followed by up to five tuning trials, across at most three tuning rounds with two trials per round. Existing timeouts and thread caps still apply. An explicit five-experiment budget leaves no tuning slots with the default roster. Resuming a saved run uses its saved budget and roster rather than the new defaults.

`ensemble: true` (see `configs/ensemble.yaml`) reserves one experiment for the strongest two distinct validated families. Their effective configurations are refitted inside every outer fold with a shared fold-fit preprocessor, and combined by soft voting for classification or equal-weight averaging for regression. The ensemble undergoes the same scoring, checkpointing and final ranking as other candidates. It is selected only if its measured CV score wins; otherwise the stronger single model remains selected. This consumes the same total experiment/time budgets and never combines test labels or public leaderboard results.

## Metrics and output

Higher is better: accuracy, binary F1, binary ROC AUC and R². Lower is better: log loss, RMSE, RMSLE, natural-log RMSE and MAE. Binary probability output uses the last class in sorted class order as the positive class. Multiclass log loss uses the complete probability matrix. A multiclass probability template must name its prediction columns with the class labels.

Output validation checks exact column order, row count, non-missing finite values, label membership, probability range/sums and identifier alignment against both test data and the sample. IDs are read verbatim to preserve leading zeros. CSV is read back and revalidated after serialization.

House Prices uses `log_rmse`: RMSE on natural-log sale prices, with strictly positive predictions. Its log1p fitting transform is an independent modeling choice; scoring uses the actual natural logarithm. Generic `rmsle` instead scores log1p values and permits zero targets.
