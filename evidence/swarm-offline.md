# Offline swarm evidence

These runs used **scripted fake LLM responses** and actual numerical training/inference.
No paid/live LLM endpoint or Kaggle upload was used. Scores below are synthetic local CV
measurements, not competition performance claims.

The versioned [trace](swarm-offline-trace.json) records task handoffs, action IDs,
typed tool arguments, measured results, effective pipelines and artifact SHA-256 hashes
for classification and regression. Both runs completed 31 scripted coordinator/specialist
requests, two trained experiments, a full-data refit, prediction generation and serialized
submission validation. Unknown public scores/ranks remain unknown.

| Fixture | Metric | Initial experiment | Explicit follow-up trial | Selected result |
|---|---|---:|---:|---|
| Classification | accuracy, maximize | 0.744444 | 0.788889 | `refine-tune_experiment` |
| Regression | RMSE, minimize | 18535.265928 | 18124.971681 | `refine-tune_experiment` |

In both cases `model-configure_pipeline` selected mean imputation, standard scaling,
row missing counts and signed-log numeric features. `model-run_experiment` executed
three fold-safe fits plus a full-data refit. `audit-evaluate` supplied the measured
evidence used by the scripted coordinator to delegate `refine`. Its `tune_experiment`
changed classification C or regression alpha from 2.0 to 0.25. `reaudit-evaluate`
computed the winner; `results-generate_predictions` and `results-validate_predictions`
produced/verified the final artifact. Scores are tool outputs, not model text.

The [compact trace](swarm-offline-trace.json) is committed and available in a fresh
clone. The first two commands below generate full traces at
`artifacts/swarm-example/example_trace.json` and
`artifacts/swarm-regression/example_trace.json`, respectively, together with local
models, OOF arrays and CSV outputs. These ignored files become available after
reproduction. Reproduce with:

```bash
uv run python -m examples.scripted_swarm --output artifacts/swarm-example
uv run python -m examples.scripted_swarm --task regression --output artifacts/swarm-regression
uv run python scripts/export_swarm_contracts.py
uv run pytest --timeout=300 --junitxml=artifacts/swarm-full-tests.xml
uv run ruff check src tests dashboard scripts examples
uv run ruff format --check src tests dashboard scripts examples
```

The regression suite uses HTTP mock transports/scripted providers and local fixtures.
Coverage includes observation-dependent follow-up with the same provider policy but
different measured datasets, chosen transformations/parameters in fitted pipelines,
concurrent LLM sessions and training reservations, invalid actions and injections,
failed/exhausted budgets, hard deadlines/cancellation, terminal confirmation barriers,
comparison lineages, worker-result adoption, interrupted handoff repair, transient Windows
checkpoint locks, native/JSON protocols, and submission provenance/hash/authorization/
ambiguous-upload protections. Existing metric/leakage/group/time tests remain enabled.

Windows verification uses a workspace uv cache and the installed locked environment.
Streamlit AppTest needs local loopback access. The existing 12-experiment default-roster
test uses a 300-second test ceiling; its configured run/resource limits are unchanged.
LightGBM emits existing `eval_set` deprecation warnings. **Final result: 175 passed**, with Ruff lint/format and locked offline dependency sync
passing. See the [verification report](swarm-verification.json) and
[JUnit results](swarm-test-results.xml).
